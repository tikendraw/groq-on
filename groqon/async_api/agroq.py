import asyncio
import json
import os
import time
from asyncio import Queue, QueueEmpty
from typing import Any, Dict, List
from urllib.parse import parse_qs

from line_profiler import profile
from playwright.async_api import (
    BrowserContext,
    Page,
    Request,
    Response,
    Route,
    async_playwright,
)
from pydantic import ValidationError

from ..element_selectors import (
    CHAT_INPUT_SELECTOR,
    QUERY_SUBMIT_SELECTOR,
    SIGNIN_BUTTON_SELECTOR,
    SIGNIN_BUTTON_TEXT,
)
from ..groq_config import (
    ENDTOKEN,
    MODEL_LIST_FILE,
    PORT,
    URL,
    groq_error_folder,
    modelindex,
)
from ..logger import get_logger
from ..utils import cc, log_function_call
from ._utils import (
    file_exists,
    get_cookie,
    now,
    save_cookie,
    write_dict_to_json,
)
from .schema import (
    ChatCompletion,
    ChatCompletionChunk,
    ErrorModel,
    ErrorResponse,
    GROQ_APIRequest,
    GroqonConfig,
)

logger = get_logger(__name__)


class Groqon:
    def __init__(self, config: GroqonConfig):
        self.url = URL
        self.config = config

        if self.config.reset_login:
            os.remove(self.config.cookie_file)

        self.model_index = modelindex
        self.running = True
        self._PORT = PORT
        self.n_request = 0
        self.set_worker_attrs(self.config.n_workers)
        self.got_models = False
        self.response_queues = {i: Queue() for i in self.worker_ids}
        self.request_queue = Queue()
        self.output_queue = Queue()
        self.shutdown_event = asyncio.Event()
        self.browser_context = None
        self.worker_tasks = []

    def set_worker_attrs(self, n_workers):
        self.worker_ids = [f"worker_{i}" for i in range(self.config.n_workers)]
        self.workers_dict = {worker_id: None for worker_id in self.worker_ids}
        self.worker_score = {worker_id: 0 for worker_id in self.worker_ids}

    @log_function_call
    @profile
    async def astart(self) -> None:
        logger.debug("Groqon server starting...")
        self.running = True
        try:
            cookie = await self.get_cookie_or_login(self.url, self.config.cookie_file)

            async with async_playwright() as p:
                browser = await p.firefox.launch(headless=self.config.headless)
                self.browser_context = await browser.new_context()
                await self.browser_context.add_cookies(cookie)

                self.worker_tasks = [
                    asyncio.create_task(self.worker_task(worker_id))
                    for worker_id in self.worker_ids
                ]

                # self.server = await asyncio.start_server(
                #     self.handle_server_request, "127.0.0.1", self._PORT
                # )

                # addrs = ", ".join(
                #     str(sock.getsockname()) for sock in self.server.sockets
                # )
                # logger.info(f"Serving on {addrs}")

                # async with self.server:
                # Run server and wait for stop event
                await asyncio.gather(
                    # self.server.serve_forever(),
                    self.wait_for_stop(),
                    *self.worker_tasks,
                )

        except Exception as e:
            logger.exception("Error in astart", exc_info=e)
        finally:
            # await self.astop()
            return None

    def start(self):
        asyncio.run(self.astart())

    async def wait_for_stop(self):
        await self.shutdown_event.wait()
        # self.server.close()
        # await self.server.wait_closed()
        await self.astop()
        print("Server stopped.")

    # async def handle_server_request(self, reader, writer):
    #     intime = time.perf_counter_ns()
    #     data = await self.read_full_request(reader)
    #     logger.info(f">>>>>>>> request: {intime} -{data.decode()}")

    #     if ENDTOKEN in data.decode() and self.request_queue.empty():
    #         writer.write(b"Server stopped")
    #         await writer.drain()
    #         writer.close()
    #         await writer.wait_closed()
    #         self.shutdown_event.set()
    #         return

    #     parsed_data = (
    #         self.parse_http_request(data.decode())
    #         if data.startswith(b"POST") or data.startswith(b"GET")
    #         else None
    #     )
    #     if parsed_data:
    #         response = self.process_request(parsed_data)
    #         if response:
    #             outtime = time.perf_counter_ns()
    #             logger.info(
    #                 f"<<<<<< response: {outtime} took {(outtime - intime):,}ns -{response}"
    #             )
    #             await self.send_response(writer, response)
    #     else:
    #         writer.close()
    #         await writer.wait_closed()

    async def process_request(self, request: dict | GROQ_APIRequest):

        if ENDTOKEN in str(request) and self.request_queue.empty():
            self.shutdown_event.set()
            return

        await self.add_to_request_queue(request)
        response = await self.output_queue.get()
        return response

    async def read_full_request(self, reader):
        # Read headers
        headers = await reader.readuntil(b"\r\n\r\n")

        # If it's not an HTTP request, return the data as is
        if not headers.startswith(b"POST") and not headers.startswith(b"GET"):
            return headers

        # Parse Content-Length
        content_length = 0
        for line in headers.split(b"\r\n"):
            if line.lower().startswith(b"content-length:"):
                content_length = int(line.split(b":")[1].strip())
                break

        # Read body
        body = await reader.read(content_length)

        return headers + body

    def parse_http_request(self, request_str: str) -> Dict:
        # Split the request into headers and body
        headers, _, body = request_str.partition("\r\n\r\n")

        # Parse the request line
        request_line = headers.split("\r\n")[0]
        method, path, _ = request_line.split(" ")

        # Parse query parameters if it's a GET request
        if method == "GET":
            query_params = parse_qs(path.split("?")[1]) if "?" in path else {}
            return {k: v[0] for k, v in query_params.items()}

        # For POST requests, parse the body
        elif method == "POST":
            content_type = next(
                (
                    line.split(": ")[1]
                    for line in headers.split("\r\n")
                    if line.startswith("Content-Type:")
                ),
                None,
            )

            if content_type == "application/json":
                try:
                    return json.loads(body)
                except json.JSONDecodeError:
                    logger.error(f"Failed to parse JSON body: {body}")
                    return {}
            elif content_type == "application/x-www-form-urlencoded":
                return {k: v[0] for k, v in parse_qs(body).items()}
            else:
                return {}

        else:
            return {}

    async def add_to_request_queue(self, request: dict | GROQ_APIRequest):
        if isinstance(request, dict):
            try:
                request = GROQ_APIRequest(**request)
            except Exception as e:
                logger.exception(e, exc_info=True)
                raise e

        self.request_queue.put_nowait(request)
        self.n_request += 1

    async def get_all_response(self, local_ids: list):
        put_back_to_output_queue = []
        responses = []

        if not isinstance(local_ids, list):
            local_ids = [local_ids]

        while True:
            try:
                response = await self.output_queue.get()
                if response.local_id in local_ids:
                    responses.append(response)
                else:
                    put_back_to_output_queue.append(response)
            except QueueEmpty:
                break
            finally:
                self.output_queue.task_done()

        if put_back_to_output_queue:
            for i in put_back_to_output_queue:
                self.output_queue.put_nowait(i)

        return responses

    async def send_response(self, writer, response_data: dict | List[dict]):
        writer.write(
            b"HTTP/1.1 200 OK\r\n"
            b"Content-Type: text/event-stream\r\n"
            b"Cache-Control: no-cache\r\n"
            b"Connection: keep-alive\r\n"
            b"\r\n"
        )

        if isinstance(response_data, list):
            # Streaming response
            for i, chunk in enumerate(response_data):
                sse_message = f"data: {json.dumps(chunk)}\n"
                if i == len(response_data) - 1:
                    sse_message += "data: [DONE]\n"
                writer.write(sse_message.encode())
                await writer.drain()
        else:
            # Single response
            sse_message = json.dumps(response_data)
            writer.write(sse_message.encode())

        await writer.drain()
        writer.close()
        await writer.wait_closed()

    def dict_to_byte(self, x: dict) -> bytes:
        return (json.dumps(x)).encode()

    @log_function_call
    @profile
    async def astop(self):
        self.running = False

        # Create a copy of the workers_dict items

        # Cancel all worker tasks
        for worker in self.worker_tasks:
            if worker:
                worker.cancel()
                try:
                    await worker
                except asyncio.CancelledError:
                    logger.info("Worker cancelled successfully.")
                except Exception as e:
                    logger.error(f"Error while cancelling worker: {e}")

        # if hasattr(self, "server"):
        #     self.server.close()
        #     await self.server.wait_closed()

        # Cancel any remaining tasks and clear queues
        self.reset()
        logger.info("Groqon Server stopped.")

    def stop(self):
        asyncio.run(self.astop())

    def reset(self):
        self.request_queue = Queue()
        self.response_queues = {i: Queue() for i in self.worker_ids}
        self.output_queue = Queue()

    async def worker_task(self, worker_id: str) -> None:
        page = await self.browser_context.new_page()
        response_queue = self.response_queues[worker_id]
        await self.setup_page(page)

        while self.running:
            try:
                print("++ n requests: ", self.request_queue.qsize())
                request_model = await self.request_queue.get()

                if request_model:

                    async def wrapper_handle_chat_completions(
                        route: Route, request: Request
                    ):
                        await self.handle_chat_completions(
                            route, request, request_model, queue=response_queue
                        )

                    await page.route(
                        "**/**/chat/completions", wrapper_handle_chat_completions
                    )
                    await self.do_query(
                        page, query=request_model.get_query(), queue=response_queue
                    )
                    self.request_queue.task_done()
                    self.n_request -= 1

                    response = await response_queue.get()
                    if response:
                        await self.output_queue.put(response)
                    response_queue.task_done()

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.exception(f"Error in worker {worker_id}: {e}")

        await page.close()

    @log_function_call
    @profile
    async def do_query(self, page: Page, query: str, queue: Queue) -> None:
        try:
            textarea = await page.wait_for_selector(CHAT_INPUT_SELECTOR)
            n_try = 10
            textarea_is_disabled = await textarea.is_disabled()

            while textarea_is_disabled:
                if n_try > 0:
                    await page.wait_for_timeout(500)
                    textarea_is_disabled = await textarea.is_disabled()
                    # cc('chat is temp disabled', 'yellow')
                    n_try -= 1
                else:
                    raise TimeoutError(
                        f"Chat is temporarily disabled, Query: ({query}) not submitted."
                    )
                    break

            if not textarea_is_disabled:
                await textarea.fill(query)
                await page.locator(QUERY_SUBMIT_SELECTOR).click()

        except Exception as e:
            logger.exception("Exception occurred: func - do_query ", exc_info=e)
            await queue.put(
                self.create_error_api_response(
                    error_message=f"Chat is temporarily disabled, Query: ({query}) not submitted.",
                    error_code="timeout_error",
                    error_type="timeout_error",
                )
            )

    @log_function_call
    @profile
    async def setup_page(self, page: Page):
        await page.route("**/**/*.woff2", lambda x: x.abort())
        await page.route("**/**/*.woff", lambda x: x.abort())
        await page.route("**/**/*.css", lambda x: x.abort())
        # await page.route("**/**/web/metrics", lambda x:x.abort())
        await page.route(
            "https://api.groq.com/openai/v1/models", self.get_models_from_api_response
        )
        await page.goto(
            self.url, timeout=60 * 1000, wait_until="commit"
        )  # ["commit", "domcontentloaded", "load", "networkidle"]

    @log_function_call
    @profile
    async def handle_chat_completions(
        self,
        route: Route,
        request: Request,
        api_request_model: GROQ_APIRequest,
        queue: Queue,
    ) -> None:
        request = route.request

        if request.method == "POST":
            # cc('method is post', 'blue')
            data = request.post_data_json

            if data:
                api_payload_dict = api_request_model.model_dump_json()

                # cc(f">>>>>>>>>>> :: {api_payload_dict}", "green", "on_black")
                await route.continue_(
                    post_data=api_payload_dict,
                )
            else:
                route.continue_()

        else:
            # cc('method is not post', 'red')
            route.continue_()

        response = await route.fetch()
        response = await self.handle_streamed_response(
            response, api_request_model.get_query()
        )
        await queue.put(response)

    @profile
    async def handle_streamed_response(
        self, response: Response, query: str
    ) -> dict | list[dict]:
        """Handle both streamed and non-streamed responses from the server."""
        body_bytes = await response.body()
        body_str = body_bytes.decode("utf-8")

        try:
            # Check if it's a non-streamed response
            try:
                non_streamed_data = json.loads(body_str)

                if "error" in non_streamed_data.keys():
                    logger.error(
                        f"Error response received: for query {query} -{non_streamed_data}"
                    )
                    await write_dict_to_json(
                        non_streamed_data, groq_error_folder, f"Error {query}.json"
                    )
                return non_streamed_data

            except json.JSONDecodeError:
                pass  # Not a valid JSON, proceed with streamed response handling

            lines = body_str.split("\n\n")
            # stream_chunks = []
            # for line in lines:
            #     if line.strip() == "data: [DONE]":
            #         break
            #     if line.startswith("data: "):
            #         json_content = line.lstrip("data: ").strip()
            #         try:
            #             chunk = json.loads(json_content)
            #             stream_chunks.append(chunk)

            #         except json.JSONDecodeError:
            #             logger.exception("Failed to decode JSON")

            return lines

        except Exception as e:
            logger.exception("Exception in handle_streamed_response", exc_info=e)
            return self.create_error_api_response(str(e))

    @profile
    def process_non_streamed_response(self, data: Dict[str, Any]) -> ChatCompletion:
        """Process a non-streamed response."""
        return ChatCompletion(**data)

    @profile
    def process_streamed_response(self, data: Dict[str, Any]) -> ChatCompletionChunk:
        return ChatCompletionChunk(**data)

    def create_error_api_response(
        self,
        error_message: str = None,
        error_code: str = "Unknown error",
        error_type: str = "unknown_type",
    ) -> ErrorResponse:
        """Create an ErrorResponse instance for error responses."""

        return ErrorResponse(
            error=ErrorModel(message=error_message, code=error_code, type=error_type)
        )

    @profile
    async def get_models_from_api_response(self, route: Route, *args, **kwargs):
        response = await route.fetch()
        json_response = await response.json()

        if not self.got_models:
            await write_dict_to_json(
                json_response, MODEL_LIST_FILE.parent.absolute(), MODEL_LIST_FILE.name
            )
            self.got_models = True
        await route.continue_()

    @profile
    def byte_to_model_kwargs(self, byte: bytes) -> Dict:
        return json.loads(byte.decode())

    @log_function_call
    @profile
    async def get_cookie_or_login(self, url: str, cookie_file: str) -> List[Dict]:
        cookie = get_cookie(cookie_file) if file_exists(cookie_file) else None
        if not cookie:
            logger.warning(
                f"""
    Cookie not loaded!!!
    LOGIN REQUIRED!!!
    You have 100 seconds to login to groq.com, Make it quick!!! 
    Use Google Accounts to login not the other one.
    page will close in 100 seconds, Your time started: {now()}
    .""",
            )
            cookie = await self.login_user()
            await save_cookie(cookie, cookie_file)
        return cookie

    # @log_function_call
    @profile
    async def login_user(self) -> list:
        async with async_playwright() as p:
            browser = await p.firefox.launch(headless=False)
            context = await browser.new_context()
            page = await context.new_page()
            await page.goto(self.url, timeout=60_000)

            max_wait_time = 100  # seconds
            start_time = time.time()

            while time.time() - start_time < max_wait_time:
                await page.wait_for_timeout(1_000)

                # break early if logged in
                if "groq.com" in page.url:
                    try:
                        response = await page.wait_for_selector(SIGNIN_BUTTON_SELECTOR)
                        name = await response.inner_text()

                        if name.strip().lower() not in [
                            SIGNIN_BUTTON_TEXT.strip().lower(),
                            "sign in ",
                            "signin",
                        ]:
                            print("name : ", name)
                            break
                    except:  # noqa
                        pass
            cookie = await context.cookies()
            logger.debug("login page closed!")
            await page.close()
        return cookie
