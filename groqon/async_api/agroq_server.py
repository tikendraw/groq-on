import asyncio
import json
from asyncio import Queue, QueueEmpty
from typing import Any, Dict, List

from icecream import ic
from line_profiler import profile
from playwright.async_api import (
    BrowserContext,
    Page,
    Request,
    Response,
    Route,
    async_playwright,
)
from termcolor import colored

from ..element_selectors import (
    CHAT_INPUT_SELECTOR,
    QUERY_SUBMIT_SELECTOR,
    SIGNIN_TO_GROQ_BUTTON,
)
from ..groq_config import (
    ENDTOKEN,
    MODEL_LIST_FILE,
    PORT,
    URL,
    groq_config_folder,
    modelindex,
)
from ..logger import get_logger
from ..utils import cc, log_function_call
from .agroq_utils import (
    file_exists, 
    get_cookie,
    now,
    save_cookie,
    write_dict_to_json,
)

from .schema import AgroqServerConfig, APIRequestModel, APIResponseModel, ChoiceModel, ErrorResponse, MessageModel, RequestModel, ResponseModel, Xgroq
from pydantic import ValidationError
import time

logger = get_logger(__name__)
ic.disable()




class AgroqServer:
    def __init__(self, config: AgroqServerConfig):
        self.url=URL
        self.config = config
        self.model_index = modelindex
        self.running = True
        self._PORT = PORT
        self.n_request = 0
        self.set_worker_attrs(self.config.n_workers)
        self.got_models = False
        self.response_queues = {i: Queue() for i in self.worker_ids}
        self.request_queue = Queue()
        self.output_queue = Queue()
        self.stop_event = asyncio.Event()
        
    def set_worker_attrs(self, n_workers):
        self.worker_ids = [f'worker_{i}' for i in range(self.config.n_workers)]
        self.workers_dict = {worker_id: None for worker_id in self.worker_ids}
        self.worker_score = {worker_id: 0 for worker_id in self.worker_ids}
        

    # @log_function_call
    @profile
    async def astart(self) -> None:
        logger.debug('Agroq server starting...')
        self.running = True
        try:
            cookie = await self.get_cookie_or_login(self.url, self.config.cookie_file)
            
            
            async with async_playwright() as p:
                browser = await p.firefox.launch(headless=self.config.headless)
                context = await browser.new_context()
                await context.add_cookies(cookie)

                for worker_id in self.worker_ids:
                    worker = asyncio.create_task(self.worker_task(context, worker_id=worker_id, queue=self.response_queues[worker_id]))
                    self.workers_dict[worker_id] = worker

                self.server = await asyncio.start_server(self.handle_server_request, '127.0.0.1', self._PORT)

                addrs = ', '.join(str(sock.getsockname()) for sock in self.server.sockets)
                logger.debug(f'Serving on {addrs}')

                async with self.server:
                    try:
                        await self.server.serve_forever()
                    except KeyboardInterrupt as e:
                        logger.exception('KeyboardInterrupt: Exiting...', exc_info=e)
                    except asyncio.CancelledError as e:
                        logger.exception('asyncio.CancelledError: Exiting...', exc_info=e)
                    
        except Exception as e:
            logger.exception("Error in astart", exc_info=e)
        finally:
            self.astop()
            return None
                

    # @log_function_call
    @profile
    async def handle_server_request(self, reader, writer):
        data = await reader.read(99999) # TODO: find something that gets all the incoming data


        logger.debug(f'received request: server side : {data.decode()}')

        # # Closing if endtoken is received
        if ENDTOKEN in data.decode() and self.request_queue.empty():
            
            # cc('Found endtoken. Exiting...', 'white', 'on_red')
            self.running = False
            # self.stop_event.set()
            
            await self.astop()
            writer.close()
            await writer.wait_closed()
            return
        
        mm_dict = self.byte_to_model_kwargs(data)
        mm = RequestModel(**mm_dict)

        self.request_queue.put_nowait(mm)
        self.n_request +=1

        # do not put anything above the code: it passes without any output
        response =  await self.output_queue.get()
                
        if response:
            logger.debug(f'sending response: server side : {response.model_dump()}')
            response_byte = self.model_to_byte(response)            
            writer.write(response_byte)
            await writer.drain()

        writer.close()
        await writer.wait_closed()

    # @log_function_call
    @profile
    async def astop(self):
        self.running = False

        # Cancel all worker tasks
        for worker_id, worker in self.workers_dict.items():
            if worker:
                worker.cancel()
                try:
                    await worker
                except asyncio.CancelledError:
                    logger.error(f"Worker {worker_id} cancelled.")
                except Exception as e:
                    logger.error(f"Error while cancelling worker {worker_id}: {e}")

        # Cancel server task if it exists
        if hasattr(self, 'server'):
            self.server.close()

        # Cancel any remaining tasks and clear queues
        self.workers_dict.clear()
        self.reset()
            
    
    def reset(self):
        self.request_queue = Queue()
        self.response_queues = {i: Queue() for i in self.worker_ids}
        self.output_queue = Queue()
        


    # @log_function_call
    @profile        
    async def worker_task(self, context: BrowserContext, worker_id: str, queue: Queue) -> None:
        page = await context.new_page()
        await self.setup_page(page)

        while self.running:
            try:
                try:
                    request_model = await self.request_queue.get()
                except QueueEmpty:
                    continue

                if request_model:
                    # cc(f'{worker_id} is processing request:({request_model.model}):{request_model.query}', 'red', 'on_black')

                    async def wrapper_handle_chat_completions(route: Route, request: Request):
                        await self.handle_chat_completions(route, request, request_model, queue)

                    await page.route("**/**/chat/completions", wrapper_handle_chat_completions)
                    await self.do_query(page, request_model, queue=queue)
                    self.request_queue.task_done()
                    self.n_request -= 1
                    # cc(f'n request: {self.n_request}', 'yellow', 'on_black')
                    self.worker_score[worker_id] = self.worker_score.get(worker_id, 0) + 1

                    # cc(self.worker_score, 'blue', 'on_white')

                    # cc('waiting after entering query response', 'red', 'on_white', ['bold', 'blink'])
                    response = await queue.get()
                    
                    if response:                       
                        self.output_queue.put_nowait(response)
                    queue.task_done()
                    
                    
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.exception(f"Error in worker: {e}")
                break

        await page.close()
        return None
    
        
    # @log_function_call
    @profile
    async def do_query(self, page: Page, request_model: RequestModel, queue: Queue) -> None:
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
                    raise TimeoutError(f"Chat is temporarily disabled, Query: ({request_model.query}) not submitted.")
                    break
                
            if not textarea_is_disabled:
                await textarea.fill(request_model.query)
                await page.locator(QUERY_SUBMIT_SELECTOR).click()

        except Exception as e:
            logger.exception("Exception occurred: func - do_query ", exc_info=e)
            await queue.put(
                ResponseModel(
                    id=request_model.id,
                    query=request_model.query,
                    response_text="chat is disabled, no query submitted",
                    model=request_model.model
                )
            )
            
    async def is_signedin(self, page):
        signin_button = await page.wait_for_selector(SIGNIN_TO_GROQ_BUTTON)
        text = await signin_button.inner_text()
        if 'sign in' in text.lower():
            return False
        else:
            return True
        
        
    # @log_function_call
    @profile
    async def setup_page(self, page: Page):
        # await page.route("**/**/*.woff2", lambda x:x.abort())
        # await page.route("**/**/*.woff", lambda x:x.abort())
        # await page.route("**/**/*.css", lambda x:x.abort())
        # await page.route("**/**/web/metrics", lambda x:x.abort())
        await page.goto(self.url, timeout=60 * 1000)

    # @log_function_call
    @profile        
    async def handle_chat_completions(self, route: Route, request:Request, request_model: RequestModel, queue: Queue) -> None:
        request = route.request

        if request.method == "POST":
            # cc('method is post', 'blue')
            data = request.post_data_json
            
            if data:
                request_model.add_message(
                    content=request_model.system_prompt,
                    role="system",
                )
                
                request_model.add_message(
                    content=request_model.query,
                    role="user",
                )
                
                api_payload = APIRequestModel.from_request_model(request_model)
                api_payload_dict = api_payload.model_dump_json()

                # cc(api_payload_dict, 'green')
                await route.continue_(
                    post_data=api_payload_dict,
                )
            else:
                route.continue_()

        else:
            # cc('method is not post', 'red')
            route.continue_()
            
        response = await route.fetch()
        response = await self.handle_streamed_response(response, request_model.query)

        await queue.put(response)


    async def handle_streamed_response(self, response: Response, query: str) -> APIResponseModel:
        """Handle both streamed and non-streamed responses from the server."""
        status_code = response.status
        body_bytes = await response.body()
        body_str = body_bytes.decode("utf-8")

        try:
            # Check if it's a non-streamed response
            try:
                non_streamed_data = json.loads(body_str)
                if "choices" in non_streamed_data:
                    return self.process_non_streamed_response(non_streamed_data)
            except json.JSONDecodeError:
                pass  # Not a valid JSON, proceed with streamed response handling

            # Handle streamed response
            accumulated_content = ""
            created = int(time.time())
            model = ""
            usage = {}
            system_fingerprint = ""
            x_groq_id = ""

            lines = body_str.split("\n\n")
            for line in lines:
                if line.strip() == "data: [DONE]":
                    break
                if line.startswith("data: "):
                    json_content = line[len("data: "):]
                    try:
                        chunk = json.loads(json_content)
                        accumulated_content += chunk["choices"][0]["delta"].get("content", "")
                        model = chunk.get("model", model)
                        created = chunk.get("created", created)
                        system_fingerprint = chunk.get("system_fingerprint", system_fingerprint)
                        if "x_groq" in chunk:
                            x_groq_id = chunk["x_groq"].get("id", x_groq_id)
                            usage = chunk["x_groq"].get("usage", usage)
                    except json.JSONDecodeError:
                        logger.exception("Failed to decode JSON")
                elif line.startswith('{"error"'):
                    return self.process_error_response(line, status_code=status_code)

            return APIResponseModel(
                id=f"chatcmpl-{created}",
                object="chat.completion",
                created=created,
                model=model,
                choices=[
                    ChoiceModel(
                        index=0,
                        message=MessageModel(role="assistant", content=accumulated_content),
                        finish_reason="stop"
                    )
                ],
                usage=usage,
                system_fingerprint=system_fingerprint,
                x_groq=Xgroq(id=x_groq_id)
            )

        except Exception as e:
            logger.exception("Exception in handle_streamed_response", exc_info=e)
            return self.create_error_api_response(str(e))

    def process_non_streamed_response(self, data: Dict[str, Any]) -> APIResponseModel:
        """Process a non-streamed response."""
        return APIResponseModel(**data)

    def process_error_response(self, error_line: str, status_code: int = 500) -> APIResponseModel:
        """Process an error response using the ErrorResponse Pydantic model."""
        try:
            error_data = json.loads(error_line)
            error_response = ErrorResponse(**error_data)
            return self.create_error_api_response(error_response=error_response, status_code=status_code)
        except (json.JSONDecodeError, ValidationError) as e:
            logger.error(f"Failed to parse error response: {error_line}", exc_info=e)
            return self.create_error_api_response(error_message=f"An unknown error occurred: {error_line}")

    def create_error_api_response(self,error_message:str, error_code:str = 'Unknown error', error_response: ErrorResponse=None, status_code: int = 500) -> APIResponseModel:
        """Create an APIResponseModel instance for error responses."""
        created = int(time.time())
        return APIResponseModel(
            id=f"error-{created}",
            object="chat.completion",
            created=created,
            model="error",
            choices=[
                ChoiceModel(
                    index=0,
                    message=MessageModel(role="assistant", content=error_response.model_dump_json() or error_message),
                    finish_reason=f"error (status_code={status_code}, error_code={error_response.error.code or error_code})"
                )
            ],
            usage={'status_code': status_code},
            system_fingerprint="error",
            x_groq=Xgroq(id=f"error-{error_response.error.code or error_code}")
        )

    @profile
    async def get_models_from_api_response(self, route: Route, *args, **kwargs):
        response = await route.fetch()
        json_response = await response.json()

        if not self.got_models:
            await write_dict_to_json(json_response, MODEL_LIST_FILE.parent.absolute(), MODEL_LIST_FILE.name)
            self.got_models = True
        await route.continue_()
        

    @profile
    def model_to_byte(self, model: str) -> bytes:
        return model.model_dump_json().encode()
    
    @profile
    def byte_to_model_kwargs(self, byte: bytes) -> Dict:
        return json.loads(byte.decode())    
     
    # @log_function_call
    @profile
    async def get_cookie_or_login(self, url: str, cookie_file: str) -> List[Dict]:
        cookie = get_cookie(cookie_file) if file_exists(cookie_file) else None
        if not cookie:
            logger.warning(
                f"""
    Cookie not loaded!!!
    LOGIN REQUIRED!!!
    You have 100 seconds to login to groq.com, Make it quick!!! 
    page will close in 100 seconds, Your time started: {now()}
    .""",
            )
            cookie = await self.login_user()
            print('type fo coookie',type(cookie))
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

            sec = 100
            while sec > 0:
                await page.wait_for_timeout(1000)
                sec -= 1
                
                # TODO: break early if logged in or timeout
                # if "groq.com" in page.url:
                #     if await self.is_signedin(page):
                #         sec = 0

            cookie = await context.cookies()
            logger.debug("login page closed!")
            await page.close()
        return cookie


        
def write_str_to_file(context, file_name):
    with open(file_name, 'a') as f:
        f.write(context)
        f.write('\n')
        

