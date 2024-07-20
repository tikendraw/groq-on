import asyncio
import json
import os
import time
from asyncio import Queue, QueueEmpty
from typing import Any, Dict, List
from urllib.parse import parse_qs

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
from .agroq_utils import (
    file_exists,
    get_cookie,
    now,
    save_cookie,
    write_dict_to_json,
)
from .schema import (
    AgroqServerConfig,
    APIRequestModel,
    APIResponseModel,
    ChoiceModel,
    ErrorModel,
    ErrorResponse,
    MessageModel,
    Xgroq,
    set_local_id_and_query,
)

logger = get_logger(__name__)
ic.disable()


class EndTokenFoundError(Exception):
    pass 


class AgroqServer:
    def __init__(self, config: AgroqServerConfig):
        self.url=URL
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
                logger.info(f'Serving on {addrs}')
                
                async with self.server:
                    # Run server and wait for stop event
                    await asyncio.gather(
                        self.server.serve_forever(),
                        self.wait_for_stop()
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
        
    # @log_function_call
    @profile
    async def handle_server_request(self, reader, writer):
        data = await reader.read(99999) # TODO: find something that gets all the incoming data
        is_http=False
        
        print('requests in queue: ',self.request_queue.qsize())
        logger.info(f'>>>>>>>> request: {data.decode()}')
        # Check if it's a regular asyncio request or a HTTP request
        if data.startswith(b'POST') or data.startswith(b'GET'):
            data = self.parse_http_request(data.decode())
            data = self.dict_to_byte(data)
            is_http =True
            
        # # Closing if endtoken is received
        if ENDTOKEN in data.decode() and self.request_queue.empty():
            
            # cc('Found endtoken. Exiting...', 'white', 'on_red')
            writer.write(b"Server stopped")
            await writer.drain()
            writer.close()
            await writer.wait_closed()
            # cc('Groqon Server stopped.', 'white', 'on_red')
            self.shutdown_event.set()  # Signal the main loop to stop

            return
        
        mm_dict = self.byte_to_model_kwargs(data)
        mm = APIRequestModel(**mm_dict)

        self.request_queue.put_nowait(mm)
        self.n_request +=1

        # do not put anything above the code: it passes without any output
        response =  await self.output_queue.get()
        
        if response:
            logger.info(f'<<<<<<< response: server side : {response.model_dump()}')
            await self.send_response(writer, response, is_http)

        await writer.wait_closed()
        
    async def send_response(self, writer, response, is_http: bool):
        if is_http:
            response_json = response.model_dump_json()
            http_response = (
                b"HTTP/1.1 200 OK\r\n"
                b"Content-Type: application/json\r\n"
                b"Content-Length: " + str(len(response_json)).encode() + b"\r\n"
                b"Content-Disposition: inline\r\n"  # This tells the client to display the content inline
                b"X-Content-Type-Options: nosniff\r\n"  # This prevents MIME type sniffing
                b"\r\n" + response_json.encode()
            )
            writer.write(http_response)
        else:
            # For regular asyncio requests, send the response as before
            response_byte = self.model_to_byte(response)
            writer.write(response_byte)
        
        await writer.drain()
        writer.close()
        

    def parse_http_request(self, request_str: str) -> Dict:
        # Split the request into headers and body
        headers, _, body = request_str.partition('\r\n\r\n')
        
        # Parse the request line
        request_line = headers.split('\r\n')[0]
        method, path, _ = request_line.split(' ')

        # Parse query parameters if it's a GET request
        if method == 'GET':
            query_params = parse_qs(path.split('?')[1]) if '?' in path else {}
            return {k: v[0] for k, v in query_params.items()}
        
        # For POST requests, parse the body
        elif method == 'POST':
            content_type = next((line.split(': ')[1] for line in headers.split('\r\n') if line.startswith('Content-Type:')), None)
            
            if content_type == 'application/json':
                return json.loads(body)
            elif content_type == 'application/x-www-form-urlencoded':
                return {k: v[0] for k, v in parse_qs(body).items()}
            else:
                # Handle other content types as needed
                return {}
        
        # Handle other HTTP methods as needed
        else:
            return {}
        
    # @log_function_call
    @profile
    async def astop(self):
        self.running = False

        # Create a copy of the workers_dict items
        workers = list(self.workers_dict.items())

        # Cancel all worker tasks
        for worker_id, worker in workers:
            if worker:
                worker.cancel()
                try:
                    await worker
                except asyncio.CancelledError:
                    logger.info(f"Worker {worker_id} cancelled successfully.")
                except Exception as e:
                    logger.error(f"Error while cancelling worker {worker_id}: {e}")


        self.server.close()
        await self.server.wait_closed()
            

        # Cancel any remaining tasks and clear queues
        self.workers_dict.clear()
        self.reset()
        logger.info("Groqon Server stopped.")

    
    def stop(self):
        asyncio.run(self.astop())
        
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
                    await self.do_query(page, query=request_model.get_query(), queue=queue)
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
    async def do_query(self, page: Page, query:str, queue: Queue) -> None:
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
                    raise TimeoutError(f"Chat is temporarily disabled, Query: ({query}) not submitted.")
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
                    status_code=500,
                    query=query
                )
                            )
                    
        
    # @log_function_call
    @profile
    async def setup_page(self, page: Page):
        # await page.route("**/**/*.woff2", lambda x:x.abort())
        # await page.route("**/**/*.woff", lambda x:x.abort())
        # await page.route("**/**/*.css", lambda x:x.abort())
        # await page.route("**/**/web/metrics", lambda x:x.abort())
        await page.route("https://api.groq.com/openai/v1/models", self.get_models_from_api_response)
        await page.goto(self.url, timeout=60 * 1000, wait_until='commit') # ["commit", "domcontentloaded", "load", "networkidle"]

    # @log_function_call
    @profile        
    async def handle_chat_completions(self, route: Route, request:Request, api_request_model: APIRequestModel, queue: Queue) -> None:
        request = route.request

        if request.method == "POST":
            # cc('method is post', 'blue')
            data = request.post_data_json
            
            if data:                
                api_payload_dict = api_request_model.model_dump_json()

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
        response = await self.handle_streamed_response(response, api_request_model.query)
        response = set_local_id_and_query(api_request=api_request_model, api_response=response)
        await queue.put(response)

    @profile        
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

                elif 'error' in non_streamed_data.keys():
                    logger.error(f"Error response received: for query {query} -{non_streamed_data}")
                    output = self.process_error_response(error_dict=non_streamed_data, status_code=status_code, query=query)
                    await write_dict_to_json(output.model_dump(), groq_error_folder, f"Error {query}.json")
                    return output
                
                
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

    @profile
    def process_non_streamed_response(self, data: Dict[str, Any]) -> APIResponseModel:
        """Process a non-streamed response."""
        return APIResponseModel(**data)

    @profile
    def process_error_response(self, error_line: str=None, error_dict:dict=None, status_code: int = 500, query:str=None) -> APIResponseModel:
        """Process an error response using the ErrorResponse Pydantic model."""
        try:
            if error_line and error_dict is None:
                error_dict = json.loads(error_line)
                            
            error_response = ErrorResponse(**error_dict)
            error = error_response.error
            return self.create_error_api_response(error_message=error.message,error_code=error.code, error_type=error.type, status_code=status_code, query=query)
        except (json.JSONDecodeError, ValidationError) as e:
            logger.error(f"Failed to parse error response: {error_line}", exc_info=e)
            return self.create_error_api_response(error_message=f"An unknown error occurred: {error_line}")

    @profile
    def create_error_api_response(self,error_message:str=None, error_code:str = 'Unknown error', error_type:str="unknown_type",query:str=None, status_code: int = 500) -> APIResponseModel:
        """Create an APIResponseModel instance for error responses."""
            
        return ErrorResponse(
            error=ErrorModel(
                message=error_message,
                code=error_code,
                type=error_type
            ),
            status_code=status_code,
            query=query
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
    
    def dict_to_byte(self, x:dict) -> bytes:
        return (json.dumps(x)).encode()
     
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
    Use Google Accounts to login not the other one.
    page will close in 100 seconds, Your time started: {now()}
    .""",
            )
            cookie = await self.login_user()
            await save_cookie(cookie, cookie_file)
        return cookie


                
    # # @log_function_call
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
                        
                        if name.strip().lower() not in  [SIGNIN_BUTTON_TEXT.strip().lower(), 'sign in ', 'signin']:
                            print('name : ',name)
                            break
                    except Exception as e:
                        logger.error('Exception while checking if logged in (Not a Problem)', exc_info=e)
                        pass
            cookie = await context.cookies()
            logger.debug("login page closed!")
            await page.close()
        return cookie

