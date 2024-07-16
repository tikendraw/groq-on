import asyncio
from pathlib import Path

import click

from .async_api.agroq_client import AgroqClient
from .async_api.agroq_server import AgroqServer
from .async_api.schema import AgroqClientConfig, AgroqServerConfig
from .groq_config import (
    GROQON_CONFIG_FILE,
    modelindex,
)
from .utils import load_config, save_config

config = load_config(config_file=GROQON_CONFIG_FILE.absolute())
CLIENT_CONFIG = config.get("client")
DEFAULTS = config.get("defaults")


@click.group()
@click.version_option()
@click.pass_context
def one(ctx: click.Context):
    """ Start Groqon CLI """
    ctx.ensure_object(dict)
    ctx.obj['config'] = config
    ctx.obj['defaults'] = DEFAULTS
    ctx.obj['modelindex'] = modelindex


@click.command()
@click.argument("query", type=str, nargs=-1)
@click.option("--save_dir",     "-o",   default=None,                              type=str,  help="file path to save the generated response file, not saved if not provided")
@click.option("--models",       "-m",   default=','.join(DEFAULTS.get("models")),  type=str,  help=f"Comma separated(no spaces) list of models to be used, pick from {modelindex}(pick llms only).")
@click.option("--system_prompt","-sp",  default=CLIENT_CONFIG.get("system_prompt"),type=str,  help="set False to see the browser window, (you may not want to see)")
@click.option("--print_output", "-p",   default=True,                              type=bool, help="number of windows to serve as query workers. (more windows==more memory usage==faster query if multiple queries are running)")
@click.option("--temperature",  "-t",   default=CLIENT_CONFIG.get("temperature"),  type=float,help="Reset the login information in cookie file, you have to login again when window opens")
@click.option("--max_tokens",   "-mt",  default=CLIENT_CONFIG.get("max_tokens"),   type=int,  help="WARNING: server model config file, No not change this, NOT FOR USER.")
@click.option("--stream",       "-s",   default=CLIENT_CONFIG.get("stream"),       type=bool, help="Set True to see verbose output")
@click.option("--top_p",        "-tp",  default=CLIENT_CONFIG.get("top_p"),        type=int,  help="Set True to see verbose output")
@click.option("--stop_server",  "-ss",  default=False,                             type=bool, help="Stop server after query")
@click.pass_context
def query(
    ctx:click.Context, 
    query:str,
    save_dir:Path,
    models:str|list[str],
    system_prompt:str,
    print_output:bool,
    temperature:float,
    max_tokens:int,
    stream:bool,
    top_p:int,
    stop_server:bool
    ):
    
    
    if isinstance(models, str):
        models = models.split(',')
    
    make_calls(
        models = models,
        save_dir = save_dir,
        system_prompt = system_prompt,
        print_output = print_output,
        temperature = temperature,
        max_tokens = max_tokens,
        top_p = top_p,
        stream = stream,
        stop_server = stop_server,
        query=query
    )

def make_calls(**kwargs):
    query = kwargs.pop('query')
    stop_server=kwargs.pop('stop_server') or False
    config = AgroqClientConfig(**kwargs)
    client = AgroqClient(config=config)
    asyncio.run(client.multi_query_async(query=query, stop_server=stop_server))

def run_server(**kwargs):
    config = AgroqServerConfig(**kwargs)
    server = AgroqServer(config)
    
    async def main():
        try:
            await server.astart()
        except Exception as e:
            print("Error in server execution: ", e)
    asyncio.run(main())

    
@click.command()
@click.pass_context
@click.option("--cookie_file", "-cf", default=DEFAULTS.get("cookie_file"),      type=click.Path(),   help="Set cookie file, provide path to cookie file path, default is ~/.groqon/groq_cookie.json.")
@click.option("--models",      "-m",  default=','.join(DEFAULTS.get("models")), type=str,                       help=f"Comma separated(no spaces) list of models to be used, pick from {modelindex}(pick llms only).")
@click.option("--headless",    "-hl", default=DEFAULTS.get("headless"),         type=bool,                      help="set False to see the browser window, (you may not want to see)")
@click.option("--n_workers",   "-w",  default=DEFAULTS.get("n_workers"),        type=int,                       help="number of windows to serve as query workers. (more windows==more memory usage==faster query if multiple queries are running)")
@click.option("--reset_login", "-rl", default=DEFAULTS.get("reset_login"),      type=bool,                      help="Reset the login information in cookie file, you have to login again when window opens")
@click.option("--verbose",     "-v",  default=DEFAULTS.get("verbose"),          type=bool,                      help="Set True to see verbose output")
def serve(
    ctx: click.Context,
    cookie_file:Path,          
    models:str|list[str],
    headless:bool,             
    n_workers:int,            
    reset_login:bool,          
    verbose:bool,              
):
    """starts server and listens for queries"""
    
    if isinstance(models, str):
        models = models.split(',')
        
    if cookie_file:
        cookie_file_path = Path(cookie_file)
        if not cookie_file_path.exists():
            cookie_file_path.write_text('{}')

    run_server(
        cookie_file = cookie_file,
        models = models,
        headless = headless,
        n_workers = n_workers,
        reset_login = reset_login,
        verbose = verbose,
    )
    


@click.command()
@click.pass_context
def stop(ctx: click.Context):
    """stops server"""
    make_calls(
        stop_server=True,
        models          = ctx.obj['modelindex'], 
        save_dir        =None,
        system_prompt   ='you are a good boy',
        print_output    =False,
        temperature     =0.1,
        max_tokens      =2024,
        top_p           =1,
        stream          =True,
        query           =None,
    )    
     
@click.command()
@click.option("--cookie_file",  "-cf", default=DEFAULTS.get("cookie_file"),     type=click.Path(),   help="Set cookie file, provide path to cookie file path, default is ~/.groqon/groq_cookie.json.")
@click.option("--models",       "-m",  default=','.join(DEFAULTS.get("models")),type=str,                       help=f"Comma separated(no spaces) list of models to be used, pick from {modelindex}(pick llms only).")
@click.option("--headless",     "-hl", default=DEFAULTS.get("headless"),        type=bool,                      help="set False to see the browser window, (you may not want to see)")
@click.option("--n_workers",    "-w",  default=DEFAULTS.get("n_workers"),       type=int,                       help="number of windows to serve as query workers. (more windows==more memory usage==faster query if multiple queries are running)")
@click.option("--reset_login",  "-rl", default=DEFAULTS.get("reset_login"),     type=bool,                      help="Reset the login information in cookie file, you have to login again when window opens")
@click.option("--verbose",      "-v",  default=DEFAULTS.get("verbose"),         type=bool,                      help="Set True to see verbose output")
@click.option("--print_output", "-p",  default=DEFAULTS.get("print_output"),    type=bool,                      help="Prints output to the terminal")
def config(
    cookie_file:Path,
    models:str|list[str],
    headless:bool,
    n_workers:int,
    reset_login:bool,
    verbose:bool,
    print_output:bool,
    ):
    """saves Configuration options."""
    
    if isinstance(models, str):
        models = models.split(',')
    
    if cookie_file:
        cookie_file_path = Path(cookie_file)
        if not cookie_file_path.exists():
            cookie_file_path.write_text('{}')
    
    config = {
        "defaults": {
            "cookie_file" : cookie_file,
            "models" : models,
            "headless" : headless,
            "n_workers" : n_workers,
            "reset_login" : reset_login,
            "verbose" : verbose,
            "print_output" : print_output,
        }
    }
    
    save_config(config, GROQON_CONFIG_FILE)
    print(config)


one.add_command(config)
one.add_command(serve)
one.add_command(query)
one.add_command(stop)

