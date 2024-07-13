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
    ctx.obj['server_running'] = False


@click.command()
@click.argument("query", type=str, nargs=-1)
@click.option("--save_dir",     "-s", type=str, default=None,  help="file path to save the generated response file, not saved if not provided")
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
    
    config = AgroqClientConfig(
        models = models,
        save_dir = save_dir,
        system_prompt = system_prompt,
        print_output = print_output,
        temperature = temperature,
        max_tokens = max_tokens,
        top_p = top_p,
        stream = stream,
        stop_server = stop_server,
    )
    client = AgroqClient(config=config)
    asyncio.run(client.multi_query_async(query=query))




def run_server(**kwargs):
    config = AgroqServerConfig(**kwargs)
    server = AgroqServer(config)
    
    async def main():
        await server.astart()
                
    asyncio.run(main())

    
@click.command()
@click.pass_context
@click.option("--cookie_file",          "-cf", default=DEFAULTS.get("cookie_file"), type=click.Path(exists=True),   help="Set cookie file, provide path to cookie file path, default is ~/.groqon/groq_cookie.json.")
@click.option("--models",               "-m",  default=','.join(DEFAULTS.get("models")),       type=str,                help=f"Comma separated(no spaces) list of models to be used, pick from {modelindex}(pick llms only).")
@click.option("--headless",             "-hl", default=DEFAULTS.get("headless"),     type=bool,                     help="set False to see the browser window, (you may not want to see)")
@click.option("--n_workers",            "-w",  default=DEFAULTS.get("n_workers"),    type=int,                      help="number of windows to serve as query workers. (more windows==more memory usage==faster query if multiple queries are running)")
@click.option("--reset_login",          "-rl", default=DEFAULTS.get("reset_login"),  type=bool,                     help="Reset the login information in cookie file, you have to login again when window opens")
@click.option("--verbose",              "-v",  default=DEFAULTS.get("verbose"),        type=bool,                   help="Set True to see verbose output")
def serve(
    ctx: click.Context,
    cookie_file:Path,          
    models:str|list[str],
    headless:bool,             
    n_workers:int,            
    reset_login:bool,          
    verbose:bool,              
):
    
    if isinstance(models, str):
        models = models.split(',')

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
def stop_server(ctx: click.Context):
    if not ctx.obj['server_running']:
        click.echo("Server is not running.")
        return
    
    async def stop():
        server = ctx.obj['server']
        await server.astop()
        ctx.obj['server_running'] = False
        click.echo("Server stopped successfully.")
    
    asyncio.run(stop())
    
    
@click.command()
@click.option("--cookie_file",          "-cf", default=DEFAULTS.get("cookie_file"), type=click.Path(exists=True),   help="Set cookie file, provide path to cookie file path, default is ~/.groqon/groq_cookie.json.")
@click.option("--models",               "-m",  default=','.join(DEFAULTS.get("models")),       type=str,                help=f"Comma separated(no spaces) list of models to be used, pick from {modelindex}(pick llms only).")
@click.option("--headless",             "-hl", default=DEFAULTS.get("headless"),     type=bool,                     help="set False to see the browser window, (you may not want to see)")
@click.option("--n_workers",            "-w",  default=DEFAULTS.get("n_workers"),    type=int,                      help="number of windows to serve as query workers. (more windows==more memory usage==faster query if multiple queries are running)")
@click.option("--reset_login",          "-rl", default=DEFAULTS.get("reset_login"),  type=bool,                     help="Reset the login information in cookie file, you have to login again when window opens")
@click.option("--verbose",              "-v",  default=DEFAULTS.get("verbose"),        type=bool,                   help="Set True to see verbose output")
@click.option("--print_output",         "-p",  default=DEFAULTS.get("print_output"),   type=bool,                   help="Prints output to the terminal")
def config(cookie_file:Path          ,#= DEFAULTS.get("cookie_file"),
           models:str|list[str]      ,#= DEFAULTS.get("models"),
           headless:bool             ,#= DEFAULTS.get("headless"),
           n_workers:int             ,#= DEFAULTS.get("n_workers"),
           reset_login:bool          ,#= DEFAULTS.get("reset_login"),
           server_model_configs:Path ,#= DEFAULTS.get("server_model_configs"),
           verbose:bool              ,#= DEFAULTS.get("verbose"),
           print_output:bool         ,#= DEFAULTS.get("print_output"),
           ):
    """saves Configuration options."""
    
    if isinstance(models, str):
        models = models.split(',')
    
    config = {
        "defaults": {
            "cookie_file" : cookie_file,
            "models" : models,
            "headless" : headless,
            "n_workers" : n_workers,
            "reset_login" : reset_login,
            "server_model_configs" : server_model_configs,
            "verbose" : verbose,
            "print_output" : print_output,
        }
    }
    
    save_config(config, GROQON_CONFIG_FILE)
    print(config)


one.add_command(config)
one.add_command(serve)
one.add_command(query)

