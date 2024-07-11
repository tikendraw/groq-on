from .async_api.agroq_server import AgroqServer
from .async_api.schema import AgroqServerConfig, AgroqClientConfig
from .async_api.agroq_client import AgroqClient
from .groq_config import GROQON_CONFIG_FILE, modelindex
from .utils import save_config, load_config
from pathlib import Path
import click
import asyncio


config = load_config(config_file=GROQON_CONFIG_FILE.absolute())
DEFAULTS = config.get("defaults")


@click.group()
@click.version_option()
@click.pass_context
def one(ctx: click.Context):
    """ Start Groqon CLI """
    ctx.ensure_object(dict)
    ctx.obj['config']=config
    ctx.obj['defaults']=DEFAULTS
    ctx.obj['modelindex']=modelindex
    ctx.obj['server_running']=0
    
    if ctx.obj['server_running']==1:
        print('server running')

@click.command()
@click.argument("query", type=str)
@click.pass_context
def query(ctx:click.Context, query:str):
    config = AgroqClientConfig(
        **ctx.obj['defaults']
    )
    client = AgroqClient(config)
    asyncio.run(client.multi_query_async(**ctx.obj['query']))



def run_server(ctx):
    config = AgroqServerConfig(
        **ctx.obj['defaults']
    )
    server = AgroqServer(config)
    
    async def main():
        await server.astart()
        ctx.obj['server_running']=1
                
    asyncio.run(main())

    
@click.command()
@click.pass_context
def serve(ctx: click.Context):
    run_server(ctx)

@click.command()
@click.option("--cookie_file",          "-cf", default=DEFAULTS.get("cookie_file"), type=click.Path(exists=True),   help="Set cookie file, provide path to cookie file path, default is ~/.groqon/groq_cookie.json.")
@click.option("--models",               "-m",  default=','.join(DEFAULTS.get("models")),       type=str,                help=f"Comma separated(no spaces) list of models to be used, pick from {modelindex}(pick llms only).")
@click.option("--headless",             "-hl", default=DEFAULTS.get("headless"),     type=bool,                     help="set False to see the browser window, (you may not want to see)")
@click.option("--n_workers",            "-w",  default=DEFAULTS.get("n_workers"),    type=int,                      help="number of windows to serve as query workers. (more windows==more memory usage==faster query if multiple queries are running)")
@click.option("--reset_login",          "-rl", default=DEFAULTS.get("reset_login"),  type=bool,                     help="Reset the login information in cookie file, you have to login again when window opens")
@click.option("--server_model_configs", "-smc",default=DEFAULTS.get("server_model_configs"), type=click.Path(exists=True),  help="WARNING: server model config file, No not change this, NOT FOR USER.")
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
    """Configuration options."""
    
    print(models)
    print(type(models))

    if isinstance(models, str):
        models = models.split(',')
        print('model is string')
    
        
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
