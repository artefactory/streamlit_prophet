# type: ignore[attr-defined]

import typer
from rich.console import Console
from streamlit_prophet import __version__
from streamlit_prophet.cli import deploy

app = typer.Typer(
    name="streamlit_prophet",
    help="`streamlit_prophet` is a Python cli/package",
    add_completion=True,
)
app.add_typer(deploy.app, name="deploy")
console = Console()


def version_callback(value: bool):
    """Prints the version of the package."""
    if value:
        console.print(f"[yellow]streamlit_prophet[/] version: [bold blue]{__version__}[/]")
        raise typer.Exit()
