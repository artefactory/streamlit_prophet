import typer
from rich.console import Console
from streamlit_prophet.app import deploy_streamlit, deploy_streamlit_with_base_path

app = typer.Typer()
console = Console()


@app.command()
def dashboard() -> None:
    """Deploys the streamlit dashboard."""
    deploy_streamlit()

@app.command()
def dashboard_with_base_path() -> None:
    """Deploys the streamlit dashboard with a base path."""
    deploy_streamlit_with_base_path()