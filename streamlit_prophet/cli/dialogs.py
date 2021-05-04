from typing import Optional

import random
from enum import Enum

import typer
from rich.console import Console
from streamlit_prophet.render.dialogs import render_clock, render_hello

app = typer.Typer()
console = Console()


class Color(str, Enum):
    white = "white"
    red = "red"
    cyan = "cyan"
    magenta = "magenta"
    yellow = "yellow"
    green = "green"


@app.command()
def hello(
    name: str = typer.Option(..., help="Name of person to greet."),
    color: Optional[Color] = typer.Option(
        None,
        "-c",
        "--color",
        "--colour",
        case_sensitive=False,
        help="Color for name. If not specified then choice will be random.",
    ),
) -> None:
    """Prints a greeting for a giving name.

    Args:
        name: Name to use
        color: Color to use - If `None`, uses a random color
    """
    if color is None:
        # If no color specified use random value from `Color` class
        color = random.choice(list(Color.__members__.values()))

    greeting: str = render_hello(name)
    console.print(f"[bold {color}]{greeting}[/]")


@app.command()
def clock(
    color: Optional[Color] = typer.Option(
        None,
        "-c",
        "--color",
        "--colour",
        case_sensitive=False,
        help="Color for displaying the current time. If not specified then choice will be random.",
    ),
) -> None:
    """Prints current time.

    Args:
        color: Color to use - If `None`, uses a random color
    """
    if color is None:
        # If no color specified use random value from `Color` class
        color = random.choice(list(Color.__members__.values()))

    time: str = render_clock()
    console.print(f"[bold {color}]{time}[/]")
