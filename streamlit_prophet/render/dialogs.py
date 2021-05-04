"""We use this file as an example for some module."""
from datetime import datetime


def render_hello(name: str) -> str:
    """Just a greetings example.

    Args:
        name: Name to render

    Returns:
        The rendered message

    Examples:
        .. code:: python

            >>> render_hello("Roman")
            'Hello Roman!'
    """
    return f"Hello {name}!"


def render_clock() -> str:
    """Just an example that gets current time.

    Returns:
        The current time
    """
    return f"It is {datetime.now().strftime('%c')}!"


__all__ = ["render_hello", "render_clock"]
