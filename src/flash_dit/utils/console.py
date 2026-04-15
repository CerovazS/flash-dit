from rich.console import Console

_console = Console(force_terminal=True)


def ok(msg: str) -> None:
    _console.print(f"[bold green]OK[/bold green] {msg}")


def info(msg: str) -> None:
    _console.print(f"[cyan]INFO[/cyan] {msg}")


def warn(msg: str) -> None:
    _console.print(f"[bold yellow]WARN[/bold yellow] {msg}")


def error(msg: str) -> None:
    _console.print(f"[bold red]ERROR[/bold red] {msg}")
