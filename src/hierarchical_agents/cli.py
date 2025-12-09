"""Command-line interface for hierarchical agents."""

import asyncio
from typing import Optional

import typer
from rich.console import Console
from rich.panel import Panel

app = typer.Typer(
    name="hierarchical-agents",
    help="Hierarchical Agents - Educational evaluation analysis system",
    add_completion=False,
)

console = Console()


@app.command()
def version():
    """Show version information."""
    from hierarchical_agents import __version__
    
    console.print(Panel.fit(
        f"[bold blue]Hierarchical Agents[/bold blue]\n"
        f"Version: [green]{__version__}[/green]",
        title="Version Info"
    ))


@app.command()
def test_db(
    url: Optional[str] = typer.Option(
        None,
        "--url",
        "-u", 
        help="Database URL (defaults to DATABASE_URL env var)"
    )
):
    """Test database connectivity."""
    console.print("[yellow]Testing database connection...[/yellow]")
    
    try:
        import subprocess
        import sys
        
        cmd = [sys.executable, "scripts/test_db_connection.py"]
        if url:
            import os
            os.environ["DATABASE_URL"] = url
            
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode == 0:
            console.print("[green]✅ Database connection successful![/green]")
        else:
            console.print("[red]❌ Database connection failed[/red]")
            if result.stderr:
                console.print(f"[red]Error: {result.stderr}[/red]")
                
    except Exception as e:
        console.print(f"[red]❌ Failed to run database test: {e}[/red]")


def main():
    """Entry point for the CLI."""
    app()


if __name__ == "__main__":
    main()