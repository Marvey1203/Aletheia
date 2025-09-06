# interfaces/cli.py
# Enhanced CLI with chat interface, progress reporting, and gear controls

import sys
import typer
from typing import Dict, Any, Optional, List, Callable
from pathlib import Path
from rich.console import Console
from rich.panel import Panel
from rich.status import Status
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn
from rich.live import Live
from rich.text import Text
from loguru import logger

# Import the core components
from core.identity import IdentityCore
from core.memory import MemoryGalaxy
from core.llm import LocalLLM
from core.atp import ATPLoopV2

# --- Robust State Management ---
_app_state: Dict[str, Any] = {}

def get_app_state(dummy: bool = False) -> Dict[str, Any]:
    """Initializes and returns the application state."""
    global _app_state
    if not _app_state:
        console = Console()
        console.print(Panel("🌌 [bold cyan]Aletheia Genesis Engine[/bold cyan] 🌌", border_style="cyan"))
        init_message = "Waking up Aletheia... (loading model)"
        if dummy:
            init_message = "Waking up Aletheia... (Dummy Mode)"
        
        with console.status(init_message, spinner="dots"):
            try:
                identity_core = IdentityCore()
                memory_galaxy = MemoryGalaxy()
                local_llm = LocalLLM(dummy_mode=dummy)
                atp_loop = ATPLoopV2(
                    identity=identity_core, 
                    memory=memory_galaxy, 
                    llm=local_llm
                )
                _app_state["atp_loop"] = atp_loop
                _app_state["console"] = console
            except Exception as e:
                console.print(f"\n[bold red]Fatal Error during initialization:[/bold red] {e}")
                raise typer.Exit(code=1)
        console.print("[bold green]Aletheia is awake and ready.[/bold green]")
    return _app_state

# --- CLI Application ---
app = typer.Typer(
    name="aletheia",
    help="Aletheia: A Sovereign AI Partner. Own your mind.",
    add_completion=False,
)

console = Console()

def chat_interface(atp_loop: ATPLoopV2, console: Console):
    """Interactive chat interface with Aletheia"""
    console.print(Panel(
        "[bold cyan]Welcome to Aletheia Chat[/bold cyan]\n"
        "Type '/exit' to quit, '/clear' to clear history\n"
        "Type '/gear1', '/gear2', or '/gear3' to manually set reasoning depth",
        border_style="blue"
    ))
    
    conversation_history = []
    current_gear = None  # None means auto-detect
    
    while True:
        try:
            user_input = console.input("\n[bold yellow]You:[/bold yellow] ").strip()
            
            if user_input.lower() == '/exit':
                break
            elif user_input.lower() == '/clear':
                conversation_history = []
                console.print("[dim]Conversation history cleared.[/dim]")
                continue
            elif user_input.lower() in ['/gear1', '/gear2', '/gear3']:
                gear_map = {
                    '/gear1': 'gear_1',
                    '/gear2': 'gear_2', 
                    '/gear3': 'gear_3'
                }
                current_gear = gear_map[user_input.lower()]
                console.print(f"[dim]Manual gear override set to: {current_gear}[/dim]")
                continue
            elif not user_input:
                continue
                
            # Add to conversation history
            conversation_history.append(f"You: {user_input}")
            
            # Display thinking indicator
            with console.status("[cyan]Aletheia is thinking...[/cyan]", spinner="dots") as status:
                # Create a progress callback function
                def progress_callback(stage: str, result: Any):
                    if stage == "triage":
                        if isinstance(result, str):
                            status.update(f"[cyan]Triage: {result}[/cyan]")
                        else:
                            status.update("[cyan]Triage: Analyzing query complexity...[/cyan]")
                    elif stage == "plan":
                        if isinstance(result, list):
                            status.update(f"[cyan]Planning: Plan generated with {len(result)} steps[/cyan]")
                        else:
                            status.update(f"[cyan]Planning: {result}[/cyan]")
                    elif stage == "execute":
                        if isinstance(result, str):
                            preview = result[:50] + "..." if len(result) > 50 else result
                            status.update(f"[cyan]Executing: Answer generated - {preview}[/cyan]")
                        else:
                            status.update(f"[cyan]Executing: {result}[/cyan]")
                    elif stage == "critique":
                        if isinstance(result, dict):
                            scores = ", ".join([f"{k}: {v:.2f}" for k, v in result.items()])
                            status.update(f"[cyan]Critiquing: Scores calculated ({scores})[/cyan]")
                        else:
                            status.update(f"[cyan]Critiquing: {result}[/cyan]")
                    elif stage == "reflection":
                        if isinstance(result, str):
                            status.update(f"[cyan]Reflecting: {result}[/cyan]")
                        else:
                            status.update("[cyan]Reflecting: Reflection completed[/cyan]")
                    elif stage == "gear_1":
                        status.update("[cyan]Gear 1: Generating direct response...[/cyan]")
                
                # Process the query with optional gear override
                final_trace = atp_loop.reason(user_input, progress_callback=progress_callback, gear_override=current_gear)
                
                # Reset gear override after use (one-time override)
                if current_gear:
                    console.print(f"[dim]Gear override ({current_gear}) applied. Returning to auto-detection.[/dim]")
                    current_gear = None
                
                # Add response to conversation history
                conversation_history.append(f"Aletheia: {final_trace.summary.answer}")
                
                # Display the response with gear information
                gear_info = ""
                if "Gear 1" in final_trace.summary.reasoning:
                    gear_info = " (Gear 1 - Direct Response)"
                elif "Gear 2" in final_trace.summary.reasoning:
                    gear_info = " (Gear 2 - Standard Reasoning)"
                elif "Gear 3" in final_trace.summary.reasoning:
                    gear_info = " (Gear 3 - Deep Analysis)"
                
                console.print(Panel(
                    final_trace.summary.answer,
                    title=f"[bold green]Aletheia{gear_info}[/bold green]",
                    border_style="green"
                ))
                
        except KeyboardInterrupt:
            console.print("\n[yellow]Interrupted. Type '/exit' to quit.[/yellow]")
        except Exception as e:
            logger.exception("An error occurred during chat")
            console.print(f"\n[bold red]Error:[/bold red] {e}")

@app.command()
def chat(
    dummy: bool = typer.Option(False, "--dummy", "-d", help="Run in dummy mode for testing on machines without a GPU.")
):
    """
    Start an interactive chat session with Aletheia.
    """
    state = get_app_state(dummy=dummy)
    atp_loop = state["atp_loop"]
    console = state["console"]
    
    chat_interface(atp_loop, console)

@app.command()
def reason(
    dummy: bool = typer.Option(False, "--dummy", "-d", help="Run in dummy mode for testing on machines without a GPU."),
    input_file: Optional[str] = typer.Option(None, "--input", "-i", help="Input file with queries, one per line. If not provided, read from stdin."),
    gear: Optional[str] = typer.Option(None, "--gear", "-g", help="Force a specific gear: 1, 2, or 3"),
):
    """
    Ask Aletheia to reason about one or more queries. Queries can be provided via input file or stdin.
    """
    state = get_app_state(dummy=dummy)
    atp_loop = state["atp_loop"]
    console = state["console"]

    # Map gear argument to internal format
    gear_override = None
    if gear:
        gear_map = {"1": "gear_1", "2": "gear_2", "3": "gear_3"}
        gear_override = gear_map.get(gear)
        if not gear_override:
            console.print(f"[bold red]Error:[/bold red] Invalid gear specified. Use 1, 2, or 3.")
            raise typer.Exit(code=1)

    # Read queries
    queries: List[str] = []
    if input_file:
        try:
            with open(input_file, 'r') as f:
                queries = [line.strip() for line in f.readlines() if line.strip()]
        except FileNotFoundError:
            console.print(f"[bold red]Error:[/bold red] Input file '{input_file}' not found.")
            raise typer.Exit(code=1)
        except Exception as e:
            console.print(f"[bold red]Error reading file:[/bold red] {e}")
            raise typer.Exit(code=1)
    else:
        if not sys.stdin.isatty():
            queries = [line.strip() for line in sys.stdin.readlines() if line.strip()]
        else:
            console.print("[bold red]Error:[/bold red] Please provide input via file or stdin.")
            raise typer.Exit(code=1)

    if not queries:
        console.print("[bold red]Error:[/bold red] No queries found.")
        raise typer.Exit(code=1)

    # Process queries with overall progress bar
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TimeElapsedColumn(),
    ) as progress:
        task = progress.add_task("Processing queries...", total=len(queries))

        for i, query in enumerate(queries):
            progress.update(task, advance=1, description=f"Processing query {i+1}/{len(queries)}")
            console.print(Panel(f"[cyan]Query:[/cyan] {query}", title="[bold]New Reasoning Task[/bold]", border_style="blue"))
            
            if gear_override:
                console.print(f"[dim]Using manual gear override: {gear_override}[/dim]")
            
            try:
                with console.status("Starting reasoning...", spinner="dots") as status:
                    def progress_callback(stage: str, result: Any):
                        if stage == "triage":
                            if isinstance(result, str):
                                status.update(f"Triage: {result}")
                            else:
                                status.update("Triage: Analyzing query complexity...")
                        elif stage == "plan":
                            if isinstance(result, list):
                                status.update(f"Planning: Plan generated with {len(result)} steps")
                            else:
                                status.update(f"Planning: {result}")
                        elif stage == "execute":
                            if isinstance(result, str):
                                # Preview first 50 chars of answer
                                preview = result[:50] + "..." if len(result) > 50 else result
                                status.update(f"Executing: Answer generated - {preview}")
                            else:
                                status.update(f"Executing: {result}")
                        elif stage == "critique":
                            if isinstance(result, dict):
                                scores = ", ".join([f"{k}: {v:.2f}" for k, v in result.items()])
                                status.update(f"Critiquing: Scores calculated ({scores})")
                            else:
                                status.update(f"Critiquing: {result}")
                        elif stage == "reflection":
                            if isinstance(result, str):
                                status.update(f"Reflecting: {result}")
                            else:
                                status.update(f"Reflecting: Reflection completed")
                        elif stage == "gear_1":
                            status.update("Gear 1: Generating direct response...")

                    final_trace = atp_loop.reason(query, progress_callback=progress_callback, gear_override=gear_override)

                # Display gear information
                gear_info = ""
                if "Gear 1" in final_trace.summary.reasoning:
                    gear_info = " (Gear 1 - Direct Response)"
                elif "Gear 2" in final_trace.summary.reasoning:
                    gear_info = " (Gear 2 - Standard Reasoning)"
                elif "Gear 3" in final_trace.summary.reasoning:
                    gear_info = " (Gear 3 - Deep Analysis)"

                console.print(Panel(
                    f"[bold green]Answer:[/bold green]\n{final_trace.summary.answer}\n\n"
                    f"[bold yellow]Reasoning:[/bold yellow]\n{final_trace.summary.reasoning}\n\n"
                    f"[bold magenta]Next Action:[/bold magenta]\n{final_trace.summary.next_action}",
                    title=f"[bold]Trace Complete{gear_info}: {final_trace.trace_id}[/bold]",
                    border_style="green",
                    subtitle="View the full cognitive log in the Observatory."
                ))

            except Exception as e:
                logger.exception("An error occurred during the reason command.")
                console.print(f"\n[bold red]Error during reasoning:[/bold red] {e}")
                continue

if __name__ == "__main__":
    app()