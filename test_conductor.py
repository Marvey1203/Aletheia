# test_conductor.py (V2 - With Memory Seeding)

import zmq
import json
import time
import sys
from rich.console import Console
from rich.panel import Panel
from rich.syntax import Syntax
from rich.live import Live
from rich.text import Text

# --- Configuration ---
COMMAND_ADDRESS = "tcp://localhost:5555"
TELEMETRY_ADDRESS = "tcp://localhost:5556"

# --- Rich Console for pretty printing ---
console = Console()

def seed_memory(texts: list[str]):
    """Sends a command to the server to seed the memory with sample texts."""
    console.print(Panel("[bold yellow]Initiating Memory Seeding Run[/bold yellow]", border_style="yellow"))
    
    context = zmq.Context()
    try:
        req_socket = context.socket(zmq.REQ)
        req_socket.connect(COMMAND_ADDRESS)

        command = {
            "command": "seed_memory",
            "payload": {
                "texts": texts
            }
        }
        
        console.print(f"[yellow]Sending seed command...[/yellow]")
        req_socket.send_json(command)
        
        ack = req_socket.recv_json()
        console.print(f"[green]Server Response:[/green] {json.dumps(ack, indent=2)}")

    except Exception as e:
        console.print(f"[bold red]An unexpected error occurred during seeding: {e}[/bold red]")
    finally:
        req_socket.close()
        context.term()
        console.print("\n[cyan]Seeding finished.[/cyan]")

def run_test(query: str, gear_override: str = None):
    """
    Connects to the Aletheia IPC server, sends a command,
    and listens for the full telemetry stream.
    """
    console.print(Panel(f"[bold cyan]Initiating Test Run[/bold cyan]\n[bold]Query:[/bold] {query}\n[bold]Gear Override:[/bold] {gear_override or 'auto'}", border_style="cyan"))
    
    context = zmq.Context()
    query_id = None

    try:
        # --- 1. Send the Command ---
        req_socket = context.socket(zmq.REQ)
        req_socket.connect(COMMAND_ADDRESS)

        command = {
            "command": "reason",
            "payload": {
                "query": query,
                "gear_override": gear_override
            }
        }
        
        req_socket.send_json(command)
        ack = req_socket.recv_json()
        
        if ack.get("status") != "acknowledged":
            console.print(f"[bold red]Error: Did not receive successful acknowledgment: {ack}[/bold red]")
            return

        query_id = ack.get("query_id")

        # --- 2. Listen for Telemetry ---
        sub_socket = context.socket(zmq.SUB)
        sub_socket.connect(TELEMETRY_ADDRESS)
        sub_socket.setsockopt_string(zmq.SUBSCRIBE, query_id)
        sub_socket.setsockopt(zmq.RCVTIMEO, 300000)

        with Live(console=console, screen=False, auto_refresh=True) as live:
            while True:
                try:
                    topic = sub_socket.recv_string()
                    message_json = sub_socket.recv_json()
                    
                    panel_title = f"[bold]Telemetry: {message_json.get('type', '').upper()}[/bold]"
                    if message_json.get('type') == 'stage_update':
                        panel_title += f" - [bold yellow]{message_json.get('stage', 'N/A').upper()}[/bold yellow]"

                    syntax = Syntax(json.dumps(message_json.get("payload"), indent=2), "json", theme="monokai", line_numbers=True)
                    live.update(Panel(syntax, title=panel_title, border_style="yellow"))

                    if message_json.get('type') == "final_result":
                        console.print(Panel("[bold green]Test Run Complete. Final trace received.[/bold green]", border_style="green"))
                        break
                        
                except zmq.Again:
                    console.print("[bold red]Error: Timed out waiting for telemetry message.[/bold red]")
                    break

    except Exception as e:
        console.print(f"[bold red]An unexpected error occurred: {e}[/bold red]")
    finally:
        if 'req_socket' in locals(): req_socket.close()
        if 'sub_socket' in locals(): sub_socket.close()
        context.term()
        console.print("\n[cyan]Sockets closed. Test finished.[/cyan]")


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage:")
        print("  python test_conductor.py reason \"<query>\" [gear_override]")
        print("  python test_conductor.py seed \"<text1>;<text2>;...\"")
        sys.exit(1)
        
    command_type = sys.argv[1]
    
    if command_type == "reason":
        test_query = sys.argv[2]
        test_gear = sys.argv[3] if len(sys.argv) > 3 else None
        run_test(test_query, test_gear)
    elif command_type == "seed":
        # Use a semicolon to separate multiple texts to be seeded
        seed_texts = sys.argv[2].split(';')
        seed_memory(seed_texts)
    else:
        print(f"Unknown command: {command_type}")
        sys.exit(1)