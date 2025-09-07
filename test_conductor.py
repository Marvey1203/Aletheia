# test_conductor.py

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
        console.print("[yellow]Connecting to C2 server...[/yellow]")
        req_socket = context.socket(zmq.REQ)
        req_socket.connect(COMMAND_ADDRESS)

        command = {
            "command": "reason",
            "payload": {
                "query": query,
                "gear_override": gear_override
            }
        }
        
        console.print(f"[yellow]Sending command:[/yellow] {json.dumps(command, indent=2)}")
        req_socket.send_json(command)
        
        # Wait for acknowledgment
        ack = req_socket.recv_json()
        console.print(f"[green]Received acknowledgment:[/green] {json.dumps(ack, indent=2)}")
        
        if ack.get("status") != "acknowledged":
            console.print("[bold red]Error: Did not receive successful acknowledgment.[/bold red]")
            return

        query_id = ack.get("query_id")
        if not query_id:
            console.print("[bold red]Error: Acknowledgment did not contain a query_id.[/bold red]")
            return

        # --- 2. Listen for Telemetry ---
        console.print(f"\n[yellow]Connecting to Telemetry stream for query_id: [/yellow][bold]{query_id}[/bold]")
        sub_socket = context.socket(zmq.SUB)
        sub_socket.connect(TELEMETRY_ADDRESS)
        sub_socket.setsockopt_string(zmq.SUBSCRIBE, query_id)

        # Set a timeout for receiving messages (e.g., 5 minutes)
        sub_socket.setsockopt(zmq.RCVTIMEO, 300000)

        console.print("[green]Listener connected. Awaiting telemetry...[/green]\n")

        with Live(console=console, screen=False, auto_refresh=True) as live:
            while True:
                try:
                    # Receive topic and message
                    topic = sub_socket.recv_string()
                    message_json = sub_socket.recv_json()
                    
                    panel_title = f"[bold]Telemetry Update for {topic}[/bold]"
                    message_type = message_json.get('type')
                    
                    if message_type == "stage_update":
                        panel_title = f"[bold]Stage: {message_json.get('stage', 'N/A').upper()}[/bold]"
                    elif message_type == "final_result":
                        panel_title = "[bold green]Final Trace Received[/bold green]"
                    
                    # Pretty print the JSON payload
                    syntax = Syntax(json.dumps(message_json.get("payload"), indent=2), "json", theme="monokai", line_numbers=True)
                    live.update(Panel(syntax, title=panel_title, border_style="yellow"))

                    if message_type == "final_result":
                        console.print(Panel("[bold green]Test Run Complete. Final trace received.[/bold green]", border_style="green"))
                        break
                        
                except zmq.Again:
                    console.print("[bold red]Error: Timed out waiting for telemetry message.[/bold red]")
                    break

    except Exception as e:
        console.print(f"[bold red]An unexpected error occurred: {e}[/bold red]")
    finally:
        req_socket.close()
        sub_socket.close()
        context.term()
        console.print("\n[cyan]Sockets closed. Test finished.[/cyan]")


if __name__ == "__main__":
    # Example of how to run the test from the command line:
    # python test_conductor.py "What is the capital of France?"
    # python test_conductor.py "Write a python script to list files" gear_3
    
    if len(sys.argv) < 2:
        print("Usage: python test_conductor.py <query> [gear_override]")
        sys.exit(1)
        
    test_query = sys.argv[1]
    test_gear = sys.argv[2] if len(sys.argv) > 2 else None
    
    run_test(test_query, test_gear)