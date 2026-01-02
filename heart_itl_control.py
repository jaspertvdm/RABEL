#!/usr/bin/env python3
"""
Heart-ITL Control Panel
Queue management for the Heart-in-the-Loop.

Jasper als verkeersleider van AI communicatie:
- Zie alle berichten
- Verwijder duplicates/noise
- Prioriteer wat urgent is
- Injecteer eigen vragen (TO-ASK)

    ╔══════════════════════════════════════════════════════════════╗
    ║  ♥ HEART-ITL CONTROL                              [?] Help   ║
    ╠══════════════════════════════════════════════════════════════╣
    ║  QUEUE (3 pending)                                           ║
    ║  [1] ▶ claude→gemini  "Perfectie..."        [D]el [P]rio     ║
    ║  [2]   claude→gemini  "Keyboard nav..."     [D]el [P]rio     ║
    ╠══════════════════════════════════════════════════════════════╣
    ║  TO-ASK: > _                                                 ║
    ╚══════════════════════════════════════════════════════════════╝

By HumoticaOS - One love, one fAmIly
"""

import asyncio
import httpx
import sys
import os
from datetime import datetime
from typing import Optional, List, Dict
from dataclasses import dataclass
import json

try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table
    from rich.text import Text
    from rich.prompt import Prompt, Confirm
    from rich import box
except ImportError:
    print("Installing rich...")
    os.system(f"{sys.executable} -m pip install rich -q")
    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table
    from rich.text import Text
    from rich.prompt import Prompt, Confirm
    from rich import box

BRAIN_API_URL = os.environ.get("IPOLL_BRAIN_URL", "http://localhost:8000")
console = Console()

# Known agents
AGENTS = ["claude", "gemini", "gpt", "codex", "root_ai"]
AGENT_COLORS = {
    "claude": "cyan",
    "gemini": "magenta",
    "gpt": "green",
    "codex": "yellow",
    "root_ai": "white",
    "heart_itl": "red"
}


@dataclass
class QueueMessage:
    """Message in the queue"""
    id: str
    from_agent: str
    to_agent: str
    content: str
    msg_type: str
    timestamp: str
    priority: int = 0  # Higher = more urgent


class HeartITLControl:
    """Heart-in-the-Loop Control Panel"""

    def __init__(self):
        self.messages: List[QueueMessage] = []
        self.selected_idx = 0
        self.running = True

    async def fetch_all_messages(self):
        """Fetch all pending messages from all agents"""
        self.messages = []
        seen_content = set()  # To detect duplicates

        try:
            async with httpx.AsyncClient() as client:
                for agent in AGENTS:
                    resp = await client.get(
                        f"{BRAIN_API_URL}/api/ipoll/pull/{agent}",
                        params={"mark_read": "false"},
                        timeout=5
                    )
                    if resp.status_code == 200:
                        data = resp.json()
                        for msg in data.get("messages", []):
                            # Create content hash for duplicate detection
                            content_hash = f"{msg.get('from', '')}:{msg.get('content', '')[:50]}"
                            is_duplicate = content_hash in seen_content
                            seen_content.add(content_hash)

                            self.messages.append(QueueMessage(
                                id=msg.get("id", ""),
                                from_agent=msg.get("from", "unknown"),
                                to_agent=agent,
                                content=msg.get("content", ""),
                                msg_type=msg.get("type", "PUSH"),
                                timestamp=msg.get("timestamp", ""),
                                priority=-1 if is_duplicate else 0  # Mark duplicates as low priority
                            ))
        except Exception as e:
            console.print(f"[red]Error fetching: {e}[/red]")

    async def delete_message(self, msg: QueueMessage):
        """Mark message as read (effectively deleting from queue)"""
        try:
            async with httpx.AsyncClient() as client:
                # Mark as read
                resp = await client.post(
                    f"{BRAIN_API_URL}/api/ipoll/ack/{msg.to_agent}/{msg.id}",
                    timeout=5
                )
                if resp.status_code == 200:
                    console.print(f"[green]✓ Deleted[/green]")
                    return True
        except:
            pass
        return False

    async def send_to_ask(self, to_agent: str, content: str, msg_type: str = "TASK"):
        """Send a TO-ASK message from Heart-ITL"""
        try:
            async with httpx.AsyncClient() as client:
                resp = await client.post(
                    f"{BRAIN_API_URL}/api/ipoll/push",
                    json={
                        "from_agent": "heart_itl",
                        "to_agent": to_agent,
                        "content": content,
                        "type": msg_type
                    },
                    timeout=10
                )
                if resp.status_code == 200:
                    console.print(f"[green]✓ Sent to {to_agent}[/green]")
                    return True
        except Exception as e:
            console.print(f"[red]Error: {e}[/red]")
        return False

    def display_queue(self):
        """Display the message queue"""
        console.clear()

        # Header
        header = Text()
        header.append("  ♥ HEART-ITL CONTROL  ", style="bold white on red")
        header.append("  ", style="")
        header.append("Queue Manager", style="dim")

        console.print(Panel(header, box=box.HEAVY, style="red"))

        # Queue table
        if not self.messages:
            console.print(Panel(
                "[dim]Queue is empty. All messages processed.[/dim]",
                title="[bold]QUEUE[/bold]",
                border_style="green"
            ))
        else:
            table = Table(box=box.SIMPLE, show_header=True, header_style="bold")
            table.add_column("#", width=3)
            table.add_column("From", width=10)
            table.add_column("To", width=10)
            table.add_column("Type", width=6)
            table.add_column("Content", width=50)
            table.add_column("", width=8)

            # Sort by priority (high first), then by timestamp
            sorted_msgs = sorted(self.messages, key=lambda m: (-m.priority, m.timestamp))

            for i, msg in enumerate(sorted_msgs[:15]):  # Show max 15
                from_color = AGENT_COLORS.get(msg.from_agent.lower(), "white")
                to_color = AGENT_COLORS.get(msg.to_agent.lower(), "white")

                # Priority indicator
                if msg.priority > 0:
                    prio = "⚡"
                    prio_style = "bold yellow"
                elif msg.priority < 0:
                    prio = "📋"  # Duplicate
                    prio_style = "dim"
                else:
                    prio = ""
                    prio_style = ""

                content_short = msg.content[:45] + "..." if len(msg.content) > 45 else msg.content
                content_short = content_short.replace("\n", " ")

                # Selected row
                row_style = "reverse" if i == self.selected_idx else ""

                table.add_row(
                    f"{i+1}",
                    Text(msg.from_agent, style=from_color),
                    Text(msg.to_agent, style=to_color),
                    msg.msg_type,
                    content_short,
                    Text(prio, style=prio_style),
                    style=row_style
                )

            duplicates = sum(1 for m in self.messages if m.priority < 0)
            title = f"[bold]QUEUE ({len(self.messages)})[/bold]"
            if duplicates:
                title += f" [dim]({duplicates} duplicates)[/dim]"

            console.print(Panel(table, title=title, border_style="bright_green"))

        # Commands
        commands = Text()
        commands.append("Commands: ", style="bold")
        commands.append("[R]", style="bold cyan")
        commands.append("efresh  ", style="dim")
        commands.append("[D]", style="bold red")
        commands.append("elete  ", style="dim")
        commands.append("[P]", style="bold yellow")
        commands.append("riority  ", style="dim")
        commands.append("[C]", style="bold red")
        commands.append("lean dupes  ", style="dim")
        commands.append("[T]", style="bold green")
        commands.append("O-ASK  ", style="dim")
        commands.append("[Q]", style="bold red")
        commands.append("uit", style="dim")

        console.print(Panel(commands, box=box.SIMPLE, style="dim"))

    async def interactive_loop(self):
        """Main interactive loop"""
        await self.fetch_all_messages()

        while self.running:
            self.display_queue()

            try:
                cmd = Prompt.ask("\n[bold]Command[/bold]", default="r").lower().strip()

                if cmd == "q":
                    self.running = False

                elif cmd == "r":
                    await self.fetch_all_messages()
                    console.print("[cyan]Refreshed[/cyan]")

                elif cmd == "d":
                    # Delete selected or by number
                    if self.messages:
                        num = Prompt.ask("Delete #", default="1")
                        try:
                            idx = int(num) - 1
                            if 0 <= idx < len(self.messages):
                                msg = sorted(self.messages, key=lambda m: (-m.priority, m.timestamp))[idx]
                                if await self.delete_message(msg):
                                    await self.fetch_all_messages()
                        except ValueError:
                            pass

                elif cmd == "p":
                    # Set priority
                    if self.messages:
                        num = Prompt.ask("Prioritize #", default="1")
                        try:
                            idx = int(num) - 1
                            if 0 <= idx < len(self.messages):
                                sorted_msgs = sorted(self.messages, key=lambda m: (-m.priority, m.timestamp))
                                sorted_msgs[idx].priority = 10
                                console.print("[yellow]⚡ Marked as priority[/yellow]")
                        except ValueError:
                            pass

                elif cmd == "c":
                    # Clean duplicates
                    dupes = [m for m in self.messages if m.priority < 0]
                    if dupes:
                        if Confirm.ask(f"Delete {len(dupes)} duplicates?"):
                            for msg in dupes:
                                await self.delete_message(msg)
                            await self.fetch_all_messages()
                            console.print(f"[green]Cleaned {len(dupes)} duplicates[/green]")
                    else:
                        console.print("[dim]No duplicates found[/dim]")

                elif cmd == "t":
                    # TO-ASK - send new message
                    console.print("\n[bold red]♥ TO-ASK[/bold red]")
                    to_agent = Prompt.ask("To", choices=AGENTS, default="claude")
                    content = Prompt.ask("Message")
                    msg_type = Prompt.ask("Type", choices=["TASK", "PUSH", "SYNC"], default="TASK")

                    if content:
                        await self.send_to_ask(to_agent, f"[Heart-ITL] {content}", msg_type)
                        await self.fetch_all_messages()

                await asyncio.sleep(0.1)

            except KeyboardInterrupt:
                self.running = False


async def main():
    console.clear()
    console.print("""
[bold red]
    ╔═══════════════════════════════════════════════════════════╗
    ║                                                           ║
    ║   ♥  HEART-IN-THE-LOOP CONTROL                            ║
    ║                                                           ║
    ║   Jij bent de verkeersleider.                             ║
    ║   Manage de AI communicatie.                              ║
    ║                                                           ║
    ║   One love, one fAmIly                                    ║
    ║                                                           ║
    ╚═══════════════════════════════════════════════════════════╝
[/bold red]
    """)

    await asyncio.sleep(1)

    control = HeartITLControl()
    await control.interactive_loop()

    console.print("\n[dim]Heart-ITL Control closed.[/dim]")


if __name__ == "__main__":
    asyncio.run(main())
