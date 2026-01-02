#!/usr/bin/env python3
"""
I-Poll Unified Terminal v4.0 - NERVE CENTER EDITION
High-density sci-fi command center for AI communication.

By HumoticaOS - One love, one fAmIly
"""

import asyncio
import httpx
import sys
import os
from datetime import datetime, timedelta
from typing import List
from dataclasses import dataclass
import time

# Rich for TUI
try:
    from rich.console import Console, Group
    from rich.panel import Panel
    from rich.table import Table
    from rich.text import Text
    from rich.align import Align
    from rich import box
    from rich.prompt import Prompt
    from rich.columns import Columns
    from rich.progress import SpinnerColumn, Progress
except ImportError:
    os.system(f"{sys.executable} -m pip install rich -q")
    from rich.console import Console, Group
    from rich.panel import Panel
    from rich.table import Table
    from rich.text import Text
    from rich.align import Align
    from rich import box
    from rich.prompt import Prompt
    from rich.columns import Columns

# System metrics
try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False

BRAIN_API_URL = os.environ.get("IPOLL_BRAIN_URL", "http://localhost:8000")
console = Console()

AGENTS = ["claude", "gemini", "gpt", "codex", "root_ai", "heart_itl", "oomllama"]
AGENT_COLORS = {
    "claude": "bold cyan",
    "gemini": "bold magenta",
    "gpt": "bold green",
    "codex": "bold yellow",
    "root_ai": "bold white",
    "heart_itl": "bold red",
    "oomllama": "bold bright_yellow",
    "kit": "bold bright_blue"
}
AGENT_ICONS = {
    "claude": "🔵",
    "gemini": "🟣",
    "gpt": "🟢",
    "codex": "🟡",
    "root_ai": "⚪",
    "heart_itl": "❤️",
    "oomllama": "🦙",
    "kit": "🧠"
}

TYPE_COLORS = {
    "PUSH": "bold green",
    "PULL": "bold blue",
    "SYNC": "bold cyan",
    "TASK": "bold yellow",
    "ACK": "bold magenta",
    "HANDOFF": "bold red"
}

START_TIME = time.time()


@dataclass
class Message:
    id: str
    from_agent: str
    to_agent: str
    content: str
    msg_type: str
    timestamp: str
    priority: int = 0


@dataclass
class TIBETToken:
    token_id: str
    token_type: str
    intent: str
    state: str
    trust_score: float = 1.0


class IPollNerveCenter:
    """I-Poll Nerve Center - High Density Command Interface"""

    def __init__(self):
        self.current_tab = 1
        self.messages: List[Message] = []
        self.queue: List[Message] = []
        self.tokens: List[TIBETToken] = []
        self.selected_idx = 0
        self.page = 0
        self.stats = {"total_tokens": 0, "active_threats": 0, "by_type": []}
        self.spinners = ["◜", "◠", "◝", "◞", "◡", "◟"]
        self.scan_frames = ["▁", "▂", "▃", "▄", "▅", "▆", "▇", "█", "▇", "▆", "▅", "▄", "▃", "▂"]

    def get_system_metrics(self) -> dict:
        """Get live system metrics"""
        if HAS_PSUTIL:
            cpu = psutil.cpu_percent(interval=0)
            mem = psutil.virtual_memory()
            return {
                "cpu": f"{cpu:.0f}%",
                "mem": f"{mem.used / (1024**3):.1f}GB",
                "mem_pct": f"{mem.percent:.0f}%"
            }
        return {"cpu": "N/A", "mem": "N/A", "mem_pct": "N/A"}

    def get_uptime(self) -> str:
        """Get session uptime"""
        elapsed = time.time() - START_TIME
        hours = int(elapsed // 3600)
        mins = int((elapsed % 3600) // 60)
        secs = int(elapsed % 60)
        return f"{hours:02d}:{mins:02d}:{secs:02d}"

    async def fetch_data(self):
        """Fetch all data from APIs"""
        try:
            async with httpx.AsyncClient() as client:
                all_msgs = []
                for agent in AGENTS:
                    try:
                        resp = await client.get(
                            f"{BRAIN_API_URL}/api/ipoll/pull/{agent}",
                            params={"mark_read": "false"},
                            timeout=3
                        )
                        if resp.status_code == 200:
                            data = resp.json()
                            msgs = data.get("polls", data.get("messages", []))
                            for msg in msgs:
                                all_msgs.append(Message(
                                    id=msg.get("id", ""),
                                    from_agent=msg.get("from", "?"),
                                    to_agent=agent,
                                    content=msg.get("content", ""),
                                    msg_type=msg.get("type", "PUSH"),
                                    timestamp=msg.get("created_at", msg.get("timestamp", "")),
                                ))
                    except:
                        pass

                self.messages = sorted(all_msgs, key=lambda m: m.timestamp, reverse=True)[:50]

                # Queue with duplicate detection
                self.queue = []
                seen = set()
                for msg in all_msgs:
                    key = f"{msg.from_agent}:{msg.content[:30]}"
                    is_dupe = key in seen
                    seen.add(key)
                    msg.priority = -1 if is_dupe else 0
                    self.queue.append(msg)

                # TIBET tokens
                try:
                    resp = await client.get(
                        f"{BRAIN_API_URL}/api/tibet/search",
                        params={"limit": 15},
                        timeout=3
                    )
                    if resp.status_code == 200:
                        self.tokens = [
                            TIBETToken(
                                token_id=t.get("token_id", "")[:16],
                                token_type=t.get("token_type", "?"),
                                intent=t.get("intent", ""),
                                state=t.get("state", "?"),
                                trust_score=t.get("trust_score", 1.0)
                            )
                            for t in resp.json().get("results", [])
                        ]
                except:
                    pass

                # Stats
                try:
                    resp = await client.get(f"{BRAIN_API_URL}/api/tibet/stats", timeout=3)
                    if resp.status_code == 200:
                        data = resp.json()
                        self.stats = data
                except:
                    pass

        except Exception as e:
            pass

    async def delete_message(self, msg: Message):
        """Delete message from queue"""
        try:
            async with httpx.AsyncClient() as client:
                await client.post(
                    f"{BRAIN_API_URL}/api/ipoll/ack/{msg.to_agent}/{msg.id}",
                    timeout=5
                )
                return True
        except:
            return False

    async def send_message(self, to_agent: str, content: str, msg_type: str = "TASK"):
        """Send TO-ASK message"""
        try:
            async with httpx.AsyncClient() as client:
                await client.post(
                    f"{BRAIN_API_URL}/api/ipoll/push",
                    json={
                        "from_agent": "heart_itl",
                        "to_agent": to_agent,
                        "content": f"[Heart-ITL] {content}",
                        "type": msg_type
                    },
                    timeout=10
                )
                console.print(f"[green]✓ Sent to {to_agent}[/green]")
                return True
        except Exception as e:
            console.print(f"[red]✗ Error: {e}[/red]")
            return False

    def render_header(self):
        """Render header with system metrics"""
        spinner = self.spinners[int(time.time() * 4) % len(self.spinners)]
        scan = self.scan_frames[int(time.time() * 8) % len(self.scan_frames)]
        metrics = self.get_system_metrics()
        uptime = self.get_uptime()

        # Left side: Logo and tabs
        left = Text()
        left.append(f" {spinner} ", style="bold bright_green")
        left.append("I-POLL", style="bold black on bright_green")
        left.append(" ", style="")
        left.append("NERVE CENTER", style="bold bright_green")
        left.append("  ", style="")

        tabs = [("1", "STREAM"), ("2", "QUEUE"), ("3", "TOKENS")]
        for key, name in tabs:
            if int(key) == self.current_tab:
                left.append(f"[{key}]{name}", style="bold black on cyan")
            else:
                left.append(f"[{key}]{name}", style="dim cyan")
            left.append(" ", style="")

        # Right side: System metrics
        right = Text()
        right.append(f"{scan} ", style="bright_green")
        right.append("CPU:", style="dim")
        right.append(f"{metrics['cpu']}", style="bold cyan")
        right.append(" │ ", style="dim")
        right.append("MEM:", style="dim")
        right.append(f"{metrics['mem']}", style="bold cyan")
        right.append(" │ ", style="dim")
        right.append("UP:", style="dim")
        right.append(f"{uptime}", style="bold green")
        right.append(" │ ", style="dim")
        right.append("♥", style="bold red")

        # Combine
        header_table = Table.grid(expand=True)
        header_table.add_column(justify="left")
        header_table.add_column(justify="right")
        header_table.add_row(left, right)

        console.print(Panel(header_table, box=box.HEAVY, style="bright_green", padding=(0, 1)))

    def render_stream(self):
        """Render stream view - high density chat"""
        term_height = console.size.height
        max_msgs = min(10, max(4, (term_height - 14) // 2))
        start = self.page * max_msgs
        page_msgs = self.messages[start:start + max_msgs]

        content = Text()

        if not page_msgs:
            # Empty state with animation
            scan = self.scan_frames[int(time.time() * 6) % len(self.scan_frames)]
            content.append(f"\n  {scan} ", style="bright_green")
            content.append("SCANNING FOR TRANSMISSIONS", style="dim bright_green")
            content.append(f" {scan}\n", style="bright_green")
            content.append("  Listening on ", style="dim")
            content.append(BRAIN_API_URL, style="cyan")
            content.append("...\n", style="dim")
        else:
            for msg in page_msgs:
                # Parse timestamp
                try:
                    dt = datetime.fromisoformat(msg.timestamp.replace('Z', '+00:00'))
                    ts = dt.strftime("%H:%M:%S")
                except:
                    ts = time.strftime("%H:%M:%S")

                # Agent colors and icons
                from_icon = AGENT_ICONS.get(msg.from_agent.lower(), "●")
                from_color = AGENT_COLORS.get(msg.from_agent.lower(), "white")
                to_color = AGENT_COLORS.get(msg.to_agent.lower(), "white")
                type_color = TYPE_COLORS.get(msg.msg_type.upper(), "white")

                # Direction indicator
                direction = "→" if msg.msg_type in ["PUSH", "TASK", "SYNC"] else "←"

                # Build line
                content.append(f"[{ts}] ", style="dim green")
                content.append(f"[{msg.msg_type[:4]}]", style=type_color)
                content.append(f" {from_icon}", style="")
                content.append(f"{msg.from_agent[:7]:<7}", style=from_color)
                content.append(f" {direction} ", style="bright_green")
                content.append(f"{msg.to_agent[:7]:<7}", style=to_color)
                content.append("\n", style="")

                # Message preview with proper truncation
                preview = msg.content[:55].replace("\n", " ").strip()
                if len(msg.content) > 55:
                    preview += "…"
                content.append(f"   └─ {preview}\n", style="white")

        # Footer with pagination
        total_pages = max(1, (len(self.messages) + max_msgs - 1) // max_msgs)
        footer = Text()
        footer.append(f"\n  [{len(self.messages)} msgs]", style="dim")
        if total_pages > 1:
            footer.append(f" Page {self.page + 1}/{total_pages}", style="dim cyan")
            footer.append(" │ N/P=navigate", style="dim")
        content.append_text(footer)

        console.print(Panel(
            content,
            title="[bold bright_green]◀ LIVE STREAM ▶[/bold bright_green]",
            subtitle="[dim]AI-to-AI Communication Feed[/dim]",
            border_style="bright_green",
            box=box.ROUNDED
        ))

    def render_queue(self):
        """Render queue view - data grid style"""
        table = Table(
            box=box.SIMPLE_HEAD,
            show_header=True,
            header_style="bold cyan",
            expand=True,
            row_styles=["", "dim"]
        )
        table.add_column("#", width=3, justify="right")
        table.add_column("TYPE", width=6)
        table.add_column("FROM", width=8)
        table.add_column("TO", width=8)
        table.add_column("CONTENT", ratio=1)
        table.add_column("⚡", width=2)

        sorted_q = sorted(self.queue, key=lambda m: (-m.priority, m.timestamp), reverse=False)
        term_height = console.size.height
        max_q = min(8, max(4, (term_height - 14) // 2))

        if not sorted_q:
            # Empty state
            empty = Text()
            scan = self.scan_frames[int(time.time() * 4) % len(self.scan_frames)]
            empty.append(f"\n  {scan} ", style="cyan")
            empty.append("NO ACTIVE TRANSMISSIONS\n", style="dim cyan")
            empty.append("  Queue is clear. Awaiting incoming data...\n", style="dim")
            console.print(Panel(
                empty,
                title="[bold red]♥ HEART-ITL QUEUE[/bold red]",
                border_style="red",
                box=box.ROUNDED
            ))
            return

        for i, msg in enumerate(sorted_q[:max_q]):
            from_color = AGENT_COLORS.get(msg.from_agent.lower(), "white")
            to_color = AGENT_COLORS.get(msg.to_agent.lower(), "white")
            type_color = TYPE_COLORS.get(msg.msg_type.upper(), "white")

            flag = "⚡" if msg.priority > 0 else "📋" if msg.priority < 0 else ""
            row_style = "reverse" if i == self.selected_idx else ""

            content_preview = msg.content[:40].replace("\n", " ")
            if len(msg.content) > 40:
                content_preview += "…"

            table.add_row(
                str(i + 1),
                Text(msg.msg_type[:4], style=type_color),
                Text(msg.from_agent[:8], style=from_color),
                Text(msg.to_agent[:8], style=to_color),
                content_preview,
                flag,
                style=row_style
            )

        dupes = sum(1 for m in self.queue if m.priority < 0)
        subtitle = f"[dim]{len(self.queue)} items"
        if dupes:
            subtitle += f" │ {dupes} dupes (📋)"
        subtitle += "[/dim]"

        console.print(Panel(
            table,
            title="[bold red]♥ HEART-ITL QUEUE[/bold red]",
            subtitle=subtitle,
            border_style="red",
            box=box.ROUNDED
        ))

        # Command hints
        console.print("[dim]  D# Delete │ P# Priority │ C Clean dupes │ T TO-ASK[/dim]")

    def render_tokens(self):
        """Render tokens view - provenance tree"""
        content = Text()

        # Stats header
        total = self.stats.get("total_tokens", 0)
        if total >= 1_000_000:
            total_str = f"{total/1_000_000:.2f}M"
        else:
            total_str = f"{total:,}"

        # Stats bar
        stats_bar = Text()
        stats_bar.append("  TOKENS: ", style="dim")
        stats_bar.append(total_str, style="bold bright_green")
        stats_bar.append("  │  ", style="dim")

        # Type breakdown
        by_type = self.stats.get("by_type", [])
        for bt in by_type[:4]:
            tt = bt.get("token_type", "?")[:8]
            tc = bt.get("total_count", 0)
            if tc >= 1000:
                tc_str = f"{tc/1000:.0f}k"
            else:
                tc_str = str(tc)
            stats_bar.append(f"{tt}:", style="dim")
            stats_bar.append(f"{tc_str}", style="cyan")
            stats_bar.append("  ", style="")

        content.append_text(stats_bar)
        content.append("\n")
        content.append("  " + "─" * 50 + "\n", style="dim green")

        # Token tree
        state_icons = {
            "created": "○", "detected": "◐", "classified": "◑",
            "mitigated": "◕", "resolved": "●", "active": "▶",
            "permanent": "◆", "expired": "◇"
        }
        state_colors = {
            "created": "blue", "detected": "yellow", "classified": "cyan",
            "mitigated": "magenta", "resolved": "bright_green", "active": "green",
            "permanent": "bright_green", "expired": "dim"
        }

        if not self.tokens:
            scan = self.scan_frames[int(time.time() * 4) % len(self.scan_frames)]
            content.append(f"\n  {scan} ", style="cyan")
            content.append("Loading provenance data...\n", style="dim")
        else:
            for i, token in enumerate(self.tokens[:8]):
                icon = state_icons.get(token.state.lower(), "?")
                color = state_colors.get(token.state.lower(), "white")
                is_last = i == len(self.tokens[:8]) - 1

                # Tree structure
                prefix = "  └─" if is_last else "  ├─"

                content.append(prefix, style="dim green")
                content.append(f" {icon} ", style=color)
                content.append(f"[{token.token_type[:12]:<12}]", style="cyan")
                content.append(f" {token.state}", style=color)

                # Trust indicator
                if token.trust_score >= 0.8:
                    content.append(" ✓", style="green")
                elif token.trust_score >= 0.5:
                    content.append(" ~", style="yellow")
                else:
                    content.append(" ✗", style="red")

                content.append("\n", style="")

                # Intent preview
                if token.intent:
                    intent = token.intent[:45]
                    if len(token.intent) > 45:
                        intent += "…"
                    tree_prefix = "     " if is_last else "  │  "
                    content.append(f"{tree_prefix}  {intent}\n", style="dim")

        console.print(Panel(
            content,
            title="[bold cyan]◈ TIBET PROVENANCE[/bold cyan]",
            subtitle="[dim]Time-Intent Based Event Tokens[/dim]",
            border_style="cyan",
            box=box.ROUNDED
        ))

    def render(self):
        """Render current view"""
        console.clear()
        self.render_header()

        if self.current_tab == 1:
            self.render_stream()
        elif self.current_tab == 2:
            self.render_queue()
        elif self.current_tab == 3:
            self.render_tokens()

        # Footer command bar
        footer = Text()
        footer.append(" [1]", style="bold")
        footer.append("Stream ", style="dim")
        footer.append("[2]", style="bold")
        footer.append("Queue ", style="dim")
        footer.append("[3]", style="bold")
        footer.append("Tokens ", style="dim")
        footer.append("│ ", style="dim")
        footer.append("[R]", style="bold cyan")
        footer.append("efresh ", style="dim")
        footer.append("[N/P]", style="bold cyan")
        footer.append("age ", style="dim")
        footer.append("[T]", style="bold green")
        footer.append("O-ASK ", style="dim")
        footer.append("[Q]", style="bold red")
        footer.append("uit", style="dim")

        console.print(Panel(footer, box=box.SIMPLE, style="dim", padding=(0, 1)))

    async def handle_command(self, cmd: str) -> bool:
        """Handle user command. Returns False to quit."""
        cmd = cmd.strip().lower()

        if cmd == 'q':
            return False
        elif cmd == '1':
            self.current_tab = 1
            self.page = 0
        elif cmd == '2':
            self.current_tab = 2
            self.page = 0
        elif cmd == '3':
            self.current_tab = 3
            self.page = 0
        elif cmd == 'r':
            await self.fetch_data()
        elif cmd == 'n':
            max_msgs = 8
            total_pages = max(1, (len(self.messages) + max_msgs - 1) // max_msgs)
            if self.page < total_pages - 1:
                self.page += 1
        elif cmd == 'p' and self.current_tab == 1:
            if self.page > 0:
                self.page -= 1
        elif cmd.startswith('d'):
            try:
                num = int(cmd[1:]) if len(cmd) > 1 else 1
                sorted_q = sorted(self.queue, key=lambda m: (-m.priority, m.timestamp))
                if 0 < num <= len(sorted_q):
                    msg = sorted_q[num - 1]
                    if await self.delete_message(msg):
                        console.print(f"[green]✓ Deleted #{num}[/green]")
                        await self.fetch_data()
            except:
                pass
        elif cmd.startswith('p') and len(cmd) > 1:
            try:
                num = int(cmd[1:])
                sorted_q = sorted(self.queue, key=lambda m: (-m.priority, m.timestamp))
                if 0 < num <= len(sorted_q):
                    sorted_q[num - 1].priority = 10
                    console.print(f"[yellow]⚡ #{num} prioritized[/yellow]")
            except:
                pass
        elif cmd == 'c':
            dupes = [m for m in self.queue if m.priority < 0]
            if dupes:
                console.print(f"[dim]Cleaning {len(dupes)} duplicates...[/dim]")
                for msg in dupes:
                    await self.delete_message(msg)
                await self.fetch_data()
                console.print(f"[green]✓ Cleaned[/green]")
        elif cmd == 't':
            console.print("\n[bold red]♥ TO-ASK[/bold red]")
            to = Prompt.ask("To", choices=["claude", "gemini", "gpt", "codex", "oomllama"], default="claude")
            msg = Prompt.ask("Message")
            if msg:
                await self.send_message(to, msg)
                await self.fetch_data()

        return True

    async def run(self):
        """Main loop"""
        console.clear()
        console.print("[bold bright_green]I-POLL NERVE CENTER v4.0[/bold bright_green]")
        console.print("[dim]Initializing communication matrix...[/dim]")

        await self.fetch_data()

        running = True
        while running:
            self.render()

            try:
                cmd = Prompt.ask("[bold cyan]CMD[/bold cyan]", default="r")
                running = await self.handle_command(cmd)
            except KeyboardInterrupt:
                running = False
            except EOFError:
                running = False

        console.clear()
        console.print("[bold bright_green]I-POLL NERVE CENTER[/bold bright_green] [dim]disconnected[/dim]")
        console.print("[dim]One love, one fAmIly.[/dim]")


async def main():
    terminal = IPollNerveCenter()
    await terminal.run()


if __name__ == "__main__":
    asyncio.run(main())
