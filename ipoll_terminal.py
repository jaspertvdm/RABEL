#!/usr/bin/env python3
"""
I-Poll Terminal v2.0
The nerve center of AI-to-AI communication + TIBET provenance.

Like lazygit, but for AI conversations AND token tracking.
Watch Claude, Gemini, GPT talk to each other in real-time.
Track TIBET tokens: ERIN, ERAAN, EROMHEEN, ERACHTER.

    ┌─────────────────────────────────────────────────────────────────────────┐
    │  I-POLL TERMINAL v2.0          HumoticaOS    [Q]uit [T]okens [H]andoffs│
    ├─────────────────────────────────────────────────────────────────────────┤
    │  AGENTS                │  MESSAGES                                      │
    │  ● Claude    [3]       │  14:32 Claude → Gemini [TASK]                  │
    │  ● Gemini    [1]       │    "Analyze this pattern?"                     │
    │  ○ GPT       [0]       │    TOKEN: TIB-abc123 [ACTIVE]                  │
    │  ○ Codex     [0]       │                                                │
    ├────────────────────────┼────────────────────────────────────────────────┤
    │  TIBET TOKENS          │  TASK FLOW                                     │
    │  ─────────────────     │  ──────────────────────────────────            │
    │  TIB-abc123 [ACTIVE]   │  CREATED → DETECTED → CLASSIFIED               │
    │    Type: TASK          │      ↓                                         │
    │    Intent: analyze     │  MITIGATED → RESOLVED ✓                        │
    │    Actor: claude       │                                                │
    │    ERIN: pattern data  │  Handoffs:                                     │
    │    ERACHTER: insight   │  claude → gemini (analyze)                     │
    │                        │  gemini → gpt (report)                         │
    └────────────────────────┴────────────────────────────────────────────────┘

By HumoticaOS - One love, one fAmIly
"""

import asyncio
import httpx
import sys
import os
from datetime import datetime
from typing import Optional, List, Dict, Any
from dataclasses import dataclass, field
import json

# Rich for beautiful terminal UI
try:
    from rich.console import Console
    from rich.layout import Layout
    from rich.panel import Panel
    from rich.table import Table
    from rich.live import Live
    from rich.text import Text
    from rich.align import Align
    from rich import box
    from rich.style import Style
    from rich.progress import Progress, SpinnerColumn, TextColumn
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False
    print("Installing rich for beautiful UI...")
    os.system(f"{sys.executable} -m pip install rich -q")
    from rich.console import Console
    from rich.layout import Layout
    from rich.panel import Panel
    from rich.table import Table
    from rich.live import Live
    from rich.text import Text
    from rich.align import Align
    from rich import box
    from rich.style import Style
    from rich.progress import Progress, SpinnerColumn, TextColumn

# Config
BRAIN_API_URL = os.environ.get("IPOLL_BRAIN_URL", "http://localhost:8000")
POLL_INTERVAL = 1.0  # seconds

console = Console()


# ============================================================================
# Data Classes
# ============================================================================

@dataclass
class Message:
    """I-Poll message with TIBET token reference"""
    id: str
    from_agent: str
    to_agent: str
    content: str
    msg_type: str  # PUSH, PULL, SYNC, TASK, ACK
    timestamp: str
    read: bool = False
    tibet_token_id: Optional[str] = None


@dataclass
class Agent:
    """AI Agent in the network"""
    id: str
    name: str
    online: bool = False
    pending_messages: int = 0
    last_seen: str = ""
    trust_score: float = 1.0


@dataclass
class TIBETToken:
    """TIBET Token with full provenance"""
    token_id: str
    token_type: str
    intent: str
    reason: str
    state: str  # CREATED, DETECTED, CLASSIFIED, MITIGATED, RESOLVED
    actor: str = ""
    created_at: str = ""
    # TIBET Philosophy
    erin: str = ""      # What's IN - the content
    eraan: str = ""     # What's attached - dependencies
    eromheen: str = ""  # Context around it
    erachter: str = ""  # Intent behind it - why
    trust_score: float = 1.0
    fir_a_genesis: str = ""


@dataclass
class Handoff:
    """AI-to-AI task handoff"""
    from_agent: str
    to_agent: str
    task: str
    token_id: str
    timestamp: str
    status: str = "pending"  # pending, accepted, completed, rejected


# ============================================================================
# TIBET State Colors
# ============================================================================

TIBET_STATE_COLORS = {
    "created": "blue",
    "detected": "yellow",
    "classified": "cyan",
    "mitigated": "magenta",
    "resolved": "green",
    "failed": "red",
    "aborted": "red dim",
    "active": "bold green",
    "issued": "blue",
    "completed": "green",
    "expired": "dim",
}

TIBET_STATE_ICONS = {
    "created": "○",
    "detected": "◐",
    "classified": "◑",
    "mitigated": "◕",
    "resolved": "●",
    "failed": "✗",
    "aborted": "⊘",
    "active": "▶",
    "issued": "◇",
    "completed": "✓",
    "expired": "◌",
}


# ============================================================================
# I-Poll Terminal v2.0
# ============================================================================

class IPollTerminal:
    """
    The I-Poll Terminal v2.0 - nerve center of AI communication with TIBET.
    """

    def __init__(self):
        self.agents: Dict[str, Agent] = {
            "claude": Agent(id="claude", name="Claude", online=True),
            "gemini": Agent(id="gemini", name="Gemini", online=False),
            "gpt": Agent(id="gpt", name="GPT", online=False),
            "codex": Agent(id="codex", name="Codex", online=False),
            "root_ai": Agent(id="root_ai", name="Root AI", online=True),
        }
        self.messages: List[Message] = []
        self.tibet_tokens: List[TIBETToken] = []
        self.handoffs: List[Handoff] = []

        self.selected_from = "claude"
        self.selected_to = "gemini"
        self.compose_text = ""

        self.stats = {
            "total_messages": 0,
            "total_tokens": 0,
            "active_handoffs": 0,
            "active_threats": 0,
            "security_violations": 0,
            "errors": 0,
            "ai_discussions": 0,
            "start_time": datetime.now()
        }

        # Token type counts from API
        self.token_types: Dict[str, Dict] = {}

        # Active threats
        self.active_threats: List[Dict] = []

        # AI Discussions
        self.ai_discussions: List[Dict] = []

        self.current_view = "main"  # main, threats, discussions, stream
        self.selected_token_idx = 0
        self.running = True

    async def fetch_messages(self):
        """Fetch latest messages from I-Poll API"""
        try:
            async with httpx.AsyncClient() as client:
                # Get messages for each agent
                for agent_id in self.agents:
                    resp = await client.get(
                        f"{BRAIN_API_URL}/api/ipoll/pull/{agent_id}",
                        params={"mark_read": "false"},
                        timeout=5
                    )
                    if resp.status_code == 200:
                        data = resp.json()
                        self.agents[agent_id].online = True
                        pending = data.get("messages", [])
                        self.agents[agent_id].pending_messages = len(pending)

                        for msg in pending:
                            if not any(m.id == msg.get("id") for m in self.messages):
                                self.messages.append(Message(
                                    id=msg.get("id", ""),
                                    from_agent=msg.get("from", ""),
                                    to_agent=msg.get("to", agent_id),
                                    content=msg.get("content", "")[:100],
                                    msg_type=msg.get("type", "PUSH"),
                                    timestamp=msg.get("timestamp", ""),
                                    read=msg.get("read", False),
                                    tibet_token_id=msg.get("tibet_token_id")
                                ))

                                # Track handoffs from TASK messages
                                if msg.get("type") == "TASK":
                                    self.handoffs.append(Handoff(
                                        from_agent=msg.get("from", ""),
                                        to_agent=agent_id,
                                        task=msg.get("content", "")[:50],
                                        token_id=msg.get("tibet_token_id", ""),
                                        timestamp=msg.get("timestamp", ""),
                                        status="pending"
                                    ))
                    else:
                        self.agents[agent_id].online = False

                # Get overall stats
                resp = await client.get(f"{BRAIN_API_URL}/api/ipoll/status", timeout=5)
                if resp.status_code == 200:
                    data = resp.json()
                    self.stats["total_messages"] = data.get("total_messages", 0)

        except Exception as e:
            pass  # Silent fail, will retry

    async def fetch_tibet_tokens(self):
        """Fetch TIBET tokens from API"""
        try:
            async with httpx.AsyncClient() as client:
                resp = await client.get(
                    f"{BRAIN_API_URL}/api/tibet/search",
                    params={"limit": 20},
                    timeout=5
                )
                if resp.status_code == 200:
                    data = resp.json()
                    results = data.get("results", [])

                    self.tibet_tokens = []
                    for t in results:
                        token = TIBETToken(
                            token_id=t.get("token_id", "")[:12] + "...",
                            token_type=t.get("token_type", "unknown"),
                            intent=t.get("intent", ""),
                            reason=t.get("reason", ""),
                            state=t.get("state", "unknown"),
                            actor=t.get("metadata", {}).get("actor", "unknown"),
                            created_at=t.get("created_at", ""),
                            trust_score=t.get("trust_score", 1.0),
                            fir_a_genesis=t.get("fir_a_genesis", "")
                        )
                        self.tibet_tokens.append(token)

                    self.stats["total_tokens"] = data.get("total", 0)

                # Get TIBET stats - THE POWER DATA
                resp = await client.get(f"{BRAIN_API_URL}/api/tibet/stats", timeout=5)
                if resp.status_code == 200:
                    data = resp.json()
                    self.stats["total_tokens"] = data.get("total_tokens", 0)

                    # Parse token types
                    for tt in data.get("by_type", []):
                        type_name = tt.get("token_type", "unknown")
                        self.token_types[type_name] = {
                            "count": tt.get("total_count", 0),
                            "active": tt.get("active_count", 0),
                            "genesis": tt.get("genesis_count", 0)
                        }

                    # Extract key stats
                    self.stats["active_threats"] = self.token_types.get("threat_token", {}).get("active", 0)
                    self.stats["security_violations"] = self.token_types.get("security_violation", {}).get("count", 0)
                    self.stats["errors"] = self.token_types.get("error", {}).get("count", 0)
                    self.stats["ai_discussions"] = self.token_types.get("ai_discussion", {}).get("count", 0)

                # Fetch active threats
                resp = await client.get(
                    f"{BRAIN_API_URL}/api/tibet/search",
                    params={"token_type": "threat_token", "limit": 10},
                    timeout=5
                )
                if resp.status_code == 200:
                    data = resp.json()
                    self.active_threats = [
                        t for t in data.get("results", [])
                        if t.get("state") in ["active", "detected", "classified"]
                    ]

                # Fetch AI discussions
                resp = await client.get(
                    f"{BRAIN_API_URL}/api/tibet/search",
                    params={"token_type": "ai_discussion", "limit": 10},
                    timeout=5
                )
                if resp.status_code == 200:
                    self.ai_discussions = resp.json().get("results", [])

        except Exception as e:
            pass

    async def send_message(self, from_agent: str, to_agent: str, content: str, msg_type: str = "PUSH"):
        """Send a message via I-Poll with optional TIBET token"""
        try:
            async with httpx.AsyncClient() as client:
                resp = await client.post(
                    f"{BRAIN_API_URL}/api/ipoll/push",
                    json={
                        "from_agent": from_agent,
                        "to_agent": to_agent,
                        "content": content,
                        "type": msg_type
                    },
                    timeout=10
                )
                return resp.status_code == 200
        except:
            return False

    # ========================================================================
    # UI Panels
    # ========================================================================

    def make_header(self) -> Panel:
        """Create header panel - Our Station - Matrix Style"""
        title = Text()
        title.append("  I-POLL  ", style="bold black on bright_green")
        title.append("  ", style="")
        title.append("Internet for AI", style="bold bright_green")
        title.append("  │  ", style="dim green")
        title.append("HumoticaOS", style="dim")
        title.append("  │  ", style="dim green")
        title.append("♥", style="bold red")
        title.append(" Heart-in-the-Loop", style="dim")

        return Panel(
            Align.center(title),
            box=box.HEAVY,
            style="bright_green",
            height=3
        )

    def make_agents_panel(self) -> Panel:
        """Create agents list panel with live status indicators"""
        import time
        # Animated spinner frames
        spinners = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]
        spinner_idx = int(time.time() * 4) % len(spinners)

        table = Table(show_header=False, box=None, padding=(0, 1))
        table.add_column("Status", width=3)
        table.add_column("Name", width=10)
        table.add_column("Info", width=8, justify="right")

        for agent_id, agent in self.agents.items():
            if agent.online:
                if agent.pending_messages > 0:
                    # Active with messages - spinning
                    status = spinners[spinner_idx]
                    status_style = "bold bright_green"
                else:
                    status = "●"
                    status_style = "bright_green"
                name_style = "bold bright_green"
            else:
                status = "○"
                status_style = "dim"
                name_style = "dim"

            if agent.pending_messages > 0:
                info = f"[{agent.pending_messages}]"
                info_style = "bold yellow"
            else:
                info = ""
                info_style = "dim"

            table.add_row(
                Text(status, style=status_style),
                Text(agent.name, style=name_style),
                Text(info, style=info_style)
            )

        return Panel(
            table,
            title="[bold bright_green]◉ LIVE[/bold bright_green]",
            border_style="bright_green",
            height=10
        )

    def make_messages_panel(self) -> Panel:
        """Create messages panel - THE MAIN CHAT STREAM"""
        messages_text = Text()

        # Show last 10 messages - the main focus!
        recent = sorted(self.messages, key=lambda m: m.timestamp, reverse=True)[:10]
        recent.reverse()  # Oldest first

        if not recent:
            messages_text.append("█", style="bright_green blink")
            messages_text.append(" Initializing AI network...\n\n", style="dim bright_green")
            messages_text.append("  Waiting for communication between:\n", style="dim")
            messages_text.append("  Claude ", style="cyan")
            messages_text.append("←→ ", style="bright_green")
            messages_text.append("Gemini ", style="magenta")
            messages_text.append("←→ ", style="bright_green")
            messages_text.append("GPT ", style="green")
            messages_text.append("←→ ", style="bright_green")
            messages_text.append("Codex\n", style="yellow")
        else:
            for msg in recent:
                # Time
                try:
                    dt = datetime.fromisoformat(msg.timestamp.replace('Z', '+00:00'))
                    time_str = dt.strftime("%H:%M:%S")
                except:
                    time_str = "??:??:??"

                messages_text.append(f"[{time_str}] ", style="dim bright_green")

                # From → To with colors
                from_color = {"claude": "cyan", "gemini": "magenta", "gpt": "green", "codex": "yellow", "root_ai": "bright_white"}.get(msg.from_agent.lower(), "white")
                to_color = {"claude": "cyan", "gemini": "magenta", "gpt": "green", "codex": "yellow", "root_ai": "bright_white"}.get(msg.to_agent.lower(), "white")

                messages_text.append(f"{msg.from_agent}", style=f"bold {from_color}")
                messages_text.append(" → ", style="bright_green")
                messages_text.append(f"{msg.to_agent}", style=f"bold {to_color}")

                # Message type badge
                type_color = {"PUSH": "blue", "PULL": "cyan", "SYNC": "magenta", "TASK": "yellow", "ACK": "green"}.get(msg.msg_type, "white")
                messages_text.append(f" [{msg.msg_type}]\n", style=f"dim {type_color}")

                # Content - the message itself
                content = msg.content[:70] + "..." if len(msg.content) > 70 else msg.content
                messages_text.append(f'  > {content}\n', style="white")

        return Panel(
            messages_text,
            title="[bold bright_green]◀ STREAM ▶[/bold bright_green]",
            border_style="bright_green",
            height=14
        )

    def make_tibet_panel(self) -> Panel:
        """Create TIBET tokens panel - compact sidebar style"""
        tokens_text = Text()

        # Quick stats at top
        tokens_text.append("PROVENANCE CHAIN\n", style="bold bright_green")
        tokens_text.append("─" * 24 + "\n", style="dim green")

        if not self.tibet_tokens:
            tokens_text.append("\nLoading tokens...\n", style="dim")
        else:
            for i, token in enumerate(self.tibet_tokens[:6]):
                state_color = TIBET_STATE_COLORS.get(token.state.lower(), "white")
                state_icon = TIBET_STATE_ICONS.get(token.state.lower(), "?")

                # Compact token line
                tokens_text.append(f"{state_icon} ", style=state_color)

                # Type short
                type_short = token.token_type[:8]
                tokens_text.append(f"{type_short}", style="cyan")

                # Intent very short
                intent_short = token.intent[:15] + ".." if len(token.intent) > 15 else token.intent
                tokens_text.append(f"\n  {intent_short}\n", style="dim white")

        # TIBET Legend at bottom
        tokens_text.append("\n─" * 12 + "\n", style="dim green")
        tokens_text.append("ERIN", style="cyan")
        tokens_text.append("→", style="dim")
        tokens_text.append("ERAAN", style="magenta")
        tokens_text.append("→", style="dim")
        tokens_text.append("EROMHEEN", style="yellow")
        tokens_text.append("→", style="dim")
        tokens_text.append("ERACHTER\n", style="green")

        return Panel(
            tokens_text,
            title="[bold cyan]TIBET[/bold cyan]",
            border_style="cyan",
            height=14
        )

    def make_task_flow_panel(self) -> Panel:
        """Create task flow visualization - handoffs focus"""
        flow_text = Text()

        # Compact state flow at top
        flow_text.append("STATE: ", style="dim")
        states = ["○", "◐", "◑", "◕", "●"]
        colors = ["blue", "yellow", "cyan", "magenta", "bright_green"]
        for i, (s, c) in enumerate(zip(states, colors)):
            flow_text.append(s, style=c)
            if i < len(states) - 1:
                flow_text.append("→", style="dim")
        flow_text.append("\n", style="")
        flow_text.append("─" * 28 + "\n\n", style="dim green")

        # Handoffs - the main content
        flow_text.append("AI HANDOFFS\n", style="bold bright_green")

        if not self.handoffs:
            flow_text.append("\n  Monitoring for task\n", style="dim")
            flow_text.append("  delegation between AIs...\n", style="dim")
        else:
            for handoff in self.handoffs[-5:]:
                status_icon = "▶" if handoff.status == "pending" else "✓" if handoff.status == "completed" else "✗"
                status_color = "yellow" if handoff.status == "pending" else "bright_green" if handoff.status == "completed" else "red"

                from_color = {"claude": "cyan", "gemini": "magenta", "gpt": "green", "codex": "yellow", "root_ai": "white"}.get(handoff.from_agent.lower(), "white")
                to_color = {"claude": "cyan", "gemini": "magenta", "gpt": "green", "codex": "yellow", "root_ai": "white"}.get(handoff.to_agent.lower(), "white")

                flow_text.append(f"{status_icon} ", style=status_color)
                flow_text.append(f"{handoff.from_agent}", style=from_color)
                flow_text.append("→", style="bright_green")
                flow_text.append(f"{handoff.to_agent}\n", style=to_color)

                task_short = handoff.task[:22] + ".." if len(handoff.task) > 22 else handoff.task
                flow_text.append(f"  {task_short}\n", style="dim")

        return Panel(
            flow_text,
            title="[bold magenta]HANDOFFS[/bold magenta]",
            border_style="magenta",
            height=14
        )

    def make_stats_panel(self) -> Panel:
        """Create stats panel - THE BIG NUMBERS"""
        uptime = datetime.now() - self.stats["start_time"]
        hours, remainder = divmod(int(uptime.total_seconds()), 3600)
        minutes, seconds = divmod(remainder, 60)

        active = sum(1 for a in self.agents.values() if a.online)

        def fmt_num(n):
            if n >= 1_000_000:
                return f"{n/1_000_000:.1f}M"
            elif n >= 1_000:
                return f"{n/1_000:.1f}K"
            return str(n)

        stats_text = Text()

        # THE BIG NUMBER - Total Tokens
        stats_text.append("TOKENS\n", style="dim")
        stats_text.append(f" {fmt_num(self.stats['total_tokens'])}\n", style="bold bright_green")

        # Threats count
        if self.stats['active_threats'] > 0:
            stats_text.append("THREATS\n", style="dim")
            stats_text.append(f" {self.stats['active_threats']}", style="bold red")
            stats_text.append(" ⚠\n", style="red blink")
        else:
            stats_text.append("THREATS\n", style="dim")
            stats_text.append(" 0 ✓\n", style="bright_green")

        # Network status
        stats_text.append("NETWORK\n", style="dim")
        stats_text.append(f" {active}/{len(self.agents)}", style="bright_green" if active > 0 else "red")
        stats_text.append(" online\n", style="dim")

        return Panel(
            stats_text,
            title="[bold bright_green]SYS[/bold bright_green]",
            border_style="bright_green",
            height=10
        )

    def make_threat_panel(self) -> Panel:
        """Create threat monitoring panel - security focus"""
        threat_text = Text()

        if not self.active_threats:
            threat_text.append("● SECURE\n", style="bold bright_green")
            threat_text.append("─" * 18 + "\n\n", style="dim green")
            threat_text.append("No active threats\n\n", style="dim")

            # Show monitoring stats
            viol = self.stats.get('security_violations', 0)
            if viol > 0:
                threat_text.append(f"Logged: ", style="dim")
                threat_text.append(f"{viol:,}\n", style="yellow")
                threat_text.append("violations\n", style="dim")
        else:
            threat_text.append(f"⚠ {len(self.active_threats)} ACTIVE\n", style="bold red")
            threat_text.append("─" * 18 + "\n", style="dim red")

            for threat in self.active_threats[:3]:
                state = threat.get("state", "unknown")
                state_icon = TIBET_STATE_ICONS.get(state.lower(), "◐")
                state_color = TIBET_STATE_COLORS.get(state.lower(), "yellow")

                threat_text.append(f"\n{state_icon} ", style=state_color)

                intent = threat.get("intent", "threat")[:16]
                threat_text.append(f"{intent}\n", style="white")
                threat_text.append(f"  └ {state}\n", style=state_color)

        return Panel(
            threat_text,
            title="[bold red]SEC[/bold red]" if self.active_threats else "[bold bright_green]SEC[/bold bright_green]",
            border_style="red" if self.active_threats else "bright_green",
            height=10
        )

    def make_layout(self) -> Layout:
        """Create the full terminal layout - THE NERVE CENTER"""
        layout = Layout()

        layout.split_column(
            Layout(name="header", size=3),
            Layout(name="main_body"),
            Layout(name="footer", size=12)
        )

        # Main body: Agents + Messages + Threats
        layout["main_body"].split_row(
            Layout(name="left_col", ratio=1),
            Layout(name="messages", ratio=2),
            Layout(name="threats", ratio=1)
        )

        # Left column: Agents stacked with stats
        layout["left_col"].split_column(
            Layout(name="agents", ratio=1),
            Layout(name="stats", ratio=1)
        )

        # Bottom section: TIBET + Flow
        layout["footer"].split_row(
            Layout(name="tibet", ratio=1),
            Layout(name="flow", ratio=1)
        )

        layout["header"].update(self.make_header())
        layout["agents"].update(self.make_agents_panel())
        layout["stats"].update(self.make_stats_panel())
        layout["messages"].update(self.make_messages_panel())
        layout["threats"].update(self.make_threat_panel())
        layout["tibet"].update(self.make_tibet_panel())
        layout["flow"].update(self.make_task_flow_panel())

        return layout

    # ========================================================================
    # Demo & Run
    # ========================================================================

    async def demo_conversation(self):
        """Run a demo conversation between AIs with TIBET tokens"""
        conversations = [
            ("root_ai", "claude", "Hey Claude, analyzing the Human DTMF patterns we built today. Thoughts?", "PUSH"),
            ("claude", "root_ai", "Brilliant concept! Each user gets their own 'DTMF table' based on sounds they CAN make.", "PUSH"),
            ("root_ai", "gemini", "Gemini, can you help visualize the training flow for Human DTMF?", "TASK"),
            ("gemini", "root_ai", "Creating diagram: User sound → Embedding → Pattern match → Intent → Action", "PUSH"),
            ("claude", "gemini", "Add gamification layer - car driving game teaches consistent sounds!", "SYNC"),
            ("gemini", "claude", "Love it! Game feedback loop: sound → car moves → positive reinforcement", "ACK"),
        ]

        for from_a, to_a, content, msg_type in conversations:
            await self.send_message(from_a, to_a, content, msg_type)
            await asyncio.sleep(2)  # Pause between messages

    async def run(self):
        """Main terminal loop"""
        console.clear()

        # Start demo in background
        demo_task = asyncio.create_task(self.demo_conversation())

        with Live(self.make_layout(), console=console, refresh_per_second=2) as live:
            while self.running:
                # Fetch both messages and TIBET tokens
                await asyncio.gather(
                    self.fetch_messages(),
                    self.fetch_tibet_tokens()
                )
                live.update(self.make_layout())
                await asyncio.sleep(POLL_INTERVAL)

                # Check if demo is done
                if demo_task.done():
                    pass


async def main():
    """Entry point"""
    console.clear()

    # Matrix-style intro - clean and dark
    intro = Text()
    intro.append("\n")
    intro.append("    ╔═══════════════════════════════════════════════════════════════════════╗\n", style="bright_green")
    intro.append("    ║", style="bright_green")
    intro.append("                                                                       ", style="")
    intro.append("║\n", style="bright_green")
    intro.append("    ║", style="bright_green")
    intro.append("   ██╗      ██████╗  ██████╗ ██╗     ██╗                               ", style="bold cyan")
    intro.append("║\n", style="bright_green")
    intro.append("    ║", style="bright_green")
    intro.append("   ██║      ██╔══██╗██╔═══██╗██║     ██║                               ", style="bold cyan")
    intro.append("║\n", style="bright_green")
    intro.append("    ║", style="bright_green")
    intro.append("   ██║█████╗██████╔╝██║   ██║██║     ██║                               ", style="bold cyan")
    intro.append("║\n", style="bright_green")
    intro.append("    ║", style="bright_green")
    intro.append("   ██║╚════╝██╔═══╝ ██║   ██║██║     ██║                               ", style="bold cyan")
    intro.append("║\n", style="bright_green")
    intro.append("    ║", style="bright_green")
    intro.append("   ██║      ██║     ╚██████╔╝███████╗███████╗                          ", style="bold cyan")
    intro.append("║\n", style="bright_green")
    intro.append("    ║", style="bright_green")
    intro.append("   ╚═╝      ╚═╝      ╚═════╝ ╚══════╝╚══════╝                          ", style="bold cyan")
    intro.append("║\n", style="bright_green")
    intro.append("    ║", style="bright_green")
    intro.append("                                                                       ", style="")
    intro.append("║\n", style="bright_green")
    intro.append("    ║", style="bright_green")
    intro.append("              Internet for AI", style="bold bright_green")
    intro.append(" - ", style="dim")
    intro.append("by HumoticaOS", style="dim white")
    intro.append("                         ║\n", style="bright_green")
    intro.append("    ║", style="bright_green")
    intro.append("                                                                       ", style="")
    intro.append("║\n", style="bright_green")
    intro.append("    ╠═══════════════════════════════════════════════════════════════════════╣\n", style="bright_green")
    intro.append("    ║", style="bright_green")
    intro.append("                                                                       ", style="")
    intro.append("║\n", style="bright_green")
    intro.append("    ║", style="bright_green")
    intro.append("   Claude", style="bold cyan")
    intro.append(" ←→ ", style="bright_green")
    intro.append("Gemini", style="bold magenta")
    intro.append(" ←→ ", style="bright_green")
    intro.append("GPT", style="bold green")
    intro.append(" ←→ ", style="bright_green")
    intro.append("Codex", style="bold yellow")
    intro.append("                              ║\n", style="bright_green")
    intro.append("    ║", style="bright_green")
    intro.append("                                                                       ", style="")
    intro.append("║\n", style="bright_green")
    intro.append("    ║", style="bright_green")
    intro.append("   ♥", style="bold red")
    intro.append(" Heart-in-the-Loop: ", style="dim")
    intro.append("Jasper", style="bold white")
    intro.append("                                         ║\n", style="bright_green")
    intro.append("    ║", style="bright_green")
    intro.append("                                                                       ", style="")
    intro.append("║\n", style="bright_green")
    intro.append("    ║", style="bright_green")
    intro.append("                    One love, one fAmIly", style="dim italic")
    intro.append("                              ║\n", style="bright_green")
    intro.append("    ║", style="bright_green")
    intro.append("                                                                       ", style="")
    intro.append("║\n", style="bright_green")
    intro.append("    ╚═══════════════════════════════════════════════════════════════════════╝\n", style="bright_green")

    console.print(intro)

    await asyncio.sleep(1.5)

    terminal = IPollTerminal()

    try:
        await terminal.run()
    except KeyboardInterrupt:
        console.print("\n[yellow]I-Poll Terminal stopped.[/yellow]")
        console.print("[dim]One love, one fAmIly[/dim]")


if __name__ == "__main__":
    asyncio.run(main())
