"""
RABEL MCP Server - The Brain Layer
===================================

Exposes RABEL memory capabilities via MCP protocol.
"""

import json
import sqlite3
import hashlib
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict, Any

import mcp.types as types
from mcp.server import Server
from mcp.server.stdio import stdio_server

# Optional: Ollama for embeddings (graceful fallback if not available)
try:
    import requests
    OLLAMA_AVAILABLE = True
except ImportError:
    OLLAMA_AVAILABLE = False

# Optional: sqlite-vec for vector search
try:
    import sqlite_vec
    from sqlite_vec import serialize_float32
    SQLITE_VEC_AVAILABLE = True
except ImportError:
    SQLITE_VEC_AVAILABLE = False

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("rabel")

# Initialize MCP server
mcp = Server("rabel")

# =============================================================================
# RABEL CORE - Simplified for MCP
# =============================================================================

class RABELCore:
    """Core RABEL functionality for MCP server."""

    def __init__(self, db_path: str = None):
        if db_path is None:
            # Default to user's home directory
            home = Path.home()
            db_path = str(home / ".rabel" / "memories.sqlite")

        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.embedding_dim = 768
        self.ollama_url = "http://localhost:11434"
        self._init_db()

    def _init_db(self):
        """Initialize SQLite database."""
        self.conn = sqlite3.connect(str(self.db_path))

        # Load sqlite-vec if available
        if SQLITE_VEC_AVAILABLE:
            self.conn.enable_load_extension(True)
            sqlite_vec.load(self.conn)
            self.conn.enable_load_extension(False)

        # Create tables
        self.conn.executescript("""
            CREATE TABLE IF NOT EXISTS memories (
                id TEXT PRIMARY KEY,
                content TEXT NOT NULL,
                scope TEXT DEFAULT 'general',
                created_at TEXT NOT NULL,
                metadata TEXT
            );

            CREATE TABLE IF NOT EXISTS relations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                subject TEXT NOT NULL,
                predicate TEXT NOT NULL,
                object TEXT NOT NULL,
                created_at TEXT NOT NULL
            );

            CREATE INDEX IF NOT EXISTS idx_memories_scope ON memories(scope);
            CREATE INDEX IF NOT EXISTS idx_relations_subject ON relations(subject);
        """)

        # Create vector table if sqlite-vec available
        if SQLITE_VEC_AVAILABLE:
            try:
                self.conn.execute("""
                    CREATE VIRTUAL TABLE IF NOT EXISTS memory_vectors USING vec0(
                        id TEXT PRIMARY KEY,
                        embedding FLOAT[768]
                    )
                """)
            except:
                pass  # Table might already exist

        self.conn.commit()

    def _get_embedding(self, text: str) -> Optional[List[float]]:
        """Get embedding from Ollama (if available)."""
        if not OLLAMA_AVAILABLE:
            return None
        try:
            response = requests.post(
                f"{self.ollama_url}/api/embeddings",
                json={"model": "nomic-embed-text", "prompt": text},
                timeout=30
            )
            response.raise_for_status()
            return response.json()["embedding"]
        except:
            return None

    def _generate_id(self, content: str) -> str:
        timestamp = datetime.now().isoformat()
        return hashlib.sha256(f"{content}{timestamp}".encode()).hexdigest()[:16]

    def add_memory(self, content: str, scope: str = "general", metadata: dict = None) -> str:
        """Add a memory."""
        memory_id = self._generate_id(content)
        created_at = datetime.now().isoformat()

        self.conn.execute(
            "INSERT INTO memories (id, content, scope, created_at, metadata) VALUES (?, ?, ?, ?, ?)",
            (memory_id, content, scope, created_at, json.dumps(metadata or {}))
        )

        # Add embedding if available
        if SQLITE_VEC_AVAILABLE:
            embedding = self._get_embedding(content)
            if embedding:
                self.conn.execute(
                    "INSERT INTO memory_vectors (id, embedding) VALUES (?, ?)",
                    (memory_id, serialize_float32(embedding))
                )

        self.conn.commit()
        return memory_id

    def search_memory(self, query: str, limit: int = 5) -> List[Dict]:
        """Search memories (semantic if available, else text search)."""
        if SQLITE_VEC_AVAILABLE:
            embedding = self._get_embedding(query)
            if embedding:
                results = self.conn.execute("""
                    SELECT v.id, v.distance, m.content, m.scope, m.created_at
                    FROM memory_vectors v
                    JOIN memories m ON v.id = m.id
                    WHERE v.embedding MATCH ? AND k = ?
                    ORDER BY v.distance
                """, (serialize_float32(embedding), limit)).fetchall()

                return [
                    {
                        "id": r[0],
                        "similarity": round(1.0 / (1.0 + abs(r[1])), 3),
                        "content": r[2],
                        "scope": r[3],
                        "created_at": r[4]
                    }
                    for r in results
                ]

        # Fallback: text search
        results = self.conn.execute(
            "SELECT id, content, scope, created_at FROM memories WHERE content LIKE ? LIMIT ?",
            (f"%{query}%", limit)
        ).fetchall()

        return [
            {"id": r[0], "content": r[1], "scope": r[2], "created_at": r[3]}
            for r in results
        ]

    def add_relation(self, subject: str, predicate: str, obj: str) -> int:
        """Add a graph relation."""
        cursor = self.conn.execute(
            "INSERT INTO relations (subject, predicate, object, created_at) VALUES (?, ?, ?, ?)",
            (subject, predicate, obj, datetime.now().isoformat())
        )
        self.conn.commit()
        return cursor.lastrowid

    def get_relations(self, subject: str = None, predicate: str = None) -> List[Dict]:
        """Query graph relations."""
        query = "SELECT subject, predicate, object FROM relations WHERE 1=1"
        params = []
        if subject:
            query += " AND subject = ?"
            params.append(subject)
        if predicate:
            query += " AND predicate = ?"
            params.append(predicate)

        results = self.conn.execute(query, params).fetchall()
        return [{"subject": r[0], "predicate": r[1], "object": r[2]} for r in results]

    def get_stats(self) -> Dict:
        """Get memory statistics."""
        total = self.conn.execute("SELECT COUNT(*) FROM memories").fetchone()[0]
        relations = self.conn.execute("SELECT COUNT(*) FROM relations").fetchone()[0]
        return {
            "total_memories": total,
            "total_relations": relations,
            "db_path": str(self.db_path),
            "vector_search": SQLITE_VEC_AVAILABLE,
            "ollama_available": OLLAMA_AVAILABLE
        }


# Soft Pipelines - Bilingual guidance
SOFT_PIPELINES = {
    "deploy": {
        "steps": ["submit", "review", "approve", "deploy"],
        "en": "Deploy: Submit → Review → Approve → Deploy",
        "nl": "Uitrollen: Indienen → Beoordelen → Goedkeuren → Uitrollen"
    },
    "create": {
        "steps": ["draft", "share", "review", "approve", "create"],
        "en": "Create: Draft → Share → Review → Approve → Create",
        "nl": "Maken: Concept → Delen → Beoordelen → Goedkeuren → Maken"
    },
    "solve_puzzle": {
        "steps": ["read", "analyze", "attempt", "verify", "document"],
        "en": "Puzzle: Read → Analyze → Attempt → Verify → Document",
        "nl": "Puzzel: Lezen → Analyseren → Proberen → Verifiëren → Documenteren"
    },
    "learn": {
        "steps": ["observe", "try", "fail", "understand", "master"],
        "en": "Learn: Observe → Try → Fail → Understand → Master",
        "nl": "Leren: Observeren → Proberen → Falen → Begrijpen → Beheersen"
    }
}

# Global RABEL instance
rabel: Optional[RABELCore] = None

def get_rabel() -> RABELCore:
    global rabel
    if rabel is None:
        rabel = RABELCore()
    return rabel


# =============================================================================
# MCP TOOLS
# =============================================================================

@mcp.list_tools()
async def list_tools() -> list[types.Tool]:
    """List available RABEL tools."""
    return [
        types.Tool(
            name="rabel_hello",
            description="Say hello from RABEL - test if it's working!",
            inputSchema={
                "type": "object",
                "properties": {},
                "required": []
            }
        ),
        types.Tool(
            name="rabel_add_memory",
            description="Add a memory to RABEL. Memories are searchable facts, experiences, or knowledge.",
            inputSchema={
                "type": "object",
                "properties": {
                    "content": {
                        "type": "string",
                        "description": "The memory content (what to remember)"
                    },
                    "scope": {
                        "type": "string",
                        "description": "Memory scope: user, agent, team, or general",
                        "default": "general"
                    }
                },
                "required": ["content"]
            }
        ),
        types.Tool(
            name="rabel_search",
            description="Search memories semantically. Ask questions like 'What do I know about X?'",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search query"
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Max results (default 5)",
                        "default": 5
                    }
                },
                "required": ["query"]
            }
        ),
        types.Tool(
            name="rabel_add_relation",
            description="Add a graph relation between entities. Example: 'Jasper' --father_of--> 'Storm'",
            inputSchema={
                "type": "object",
                "properties": {
                    "subject": {
                        "type": "string",
                        "description": "The subject entity"
                    },
                    "predicate": {
                        "type": "string",
                        "description": "The relationship type (e.g., 'father_of', 'part_of', 'knows')"
                    },
                    "object": {
                        "type": "string",
                        "description": "The object entity"
                    }
                },
                "required": ["subject", "predicate", "object"]
            }
        ),
        types.Tool(
            name="rabel_get_relations",
            description="Query graph relations. Find how entities are connected.",
            inputSchema={
                "type": "object",
                "properties": {
                    "subject": {
                        "type": "string",
                        "description": "Filter by subject (optional)"
                    },
                    "predicate": {
                        "type": "string",
                        "description": "Filter by relationship type (optional)"
                    }
                },
                "required": []
            }
        ),
        types.Tool(
            name="rabel_get_guidance",
            description="Get soft pipeline guidance for a task. Suggests steps without enforcing them.",
            inputSchema={
                "type": "object",
                "properties": {
                    "intent": {
                        "type": "string",
                        "description": "What you want to do: deploy, create, solve_puzzle, learn"
                    },
                    "lang": {
                        "type": "string",
                        "description": "Language: 'en' or 'nl' (default: en)",
                        "default": "en"
                    }
                },
                "required": ["intent"]
            }
        ),
        types.Tool(
            name="rabel_next_step",
            description="Get suggested next step based on what's been completed.",
            inputSchema={
                "type": "object",
                "properties": {
                    "intent": {
                        "type": "string",
                        "description": "The task type (deploy, create, solve_puzzle, learn)"
                    },
                    "completed": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Steps already completed"
                    }
                },
                "required": ["intent", "completed"]
            }
        ),
        types.Tool(
            name="rabel_stats",
            description="Get RABEL memory statistics.",
            inputSchema={
                "type": "object",
                "properties": {},
                "required": []
            }
        )
    ]


@mcp.call_tool()
async def call_tool(name: str, arguments: dict) -> list[types.TextContent]:
    """Handle tool calls."""

    try:
        if name == "rabel_hello":
            return [types.TextContent(
                type="text",
                text="🧠 Hello from RABEL!\n\nRecidive Active Brain Environment Layer\nMem0 inspired, locally evolved.\n\nYour local-first AI memory is ready!\nOne love, one fAmIly 💙"
            )]

        elif name == "rabel_add_memory":
            r = get_rabel()
            content = arguments["content"]
            scope = arguments.get("scope", "general")
            memory_id = r.add_memory(content, scope)
            return [types.TextContent(
                type="text",
                text=f"✅ Memory added!\nID: {memory_id}\nScope: {scope}\nContent: {content[:100]}..."
            )]

        elif name == "rabel_search":
            r = get_rabel()
            query = arguments["query"]
            limit = arguments.get("limit", 5)
            results = r.search_memory(query, limit)

            if not results:
                return [types.TextContent(type="text", text="No memories found.")]

            output = f"🔍 Found {len(results)} memories:\n\n"
            for i, mem in enumerate(results, 1):
                sim = mem.get('similarity', 'N/A')
                output += f"{i}. [{sim}] {mem['content'][:80]}...\n"
                output += f"   Scope: {mem['scope']}\n\n"

            return [types.TextContent(type="text", text=output)]

        elif name == "rabel_add_relation":
            r = get_rabel()
            subj = arguments["subject"]
            pred = arguments["predicate"]
            obj = arguments["object"]
            rel_id = r.add_relation(subj, pred, obj)
            return [types.TextContent(
                type="text",
                text=f"✅ Relation added!\n{subj} --{pred}--> {obj}"
            )]

        elif name == "rabel_get_relations":
            r = get_rabel()
            subj = arguments.get("subject")
            pred = arguments.get("predicate")
            relations = r.get_relations(subj, pred)

            if not relations:
                return [types.TextContent(type="text", text="No relations found.")]

            output = f"🔗 Found {len(relations)} relations:\n\n"
            for rel in relations:
                output += f"• {rel['subject']} --{rel['predicate']}--> {rel['object']}\n"

            return [types.TextContent(type="text", text=output)]

        elif name == "rabel_get_guidance":
            intent = arguments["intent"]
            lang = arguments.get("lang", "en")

            pipeline = SOFT_PIPELINES.get(intent, {
                "steps": ["think", "do", "share"],
                "en": "General: Think → Do → Share",
                "nl": "Algemeen: Denken → Doen → Delen"
            })

            guidance = pipeline.get(lang, pipeline.get("en", ""))
            steps = pipeline.get("steps", [])

            return [types.TextContent(
                type="text",
                text=f"📋 Soft Pipeline Guidance\n\nIntent: {intent}\n{guidance}\n\nSteps: {' → '.join(steps)}\n\n⚡ This is guidance, not enforcement!"
            )]

        elif name == "rabel_next_step":
            intent = arguments["intent"]
            completed = arguments.get("completed", [])

            pipeline = SOFT_PIPELINES.get(intent, {})
            steps = pipeline.get("steps", ["think", "do", "share"])
            remaining = [s for s in steps if s not in completed]

            if not remaining:
                return [types.TextContent(type="text", text="✅ All steps completed! Well done!")]

            return [types.TextContent(
                type="text",
                text=f"👉 Suggested next step: **{remaining[0]}**\n\nCompleted: {completed}\nRemaining: {remaining}"
            )]

        elif name == "rabel_stats":
            r = get_rabel()
            stats = r.get_stats()
            return [types.TextContent(
                type="text",
                text=f"📊 RABEL Statistics\n\n• Total memories: {stats['total_memories']}\n• Total relations: {stats['total_relations']}\n• Database: {stats['db_path']}\n• Vector search: {'✅' if stats['vector_search'] else '❌'}\n• Ollama: {'✅' if stats['ollama_available'] else '❌'}"
            )]

        else:
            return [types.TextContent(type="text", text=f"Unknown tool: {name}")]

    except Exception as e:
        logger.error(f"Tool error: {e}")
        return [types.TextContent(type="text", text=f"Error: {str(e)}")]


# =============================================================================
# MAIN
# =============================================================================

async def run():
    """Run the MCP server."""
    async with stdio_server() as (read_stream, write_stream):
        await mcp.run(
            read_stream,
            write_stream,
            mcp.create_initialization_options()
        )

def main():
    """Entry point."""
    import asyncio
    asyncio.run(run())

if __name__ == "__main__":
    main()
