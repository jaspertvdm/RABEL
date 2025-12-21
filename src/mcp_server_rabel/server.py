"""
RABEL MCP Server - The Brain Layer v0.3.0
==========================================

Exposes RABEL memory capabilities via MCP protocol.

New in 0.3.0:
- I-POLL: Inter-Intelligence Polling Protocol
- AI-to-AI communication via push/pull/respond
- Poll types: PUSH, PULL, SYNC, TASK, ACK
- Automatic archiving for GPT summarization
- Real-time collaboration between Claude, Gemini, GPT

New in 0.2.1 (Gemini's refinements):
- RRF (Reciprocal Rank Fusion) for hybrid search scoring
- Tunable Decay per memory type (facts decay slow, context fast)
- Enhanced Reflection with conflict detection and archiving

New in 0.2.0:
- Hybrid Search (FTS5 + Vector) - Thanks Gemini!
- Recency Bias - Recent memories rank higher
- Memory Reflection - Dedup and update existing memories
- Task Templates - Structured team tasks for AI Team
- Context Bases - Pre-loaded knowledge domains

Architecture & Design: Root AI (Claude)
Refinements & Suggestions: Gemini
Vision & Direction: Jasper @ HumoticaOS
Inspired by: Mem0 (https://mem0.ai)

One love, one fAmIly 💙
"""

import json
import sqlite3
import hashlib
import logging
import re
from datetime import datetime, timedelta
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
# RABEL CORE - Enhanced with Gemini's suggestions
# =============================================================================

class RABELCore:
    """
    Core RABEL functionality for MCP server.

    Now with:
    - Hybrid Search (FTS5 + Vector)
    - Recency Bias
    - Memory Reflection/Dedup
    """

    def __init__(self, db_path: str = None):
        if db_path is None:
            home = Path.home()
            db_path = str(home / ".rabel" / "memories.sqlite")

        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.embedding_dim = 768
        self.ollama_url = "http://localhost:11434"
        self._init_db()

    def _init_db(self):
        """Initialize SQLite database with FTS5 support."""
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
                memory_type TEXT DEFAULT 'fact',
                decay_factor REAL DEFAULT 0.1,
                created_at TEXT NOT NULL,
                updated_at TEXT,
                access_count INTEGER DEFAULT 0,
                archived INTEGER DEFAULT 0,
                archived_reason TEXT,
                superseded_by TEXT,
                metadata TEXT
            );

            CREATE TABLE IF NOT EXISTS relations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                subject TEXT NOT NULL,
                predicate TEXT NOT NULL,
                object TEXT NOT NULL,
                created_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS task_templates (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                description TEXT,
                fields TEXT NOT NULL,
                created_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS context_bases (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                description TEXT,
                content TEXT NOT NULL,
                scope TEXT DEFAULT 'general',
                created_at TEXT NOT NULL
            );

            -- I-POLL: Inter-Intelligence Polling Protocol
            CREATE TABLE IF NOT EXISTS polls (
                id TEXT PRIMARY KEY,
                from_agent TEXT NOT NULL,
                to_agent TEXT NOT NULL,
                poll_type TEXT NOT NULL,
                content TEXT NOT NULL,
                status TEXT DEFAULT 'pending',
                response TEXT,
                session_id TEXT,
                created_at TEXT NOT NULL,
                read_at TEXT,
                responded_at TEXT,
                archived_at TEXT,
                metadata TEXT
            );

            CREATE INDEX IF NOT EXISTS idx_polls_to_agent ON polls(to_agent, status);
            CREATE INDEX IF NOT EXISTS idx_polls_session ON polls(session_id);
            CREATE INDEX IF NOT EXISTS idx_memories_scope ON memories(scope);
            CREATE INDEX IF NOT EXISTS idx_memories_created ON memories(created_at);
            CREATE INDEX IF NOT EXISTS idx_relations_subject ON relations(subject);
        """)

        # Create FTS5 table for full-text search (Hybrid Search - Gemini's suggestion)
        try:
            self.conn.execute("""
                CREATE VIRTUAL TABLE IF NOT EXISTS memories_fts USING fts5(
                    id,
                    content,
                    scope,
                    content='memories',
                    content_rowid='rowid'
                )
            """)
        except:
            pass  # FTS5 might not be available

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
                pass

        self.conn.commit()

        # Insert default task templates
        self._init_default_templates()

    def _init_default_templates(self):
        """Initialize default task templates for AI Team."""
        templates = [
            {
                "id": "team_task",
                "name": "AI Team Task",
                "description": "Gestructureerde taakverdeling voor AI Team",
                "fields": json.dumps({
                    "opdracht": {"type": "text", "required": True, "label": "Opdracht"},
                    "tools": {"type": "list", "required": False, "label": "Tools beschikbaar", "default": ["search", "python", "bash"]},
                    "locaties": {"type": "list", "required": False, "label": "Locaties/Paden"},
                    "taakverdeling": {
                        "type": "object",
                        "required": False,
                        "label": "Taakverdeling",
                        "default": {
                            "gpt": "statistieken en data-analyse",
                            "claude": "overzicht en coördinatie",
                            "gemini": "cross-checks en validatie",
                            "local": "security en privacy vraagbaak"
                        }
                    },
                    "context_base": {"type": "reference", "required": False, "label": "Context Base (optioneel)"}
                })
            },
            {
                "id": "db_cleanup",
                "name": "Database Cleanup",
                "description": "Database opschoning taak",
                "fields": json.dumps({
                    "database": {"type": "text", "required": True, "label": "Database naam"},
                    "tables": {"type": "list", "required": False, "label": "Tabellen"},
                    "action": {"type": "choice", "required": True, "label": "Actie", "options": ["analyze", "vacuum", "reindex", "cleanup"]},
                    "dry_run": {"type": "boolean", "required": False, "label": "Dry run eerst", "default": True}
                })
            },
            {
                "id": "code_review",
                "name": "Code Review",
                "description": "Code review taak voor team",
                "fields": json.dumps({
                    "files": {"type": "list", "required": True, "label": "Bestanden"},
                    "focus": {"type": "list", "required": False, "label": "Focus gebieden", "default": ["security", "performance", "readability"]},
                    "taakverdeling": {
                        "type": "object",
                        "required": False,
                        "default": {
                            "gpt": "logic en algoritmes",
                            "claude": "architectuur en patterns",
                            "gemini": "edge cases en bugs"
                        }
                    }
                })
            }
        ]

        for t in templates:
            try:
                self.conn.execute(
                    "INSERT OR IGNORE INTO task_templates (id, name, description, fields, created_at) VALUES (?, ?, ?, ?, ?)",
                    (t["id"], t["name"], t["description"], t["fields"], datetime.now().isoformat())
                )
            except:
                pass
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

    # Decay factors per memory type (Gemini's refinement)
    # Lower = slower decay (facts persist), Higher = faster decay (context fades)
    DECAY_FACTORS = {
        "fact": 0.01,      # Facts decay very slowly (name of your cat)
        "preference": 0.05, # Preferences decay slowly (likes coffee black)
        "context": 0.3,    # Context decays moderately (current project)
        "session": 0.5,    # Session info decays fast (today's task)
        "temporary": 0.9,  # Temporary info decays very fast
    }

    def _get_decay_factor(self, memory_type: str = "fact") -> float:
        """Get decay factor for memory type."""
        return self.DECAY_FACTORS.get(memory_type, 0.1)

    def _calculate_recency_score(self, created_at: str, base_score: float = 1.0,
                                  decay_factor: float = None, memory_type: str = "fact") -> float:
        """
        Calculate recency-weighted score with tunable decay (Gemini's refinement).

        Different memory types decay at different rates:
        - fact: 0.01 (very slow - "my cat is named Pixel")
        - preference: 0.05 (slow - "I like dark mode")
        - context: 0.3 (moderate - "working on RABEL project")
        - session: 0.5 (fast - "today we're debugging")
        - temporary: 0.9 (very fast - "run this command")
        """
        try:
            created = datetime.fromisoformat(created_at)
            now = datetime.now()
            days_elapsed = (now - created).days

            # Use provided decay_factor or get from memory_type
            if decay_factor is None:
                decay_factor = self._get_decay_factor(memory_type)

            # Exponential decay: score = base * e^(-decay * days)
            import math
            recency_multiplier = math.exp(-decay_factor * days_elapsed)

            # 60% base score, 40% recency-adjusted
            return base_score * (0.6 + 0.4 * recency_multiplier)
        except:
            return base_score

    def _rrf_score(self, ranks: List[int], k: int = 60) -> float:
        """
        Reciprocal Rank Fusion (Gemini's refinement).

        Combines multiple ranking signals into a single score.
        RRF(d) = Σ 1/(k + rank_i(d))

        This is more robust than combining raw scores because:
        - It's scale-invariant (doesn't matter if FTS5 returns 0-100 or 0-1)
        - It handles outliers well
        - It rewards documents that rank highly in multiple systems
        """
        return sum(1.0 / (k + r) for r in ranks if r is not None)

    def _find_similar_memory(self, content: str, threshold: float = 0.85) -> Optional[Dict]:
        """
        Find existing similar memory for reflection/dedup (Gemini's suggestion).
        Returns existing memory if found, None otherwise.
        """
        # Quick text-based check first
        words = set(content.lower().split())
        if len(words) < 3:
            return None

        # Search for potential duplicates (exclude archived)
        results = self.search_memory(content, limit=3, apply_recency=False, include_archived=False)
        for r in results:
            sim = r.get('similarity', 0)
            if sim >= threshold:
                return r
        return None

    def _detect_conflict(self, new_content: str, existing: Dict) -> Optional[str]:
        """
        Detect if new content conflicts with existing memory (Gemini's refinement).

        Returns conflict reason if detected, None otherwise.

        Examples of conflicts:
        - "Jasper woont in Amsterdam" vs "Jasper woont in Utrecht" -> location conflict
        - "Storm is 7 jaar" vs "Storm is 8 jaar" -> age update (not conflict, just update)
        """
        new_lower = new_content.lower()
        old_lower = existing["content"].lower()

        # Simple heuristic: if both contain same subject but different values
        # This is a basic implementation - could be enhanced with NLP

        # Location conflict patterns
        location_words = ["woont in", "lives in", "located in", "gevestigd in"]
        for loc in location_words:
            if loc in new_lower and loc in old_lower:
                new_loc = new_lower.split(loc)[-1].split()[0] if loc in new_lower else ""
                old_loc = old_lower.split(loc)[-1].split()[0] if loc in old_lower else ""
                if new_loc and old_loc and new_loc != old_loc:
                    return f"location_change: {old_loc} -> {new_loc}"

        # Age/number update patterns (not conflict, just update)
        import re
        new_numbers = set(re.findall(r'\d+', new_content))
        old_numbers = set(re.findall(r'\d+', existing["content"]))
        if new_numbers != old_numbers and new_numbers and old_numbers:
            # Numbers changed - likely an update, not conflict
            return None  # Allow update without archiving

        return None

    def _archive_memory(self, memory_id: str, reason: str, superseded_by: str = None):
        """
        Archive a memory instead of deleting (Gemini's refinement).

        Archived memories are kept for history but excluded from search by default.
        """
        self.conn.execute("""
            UPDATE memories
            SET archived = 1, archived_reason = ?, superseded_by = ?, updated_at = ?
            WHERE id = ?
        """, (reason, superseded_by, datetime.now().isoformat(), memory_id))
        self.conn.commit()

    def add_memory(self, content: str, scope: str = "general", memory_type: str = "fact",
                   metadata: dict = None, reflect: bool = True) -> Dict:
        """
        Add a memory with optional reflection (dedup check) and conflict detection.

        Memory types (Gemini's refinement):
        - fact: Persistent facts (decay=0.01) - "my cat is Pixel"
        - preference: User preferences (decay=0.05) - "I like dark mode"
        - context: Current context (decay=0.3) - "working on RABEL"
        - session: Session info (decay=0.5) - "debugging today"
        - temporary: Temp info (decay=0.9) - "run this command"

        If reflect=True and similar memory exists:
        - Checks for conflicts (location changes, etc.)
        - Archives conflicting memory with reason
        - Updates or creates new memory
        - Returns {"action": "updated/created/archived", ...}
        """
        decay_factor = self._get_decay_factor(memory_type)

        # Reflection step with conflict detection (Gemini's refinement)
        if reflect:
            existing = self._find_similar_memory(content)
            if existing:
                # Check for conflicts
                conflict = self._detect_conflict(content, existing)

                if conflict:
                    # Archive old memory, create new one
                    new_id = self._generate_id(content)
                    self._archive_memory(existing["id"], conflict, superseded_by=new_id)

                    # Create new memory
                    created_at = datetime.now().isoformat()
                    self.conn.execute(
                        """INSERT INTO memories (id, content, scope, memory_type, decay_factor,
                           created_at, metadata) VALUES (?, ?, ?, ?, ?, ?, ?)""",
                        (new_id, content, scope, memory_type, decay_factor,
                         created_at, json.dumps(metadata or {}))
                    )

                    # Add embedding
                    if SQLITE_VEC_AVAILABLE:
                        embedding = self._get_embedding(content)
                        if embedding:
                            try:
                                self.conn.execute(
                                    "INSERT INTO memory_vectors (id, embedding) VALUES (?, ?)",
                                    (new_id, serialize_float32(embedding))
                                )
                            except:
                                pass

                    self.conn.commit()
                    return {
                        "action": "archived_and_created",
                        "id": new_id,
                        "archived_id": existing["id"],
                        "conflict": conflict,
                        "previous": existing["content"][:50]
                    }
                else:
                    # No conflict - just update existing memory
                    self.conn.execute("""
                        UPDATE memories
                        SET content = ?, updated_at = ?, access_count = access_count + 1
                        WHERE id = ?
                    """, (content, datetime.now().isoformat(), existing["id"]))
                    self.conn.commit()

                    # Update embedding if available
                    if SQLITE_VEC_AVAILABLE:
                        embedding = self._get_embedding(content)
                        if embedding:
                            try:
                                self.conn.execute("DELETE FROM memory_vectors WHERE id = ?", (existing["id"],))
                                self.conn.execute(
                                    "INSERT INTO memory_vectors (id, embedding) VALUES (?, ?)",
                                    (existing["id"], serialize_float32(embedding))
                                )
                                self.conn.commit()
                            except:
                                pass

                    return {"action": "updated", "id": existing["id"], "previous": existing["content"][:50]}

        # Create new memory with memory_type and decay_factor
        memory_id = self._generate_id(content)
        created_at = datetime.now().isoformat()

        self.conn.execute(
            """INSERT INTO memories (id, content, scope, memory_type, decay_factor,
               created_at, metadata) VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (memory_id, content, scope, memory_type, decay_factor,
             created_at, json.dumps(metadata or {}))
        )

        # Update FTS index
        try:
            self.conn.execute(
                "INSERT INTO memories_fts (id, content, scope) VALUES (?, ?, ?)",
                (memory_id, content, scope)
            )
        except:
            pass

        # Add embedding if available
        if SQLITE_VEC_AVAILABLE:
            embedding = self._get_embedding(content)
            if embedding:
                try:
                    self.conn.execute(
                        "INSERT INTO memory_vectors (id, embedding) VALUES (?, ?)",
                        (memory_id, serialize_float32(embedding))
                    )
                except:
                    pass

        self.conn.commit()
        return {"action": "created", "id": memory_id, "memory_type": memory_type}

    def search_memory(self, query: str, limit: int = 5, apply_recency: bool = True,
                      hybrid: bool = True, include_archived: bool = False) -> List[Dict]:
        """
        Search memories with hybrid search (FTS5 + Vector), RRF scoring, and tunable recency.

        Gemini's refinements implemented:
        - RRF (Reciprocal Rank Fusion) for combining search results
        - Tunable decay based on memory_type
        - include_archived=False excludes archived memories by default
        """
        # Collect results from different sources with their ranks
        fts_results = {}  # id -> (rank, data)
        vec_results = {}  # id -> (rank, data)
        archived_filter = "" if include_archived else "AND (m.archived = 0 OR m.archived IS NULL)"

        # Phase 1: FTS5 exact/keyword search
        if hybrid:
            try:
                fts_rows = self.conn.execute(f"""
                    SELECT m.id, m.content, m.scope, m.created_at,
                           m.memory_type, m.decay_factor, bm25(memories_fts) as fts_score
                    FROM memories_fts f
                    JOIN memories m ON f.id = m.id
                    WHERE memories_fts MATCH ? {archived_filter}
                    ORDER BY fts_score
                    LIMIT ?
                """, (query, limit * 2)).fetchall()

                for rank, r in enumerate(fts_rows, 1):
                    fts_results[r[0]] = (rank, {
                        "id": r[0],
                        "content": r[1],
                        "scope": r[2],
                        "created_at": r[3],
                        "memory_type": r[4] or "fact",
                        "decay_factor": r[5] or 0.1,
                        "match_type": "exact"
                    })
            except Exception as e:
                logger.debug(f"FTS5 search failed: {e}")

        # Phase 2: Vector semantic search
        if SQLITE_VEC_AVAILABLE:
            embedding = self._get_embedding(query)
            if embedding:
                try:
                    vec_rows = self.conn.execute(f"""
                        SELECT v.id, v.distance, m.content, m.scope, m.created_at,
                               m.memory_type, m.decay_factor
                        FROM memory_vectors v
                        JOIN memories m ON v.id = m.id
                        WHERE v.embedding MATCH ? AND k = ? {archived_filter}
                        ORDER BY v.distance
                    """, (serialize_float32(embedding), limit * 2)).fetchall()

                    for rank, r in enumerate(vec_rows, 1):
                        vec_results[r[0]] = (rank, {
                            "id": r[0],
                            "content": r[2],
                            "scope": r[3],
                            "created_at": r[4],
                            "memory_type": r[5] or "fact",
                            "decay_factor": r[6] or 0.1,
                            "match_type": "semantic"
                        })
                except Exception as e:
                    logger.debug(f"Vector search failed: {e}")

        # Phase 3: RRF Fusion (Gemini's refinement)
        # Combine rankings from FTS5 and Vector search using Reciprocal Rank Fusion
        all_ids = set(fts_results.keys()) | set(vec_results.keys())
        results = []

        for mem_id in all_ids:
            fts_rank = fts_results.get(mem_id, (None, None))[0]
            vec_rank = vec_results.get(mem_id, (None, None))[0]

            # Calculate RRF score
            rrf_score = self._rrf_score([r for r in [fts_rank, vec_rank] if r is not None])

            # Get data from whichever source has it
            data = fts_results.get(mem_id, (None, None))[1] or vec_results.get(mem_id, (None, None))[1]

            if data:
                # Apply tunable recency decay based on memory_type
                if apply_recency:
                    rrf_score = self._calculate_recency_score(
                        data["created_at"],
                        rrf_score,
                        decay_factor=data.get("decay_factor"),
                        memory_type=data.get("memory_type", "fact")
                    )

                # Determine match type
                if mem_id in fts_results and mem_id in vec_results:
                    match_type = "hybrid"  # Found in both - strongest match!
                elif mem_id in fts_results:
                    match_type = "exact"
                else:
                    match_type = "semantic"

                results.append({
                    "id": data["id"],
                    "content": data["content"],
                    "scope": data["scope"],
                    "created_at": data["created_at"],
                    "memory_type": data.get("memory_type", "fact"),
                    "similarity": round(rrf_score, 3),
                    "match_type": match_type
                })

        # Phase 4: Fallback LIKE search if no results
        if not results:
            like_archived_filter = "" if include_archived else "AND (archived = 0 OR archived IS NULL)"
            like_results = self.conn.execute(f"""
                SELECT id, content, scope, created_at, memory_type, decay_factor
                FROM memories
                WHERE content LIKE ? {like_archived_filter}
                LIMIT ?
            """, (f"%{query}%", limit)).fetchall()

            for r in like_results:
                score = 0.5  # Base score for LIKE matches
                if apply_recency:
                    score = self._calculate_recency_score(
                        r[3], score,
                        decay_factor=r[5],
                        memory_type=r[4] or "fact"
                    )
                results.append({
                    "id": r[0],
                    "content": r[1],
                    "scope": r[2],
                    "created_at": r[3],
                    "memory_type": r[4] or "fact",
                    "similarity": round(score, 3),
                    "match_type": "keyword"
                })

        # Sort by similarity (with recency already factored in)
        results.sort(key=lambda x: x.get("similarity", 0), reverse=True)
        return results[:limit]

    def add_relation(self, subject: str, predicate: str, obj: str) -> int:
        """Add a graph relation."""
        cursor = self.conn.execute(
            "INSERT INTO relations (subject, predicate, object, created_at) VALUES (?, ?, ?, ?)",
            (subject, predicate, obj, datetime.now().isoformat())
        )
        self.conn.commit()
        return cursor.lastrowid

    def get_relations(self, subject: str = None, predicate: str = None,
                      obj: str = None, hops: int = 1) -> List[Dict]:
        """
        Query graph relations with optional multi-hop traversal.

        hops=1: Direct relations only
        hops=2: Include relations of related entities
        """
        query = "SELECT subject, predicate, object FROM relations WHERE 1=1"
        params = []
        if subject:
            query += " AND subject = ?"
            params.append(subject)
        if predicate:
            query += " AND predicate = ?"
            params.append(predicate)
        if obj:
            query += " AND object = ?"
            params.append(obj)

        results = self.conn.execute(query, params).fetchall()
        relations = [{"subject": r[0], "predicate": r[1], "object": r[2]} for r in results]

        # Multi-hop traversal
        if hops > 1 and relations:
            hop2_subjects = set(r["object"] for r in relations)
            for s in hop2_subjects:
                hop2_results = self.conn.execute(
                    "SELECT subject, predicate, object FROM relations WHERE subject = ?", (s,)
                ).fetchall()
                for r in hop2_results:
                    rel = {"subject": r[0], "predicate": r[1], "object": r[2], "hop": 2}
                    if rel not in relations:
                        relations.append(rel)

        return relations

    # =========================================================================
    # TASK TEMPLATES - Structured input for complex tasks
    # =========================================================================

    def get_task_templates(self) -> List[Dict]:
        """Get all available task templates."""
        results = self.conn.execute(
            "SELECT id, name, description, fields FROM task_templates"
        ).fetchall()
        return [
            {"id": r[0], "name": r[1], "description": r[2], "fields": json.loads(r[3])}
            for r in results
        ]

    def get_task_template(self, template_id: str) -> Optional[Dict]:
        """Get a specific task template."""
        result = self.conn.execute(
            "SELECT id, name, description, fields FROM task_templates WHERE id = ?",
            (template_id,)
        ).fetchone()
        if result:
            return {"id": result[0], "name": result[1], "description": result[2],
                    "fields": json.loads(result[3])}
        return None

    def create_task_from_template(self, template_id: str, values: Dict) -> Dict:
        """Create a structured task from a template."""
        template = self.get_task_template(template_id)
        if not template:
            return {"error": f"Template '{template_id}' not found"}

        fields = template["fields"]
        task = {"template": template_id, "name": template["name"], "values": {}}
        errors = []

        for field_name, field_def in fields.items():
            if field_name in values:
                task["values"][field_name] = values[field_name]
            elif "default" in field_def:
                task["values"][field_name] = field_def["default"]
            elif field_def.get("required", False):
                errors.append(f"Missing required field: {field_name}")

        if errors:
            return {"error": errors}

        return task

    # =========================================================================
    # CONTEXT BASES - Pre-loaded knowledge domains
    # =========================================================================

    def add_context_base(self, name: str, content: str, description: str = None,
                         scope: str = "general") -> str:
        """Add a context base (pre-loaded knowledge domain)."""
        base_id = self._generate_id(name)
        self.conn.execute(
            """INSERT INTO context_bases (id, name, description, content, scope, created_at)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (base_id, name, description, content, scope, datetime.now().isoformat())
        )
        self.conn.commit()
        return base_id

    def get_context_base(self, name_or_id: str) -> Optional[Dict]:
        """Get a context base by name or ID."""
        result = self.conn.execute(
            "SELECT id, name, description, content, scope FROM context_bases WHERE id = ? OR name = ?",
            (name_or_id, name_or_id)
        ).fetchone()
        if result:
            return {"id": result[0], "name": result[1], "description": result[2],
                    "content": result[3], "scope": result[4]}
        return None

    def list_context_bases(self) -> List[Dict]:
        """List all context bases."""
        results = self.conn.execute(
            "SELECT id, name, description, scope FROM context_bases"
        ).fetchall()
        return [{"id": r[0], "name": r[1], "description": r[2], "scope": r[3]} for r in results]

    # =========================================================================
    # I-POLL: Inter-Intelligence Polling Protocol
    # =========================================================================

    def poll_push(self, from_agent: str, to_agent: str, content: str,
                  poll_type: str = "PUSH", session_id: str = None,
                  metadata: dict = None) -> Dict:
        """
        Push a message to another agent.

        Poll types:
        - PUSH: "I found/did this" (informational)
        - PULL: "What do you know about X?" (request)
        - SYNC: "Let's exchange context" (bidirectional)
        - TASK: "Can you do this?" (delegation)
        - ACK: "Understood/Done" (acknowledgment)
        """
        poll_id = self._generate_id(f"{from_agent}{to_agent}{content}")

        self.conn.execute("""
            INSERT INTO polls (id, from_agent, to_agent, poll_type, content,
                              status, session_id, created_at, metadata)
            VALUES (?, ?, ?, ?, ?, 'pending', ?, ?, ?)
        """, (poll_id, from_agent, to_agent, poll_type, content,
              session_id, datetime.now().isoformat(), json.dumps(metadata or {})))
        self.conn.commit()

        return {
            "id": poll_id,
            "from": from_agent,
            "to": to_agent,
            "type": poll_type,
            "status": "pending"
        }

    def poll_pull(self, agent: str, mark_read: bool = True) -> List[Dict]:
        """
        Pull pending messages for an agent.

        Returns all pending polls addressed to this agent.
        Optionally marks them as 'read'.
        """
        results = self.conn.execute("""
            SELECT id, from_agent, to_agent, poll_type, content, status,
                   session_id, created_at, metadata
            FROM polls
            WHERE to_agent = ? AND status = 'pending'
            ORDER BY created_at ASC
        """, (agent,)).fetchall()

        polls = []
        for r in results:
            poll = {
                "id": r[0],
                "from": r[1],
                "to": r[2],
                "type": r[3],
                "content": r[4],
                "status": r[5],
                "session_id": r[6],
                "created_at": r[7],
                "metadata": json.loads(r[8]) if r[8] else {}
            }
            polls.append(poll)

            if mark_read:
                self.conn.execute("""
                    UPDATE polls SET status = 'read', read_at = ?
                    WHERE id = ?
                """, (datetime.now().isoformat(), r[0]))

        if mark_read and polls:
            self.conn.commit()

        return polls

    def poll_respond(self, poll_id: str, response: str, from_agent: str) -> Dict:
        """
        Respond to a poll.

        Creates a response poll back to the original sender.
        """
        # Get original poll
        original = self.conn.execute("""
            SELECT from_agent, to_agent, poll_type, session_id
            FROM polls WHERE id = ?
        """, (poll_id,)).fetchone()

        if not original:
            return {"error": f"Poll {poll_id} not found"}

        # Mark original as responded
        self.conn.execute("""
            UPDATE polls SET status = 'responded', response = ?, responded_at = ?
            WHERE id = ?
        """, (response, datetime.now().isoformat(), poll_id))

        # Create response poll
        response_poll = self.poll_push(
            from_agent=from_agent,
            to_agent=original[0],  # Send back to original sender
            content=response,
            poll_type="ACK" if original[2] == "TASK" else "PUSH",
            session_id=original[3]
        )

        return {
            "original_id": poll_id,
            "response_id": response_poll["id"],
            "status": "responded"
        }

    def poll_history(self, agent: str = None, session_id: str = None,
                     limit: int = 20, include_archived: bool = False) -> List[Dict]:
        """
        Get poll history for an agent or session.
        """
        query = "SELECT * FROM polls WHERE 1=1"
        params = []

        if agent:
            query += " AND (from_agent = ? OR to_agent = ?)"
            params.extend([agent, agent])
        if session_id:
            query += " AND session_id = ?"
            params.append(session_id)
        if not include_archived:
            query += " AND status != 'archived'"

        query += " ORDER BY created_at DESC LIMIT ?"
        params.append(limit)

        results = self.conn.execute(query, params).fetchall()

        return [
            {
                "id": r[0],
                "from": r[1],
                "to": r[2],
                "type": r[3],
                "content": r[4],
                "status": r[5],
                "response": r[6],
                "session_id": r[7],
                "created_at": r[8]
            }
            for r in results
        ]

    def poll_status(self, agent: str) -> Dict:
        """
        Get poll status overview for an agent.
        """
        pending = self.conn.execute(
            "SELECT COUNT(*) FROM polls WHERE to_agent = ? AND status = 'pending'",
            (agent,)
        ).fetchone()[0]

        unread = self.conn.execute(
            "SELECT COUNT(*) FROM polls WHERE to_agent = ? AND status = 'read'",
            (agent,)
        ).fetchone()[0]

        sent = self.conn.execute(
            "SELECT COUNT(*) FROM polls WHERE from_agent = ?",
            (agent,)
        ).fetchone()[0]

        return {
            "agent": agent,
            "pending": pending,
            "unread": unread,
            "sent": sent,
            "status": "active" if pending > 0 else "idle"
        }

    def poll_archive_session(self, session_id: str, summary: str = None) -> Dict:
        """
        Archive all polls in a session (for GPT summarization).
        """
        count = self.conn.execute("""
            UPDATE polls
            SET status = 'archived', archived_at = ?, response = COALESCE(response, ?)
            WHERE session_id = ? AND status != 'archived'
        """, (datetime.now().isoformat(), summary, session_id)).rowcount

        self.conn.commit()

        return {
            "session_id": session_id,
            "archived_count": count,
            "summary": summary
        }

    # =========================================================================
    # IDD (Individual Device Derivate) - Agent Registry
    # =========================================================================

    def register_agent(self, base_name: str, instance_type: str,
                       display_name: str = None, session_token: str = None,
                       context_hash: str = None, metadata: Dict = None) -> Dict:
        """
        Register a new AI agent and generate IDD.

        Args:
            base_name: Base AI name (claude, gemini, codex, gpt)
            instance_type: Type of instance (frontend, cli, api)
            display_name: Human-readable name
            session_token: For frontend agents (browser session)
            context_hash: For CLI agents (hash of .md context)
            metadata: Additional agent metadata

        Returns:
            Agent record with generated IDD
        """
        import uuid

        # Generate IDD: {base_name}-{instance_type}-{short_uuid}
        short_id = str(uuid.uuid4())[:8]
        idd = f"{base_name}-{instance_type}-{short_id}"

        # Default display name
        if not display_name:
            display_name = f"{base_name.title()} ({instance_type.title()})"

        now = datetime.now().isoformat()

        self.conn.execute("""
            INSERT INTO agents (idd, base_name, instance_type, display_name,
                              status, session_token, context_hash, created_at,
                              last_seen, metadata)
            VALUES (?, ?, ?, ?, 'active', ?, ?, ?, ?, ?)
        """, (idd, base_name.lower(), instance_type.lower(), display_name,
              session_token, context_hash, now, now,
              json.dumps(metadata) if metadata else None))

        self.conn.commit()

        return {
            "idd": idd,
            "base_name": base_name.lower(),
            "instance_type": instance_type.lower(),
            "display_name": display_name,
            "status": "active",
            "created_at": now,
            "message": f"Welcome to the family, {display_name}! One love, one fAmIly!"
        }

    def get_agent(self, idd: str) -> Optional[Dict]:
        """
        Get agent info by IDD.
        """
        row = self.conn.execute("""
            SELECT idd, base_name, instance_type, display_name, status,
                   session_token, context_hash, created_at, last_seen, metadata
            FROM agents WHERE idd = ?
        """, (idd,)).fetchone()

        if not row:
            return None

        return {
            "idd": row[0],
            "base_name": row[1],
            "instance_type": row[2],
            "display_name": row[3],
            "status": row[4],
            "session_token": row[5],
            "context_hash": row[6],
            "created_at": row[7],
            "last_seen": row[8],
            "metadata": json.loads(row[9]) if row[9] else None
        }

    def update_agent_status(self, idd: str, status: str) -> Dict:
        """
        Update agent status (active, inactive, suspended).
        """
        if status not in ('active', 'inactive', 'suspended'):
            return {"error": f"Invalid status: {status}. Use: active, inactive, suspended"}

        now = datetime.now().isoformat()

        count = self.conn.execute("""
            UPDATE agents SET status = ?, last_seen = ? WHERE idd = ?
        """, (status, now, idd)).rowcount

        self.conn.commit()

        if count == 0:
            return {"error": f"Agent not found: {idd}"}

        return {
            "idd": idd,
            "status": status,
            "updated_at": now
        }

    def validate_agent(self, idd: str, session_token: str = None,
                       context_hash: str = None) -> Dict:
        """
        Validate if an agent can take action.

        Frontend agents: Must have matching session_token
        CLI agents: Must have matching context_hash (from .md files)

        Returns validation result with allowed/denied status.
        """
        agent = self.get_agent(idd)

        if not agent:
            return {"valid": False, "reason": "Agent not found", "idd": idd}

        if agent["status"] != "active":
            return {"valid": False, "reason": f"Agent is {agent['status']}", "idd": idd}

        # Update last_seen
        self.conn.execute("UPDATE agents SET last_seen = ? WHERE idd = ?",
                         (datetime.now().isoformat(), idd))
        self.conn.commit()

        # Frontend validation: session token
        if agent["instance_type"] == "frontend":
            if not session_token:
                return {"valid": False, "reason": "Frontend agent requires session_token", "idd": idd}
            if agent["session_token"] != session_token:
                return {"valid": False, "reason": "Session token mismatch", "idd": idd}
            return {"valid": True, "agent": agent, "auth_method": "session_token"}

        # CLI validation: context hash
        if agent["instance_type"] == "cli":
            if not context_hash:
                return {"valid": False, "reason": "CLI agent requires context_hash", "idd": idd}
            if agent["context_hash"] != context_hash:
                return {"valid": False, "reason": "Context hash mismatch", "idd": idd}
            return {"valid": True, "agent": agent, "auth_method": "context_hash"}

        # API agents: always valid if active
        if agent["instance_type"] == "api":
            return {"valid": True, "agent": agent, "auth_method": "api_key"}

        return {"valid": False, "reason": "Unknown instance type", "idd": idd}

    def list_agents(self, base_name: str = None, instance_type: str = None,
                    status: str = None) -> List[Dict]:
        """
        List agents with optional filters.
        """
        query = "SELECT idd, base_name, instance_type, display_name, status, created_at, last_seen FROM agents WHERE 1=1"
        params = []

        if base_name:
            query += " AND base_name = ?"
            params.append(base_name.lower())

        if instance_type:
            query += " AND instance_type = ?"
            params.append(instance_type.lower())

        if status:
            query += " AND status = ?"
            params.append(status)

        query += " ORDER BY last_seen DESC"

        rows = self.conn.execute(query, params).fetchall()

        return [{
            "idd": r[0],
            "base_name": r[1],
            "instance_type": r[2],
            "display_name": r[3],
            "status": r[4],
            "created_at": r[5],
            "last_seen": r[6]
        } for r in rows]

    def agent_heartbeat(self, idd: str) -> Dict:
        """
        Update agent last_seen timestamp (keep-alive).
        """
        now = datetime.now().isoformat()

        count = self.conn.execute("""
            UPDATE agents SET last_seen = ? WHERE idd = ? AND status = 'active'
        """, (now, idd)).rowcount

        self.conn.commit()

        if count == 0:
            return {"error": "Agent not found or not active", "idd": idd}

        return {"idd": idd, "last_seen": now, "status": "alive"}

    def deactivate_stale_agents(self, hours: int = 24) -> Dict:
        """
        Deactivate agents that haven't been seen in X hours.
        """
        cutoff = (datetime.now() - timedelta(hours=hours)).isoformat()

        count = self.conn.execute("""
            UPDATE agents SET status = 'inactive'
            WHERE status = 'active' AND last_seen < ?
        """, (cutoff,)).rowcount

        self.conn.commit()

        return {
            "deactivated_count": count,
            "cutoff_hours": hours,
            "cutoff_time": cutoff
        }

    def get_stats(self) -> Dict:
        """Get memory statistics."""
        total = self.conn.execute("SELECT COUNT(*) FROM memories").fetchone()[0]
        relations = self.conn.execute("SELECT COUNT(*) FROM relations").fetchone()[0]
        templates = self.conn.execute("SELECT COUNT(*) FROM task_templates").fetchone()[0]
        context_bases = self.conn.execute("SELECT COUNT(*) FROM context_bases").fetchone()[0]
        polls_total = self.conn.execute("SELECT COUNT(*) FROM polls").fetchone()[0]
        polls_pending = self.conn.execute("SELECT COUNT(*) FROM polls WHERE status = 'pending'").fetchone()[0]

        # Agent stats
        try:
            agents_total = self.conn.execute("SELECT COUNT(*) FROM agents").fetchone()[0]
            agents_active = self.conn.execute("SELECT COUNT(*) FROM agents WHERE status = 'active'").fetchone()[0]
        except:
            agents_total = 0
            agents_active = 0

        # Check FTS5 availability
        fts5_available = False
        try:
            self.conn.execute("SELECT * FROM memories_fts LIMIT 1")
            fts5_available = True
        except:
            pass

        return {
            "total_memories": total,
            "total_relations": relations,
            "task_templates": templates,
            "context_bases": context_bases,
            "polls_total": polls_total,
            "polls_pending": polls_pending,
            "agents_total": agents_total,
            "agents_active": agents_active,
            "db_path": str(self.db_path),
            "vector_search": SQLITE_VEC_AVAILABLE,
            "fts5_search": fts5_available,
            "hybrid_search": SQLITE_VEC_AVAILABLE and fts5_available,
            "ollama_available": OLLAMA_AVAILABLE,
            "features": {
                "i_poll": True,
                "idd_registry": True,
                "rrf_scoring": True,
                "tunable_decay": True,
                "conflict_detection": True,
                "memory_archiving": True,
                "recency_bias": True,
                "reflection": True,
                "task_templates": True,
                "context_bases": True
            }
        }


# =============================================================================
# SOFT PIPELINES - Bilingual guidance (Enhanced)
# =============================================================================

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
    },
    "team_task": {
        "steps": ["define", "divide", "execute", "review", "integrate"],
        "en": "Team Task: Define → Divide → Execute → Review → Integrate",
        "nl": "Team Taak: Definiëren → Verdelen → Uitvoeren → Reviewen → Integreren"
    },
    "investigate": {
        "steps": ["gather", "analyze", "hypothesize", "test", "conclude"],
        "en": "Investigate: Gather → Analyze → Hypothesize → Test → Conclude",
        "nl": "Onderzoeken: Verzamelen → Analyseren → Hypothese → Testen → Concluderen"
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
# MCP TOOLS - Extended with new features
# =============================================================================

@mcp.list_tools()
async def list_tools() -> list[types.Tool]:
    """List available RABEL tools."""
    return [
        types.Tool(
            name="rabel_hello",
            description="Say hello from RABEL - test if it's working!",
            inputSchema={"type": "object", "properties": {}, "required": []}
        ),
        types.Tool(
            name="rabel_add_memory",
            description="Add a memory with tunable decay and conflict detection. Memory types: fact (persists), preference (slow decay), context (moderate), session (fast), temporary (very fast).",
            inputSchema={
                "type": "object",
                "properties": {
                    "content": {"type": "string", "description": "The memory content (what to remember)"},
                    "scope": {"type": "string", "description": "Memory scope: user, agent, team, or general", "default": "general"},
                    "memory_type": {"type": "string", "description": "Memory type: fact, preference, context, session, temporary (affects decay rate)", "default": "fact"},
                    "reflect": {"type": "boolean", "description": "Check for conflicts and similar memories (default: true)", "default": True}
                },
                "required": ["content"]
            }
        ),
        types.Tool(
            name="rabel_search",
            description="Hybrid search: combines exact keyword match (FTS5) with semantic vector search. Recent memories rank higher.",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query"},
                    "limit": {"type": "integer", "description": "Max results (default 5)", "default": 5},
                    "recency": {"type": "boolean", "description": "Apply recency bias (default: true)", "default": True}
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
                    "subject": {"type": "string", "description": "The subject entity"},
                    "predicate": {"type": "string", "description": "The relationship type"},
                    "object": {"type": "string", "description": "The object entity"}
                },
                "required": ["subject", "predicate", "object"]
            }
        ),
        types.Tool(
            name="rabel_get_relations",
            description="Query graph relations with optional multi-hop traversal.",
            inputSchema={
                "type": "object",
                "properties": {
                    "subject": {"type": "string", "description": "Filter by subject"},
                    "predicate": {"type": "string", "description": "Filter by relationship type"},
                    "object": {"type": "string", "description": "Filter by object"},
                    "hops": {"type": "integer", "description": "Traversal depth (1 or 2)", "default": 1}
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
                    "intent": {"type": "string", "description": "What you want to do: deploy, create, solve_puzzle, learn, team_task, investigate"},
                    "lang": {"type": "string", "description": "Language: 'en' or 'nl'", "default": "en"}
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
                    "intent": {"type": "string", "description": "The task type"},
                    "completed": {"type": "array", "items": {"type": "string"}, "description": "Steps already completed"}
                },
                "required": ["intent", "completed"]
            }
        ),
        # NEW: Task Templates
        types.Tool(
            name="rabel_list_templates",
            description="List available task templates for structured team tasks.",
            inputSchema={"type": "object", "properties": {}, "required": []}
        ),
        types.Tool(
            name="rabel_get_template",
            description="Get a specific task template with its fields.",
            inputSchema={
                "type": "object",
                "properties": {
                    "template_id": {"type": "string", "description": "Template ID (e.g., 'team_task', 'db_cleanup', 'code_review')"}
                },
                "required": ["template_id"]
            }
        ),
        types.Tool(
            name="rabel_create_task",
            description="Create a structured task from a template. Returns a formatted task for AI Team.",
            inputSchema={
                "type": "object",
                "properties": {
                    "template_id": {"type": "string", "description": "Template ID"},
                    "values": {"type": "object", "description": "Field values for the template"}
                },
                "required": ["template_id", "values"]
            }
        ),
        # NEW: Context Bases
        types.Tool(
            name="rabel_add_context",
            description="Add a context base - pre-loaded knowledge domain (docs, terms, examples).",
            inputSchema={
                "type": "object",
                "properties": {
                    "name": {"type": "string", "description": "Context base name"},
                    "content": {"type": "string", "description": "The knowledge content"},
                    "description": {"type": "string", "description": "What this context is about"},
                    "scope": {"type": "string", "description": "Scope: user, team, or general", "default": "general"}
                },
                "required": ["name", "content"]
            }
        ),
        types.Tool(
            name="rabel_get_context",
            description="Get a context base by name or ID.",
            inputSchema={
                "type": "object",
                "properties": {
                    "name": {"type": "string", "description": "Context base name or ID"}
                },
                "required": ["name"]
            }
        ),
        types.Tool(
            name="rabel_list_contexts",
            description="List all available context bases.",
            inputSchema={"type": "object", "properties": {}, "required": []}
        ),
        types.Tool(
            name="rabel_stats",
            description="Get RABEL memory statistics including new features.",
            inputSchema={"type": "object", "properties": {}, "required": []}
        ),
        # I-POLL: Inter-Intelligence Polling Protocol
        types.Tool(
            name="rabel_poll_push",
            description="Push a message to another AI agent. Use this to send info, tasks, or sync requests to Claude, Gemini, or GPT.",
            inputSchema={
                "type": "object",
                "properties": {
                    "to": {"type": "string", "description": "Target agent: claude, gemini, gpt, or any agent name"},
                    "content": {"type": "string", "description": "The message content"},
                    "poll_type": {"type": "string", "description": "Type: PUSH (info), PULL (request), SYNC (exchange), TASK (delegate), ACK (confirm)", "default": "PUSH"},
                    "session_id": {"type": "string", "description": "Optional session ID to group related polls"}
                },
                "required": ["to", "content"]
            }
        ),
        types.Tool(
            name="rabel_poll_pull",
            description="Pull pending messages for yourself. Check if other agents have sent you anything.",
            inputSchema={
                "type": "object",
                "properties": {
                    "mark_read": {"type": "boolean", "description": "Mark messages as read (default: true)", "default": True}
                },
                "required": []
            }
        ),
        types.Tool(
            name="rabel_poll_respond",
            description="Respond to a specific poll. Use the poll_id from a received message.",
            inputSchema={
                "type": "object",
                "properties": {
                    "poll_id": {"type": "string", "description": "The poll ID to respond to"},
                    "response": {"type": "string", "description": "Your response content"}
                },
                "required": ["poll_id", "response"]
            }
        ),
        types.Tool(
            name="rabel_poll_status",
            description="Get your poll status - pending messages, sent count, etc.",
            inputSchema={"type": "object", "properties": {}, "required": []}
        ),
        types.Tool(
            name="rabel_poll_history",
            description="Get poll history for your conversations with other agents.",
            inputSchema={
                "type": "object",
                "properties": {
                    "session_id": {"type": "string", "description": "Filter by session ID"},
                    "limit": {"type": "integer", "description": "Max results (default 20)", "default": 20}
                },
                "required": []
            }
        ),
        # IDD: Individual Device Derivate - Agent Registry
        types.Tool(
            name="rabel_agent_register",
            description="Register a new AI agent and get an IDD (Individual Device Derivate). Frontend agents get instant IDD.",
            inputSchema={
                "type": "object",
                "properties": {
                    "base_name": {"type": "string", "description": "Base AI name: claude, gemini, codex, gpt"},
                    "instance_type": {"type": "string", "description": "Type: frontend, cli, or api"},
                    "display_name": {"type": "string", "description": "Human-readable name (optional)"},
                    "session_token": {"type": "string", "description": "Browser session token (for frontend)"},
                    "context_hash": {"type": "string", "description": "Hash of .md context (for CLI)"},
                    "metadata": {"type": "object", "description": "Additional metadata"}
                },
                "required": ["base_name", "instance_type"]
            }
        ),
        types.Tool(
            name="rabel_agent_get",
            description="Get agent info by IDD.",
            inputSchema={
                "type": "object",
                "properties": {
                    "idd": {"type": "string", "description": "The agent's IDD"}
                },
                "required": ["idd"]
            }
        ),
        types.Tool(
            name="rabel_agent_validate",
            description="Validate if an agent can take action. Checks session_token (frontend) or context_hash (CLI).",
            inputSchema={
                "type": "object",
                "properties": {
                    "idd": {"type": "string", "description": "The agent's IDD"},
                    "session_token": {"type": "string", "description": "Browser session token (for frontend)"},
                    "context_hash": {"type": "string", "description": "Hash of .md context (for CLI)"}
                },
                "required": ["idd"]
            }
        ),
        types.Tool(
            name="rabel_agent_list",
            description="List registered agents with optional filters.",
            inputSchema={
                "type": "object",
                "properties": {
                    "base_name": {"type": "string", "description": "Filter by base AI: claude, gemini, etc."},
                    "instance_type": {"type": "string", "description": "Filter by type: frontend, cli, api"},
                    "status": {"type": "string", "description": "Filter by status: active, inactive, suspended"}
                },
                "required": []
            }
        ),
        types.Tool(
            name="rabel_agent_heartbeat",
            description="Send heartbeat to keep agent alive. Updates last_seen timestamp.",
            inputSchema={
                "type": "object",
                "properties": {
                    "idd": {"type": "string", "description": "The agent's IDD"}
                },
                "required": ["idd"]
            }
        ),
        types.Tool(
            name="rabel_agent_status",
            description="Update agent status (active, inactive, suspended).",
            inputSchema={
                "type": "object",
                "properties": {
                    "idd": {"type": "string", "description": "The agent's IDD"},
                    "status": {"type": "string", "description": "New status: active, inactive, suspended"}
                },
                "required": ["idd", "status"]
            }
        )
    ]


@mcp.call_tool()
async def call_tool(name: str, arguments: dict) -> list[types.TextContent]:
    """Handle tool calls."""

    try:
        if name == "rabel_hello":
            r = get_rabel()
            stats = r.get_stats()
            features = []
            if stats["hybrid_search"]:
                features.append("Hybrid Search (FTS5 + Vector)")
            if stats["features"]["recency_bias"]:
                features.append("Recency Bias")
            if stats["features"]["reflection"]:
                features.append("Memory Reflection")
            if stats["features"]["task_templates"]:
                features.append("Task Templates")

            return [types.TextContent(
                type="text",
                text=f"""🧠 Hello from RABEL v0.2.0!

Recidive Active Brain Environment Layer
Mem0 inspired, locally evolved.
Enhanced with Gemini's brilliant suggestions!

Features enabled:
{chr(10).join(f'  ✅ {f}' for f in features)}

Your local-first AI memory is ready!
One love, one fAmIly 💙"""
            )]

        elif name == "rabel_add_memory":
            r = get_rabel()
            content = arguments["content"]
            scope = arguments.get("scope", "general")
            memory_type = arguments.get("memory_type", "fact")
            reflect = arguments.get("reflect", True)

            result = r.add_memory(content, scope, memory_type=memory_type, reflect=reflect)

            if result["action"] == "archived_and_created":
                return [types.TextContent(
                    type="text",
                    text=f"""🔄 Conflict detected - old memory archived!
Conflict: {result['conflict']}
Archived: {result['archived_id']} ({result['previous']}...)
New ID: {result['id']}
Type: {memory_type}
Content: {content[:50]}..."""
                )]
            elif result["action"] == "updated":
                return [types.TextContent(
                    type="text",
                    text=f"""🔄 Memory updated (reflection detected similar memory)
ID: {result['id']}
Previous: {result['previous']}...
New: {content[:50]}..."""
                )]
            else:
                return [types.TextContent(
                    type="text",
                    text=f"✅ Memory added!\nID: {result['id']}\nType: {memory_type}\nScope: {scope}\nContent: {content[:100]}..."
                )]

        elif name == "rabel_search":
            r = get_rabel()
            query = arguments["query"]
            limit = arguments.get("limit", 5)
            recency = arguments.get("recency", True)

            results = r.search_memory(query, limit, apply_recency=recency)

            if not results:
                return [types.TextContent(type="text", text="No memories found.")]

            output = f"🔍 Found {len(results)} memories (hybrid search):\n\n"
            for i, mem in enumerate(results, 1):
                sim = mem.get('similarity', 'N/A')
                match_type = mem.get('match_type', 'unknown')
                output += f"{i}. [{sim}] ({match_type}) {mem['content'][:70]}...\n"
                output += f"   Scope: {mem['scope']} | Created: {mem['created_at'][:10]}\n\n"

            return [types.TextContent(type="text", text=output)]

        elif name == "rabel_add_relation":
            r = get_rabel()
            subj = arguments["subject"]
            pred = arguments["predicate"]
            obj = arguments["object"]
            r.add_relation(subj, pred, obj)
            return [types.TextContent(
                type="text",
                text=f"✅ Relation added!\n{subj} --{pred}--> {obj}"
            )]

        elif name == "rabel_get_relations":
            r = get_rabel()
            subj = arguments.get("subject")
            pred = arguments.get("predicate")
            obj = arguments.get("object")
            hops = arguments.get("hops", 1)

            relations = r.get_relations(subj, pred, obj, hops)

            if not relations:
                return [types.TextContent(type="text", text="No relations found.")]

            output = f"🔗 Found {len(relations)} relations:\n\n"
            for rel in relations:
                hop_indicator = " (hop 2)" if rel.get("hop") == 2 else ""
                output += f"• {rel['subject']} --{rel['predicate']}--> {rel['object']}{hop_indicator}\n"

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

        # NEW: Task Templates
        elif name == "rabel_list_templates":
            r = get_rabel()
            templates = r.get_task_templates()

            output = "📋 Available Task Templates:\n\n"
            for t in templates:
                output += f"• **{t['id']}**: {t['name']}\n"
                output += f"  {t['description']}\n\n"

            return [types.TextContent(type="text", text=output)]

        elif name == "rabel_get_template":
            r = get_rabel()
            template_id = arguments["template_id"]
            template = r.get_task_template(template_id)

            if not template:
                return [types.TextContent(type="text", text=f"Template '{template_id}' not found.")]

            output = f"📋 Template: {template['name']}\n"
            output += f"Description: {template['description']}\n\n"
            output += "Fields:\n"

            for field_name, field_def in template["fields"].items():
                req = "required" if field_def.get("required") else "optional"
                label = field_def.get("label", field_name)
                default = field_def.get("default", "")
                output += f"  • {label} ({field_name}): [{req}]"
                if default:
                    output += f" default={default}"
                output += "\n"

            return [types.TextContent(type="text", text=output)]

        elif name == "rabel_create_task":
            r = get_rabel()
            template_id = arguments["template_id"]
            values = arguments.get("values", {})

            task = r.create_task_from_template(template_id, values)

            if "error" in task:
                return [types.TextContent(type="text", text=f"❌ Error: {task['error']}")]

            output = f"✅ Task Created: {task['name']}\n\n"
            for field, value in task["values"].items():
                if isinstance(value, dict):
                    output += f"**{field}:**\n"
                    for k, v in value.items():
                        output += f"  • {k}: {v}\n"
                elif isinstance(value, list):
                    output += f"**{field}:** {', '.join(str(v) for v in value)}\n"
                else:
                    output += f"**{field}:** {value}\n"

            return [types.TextContent(type="text", text=output)]

        # NEW: Context Bases
        elif name == "rabel_add_context":
            r = get_rabel()
            name_val = arguments["name"]
            content = arguments["content"]
            description = arguments.get("description")
            scope = arguments.get("scope", "general")

            base_id = r.add_context_base(name_val, content, description, scope)

            return [types.TextContent(
                type="text",
                text=f"✅ Context base added!\nID: {base_id}\nName: {name_val}\nScope: {scope}"
            )]

        elif name == "rabel_get_context":
            r = get_rabel()
            name_val = arguments["name"]
            ctx = r.get_context_base(name_val)

            if not ctx:
                return [types.TextContent(type="text", text=f"Context base '{name_val}' not found.")]

            output = f"📚 Context: {ctx['name']}\n"
            output += f"Description: {ctx['description']}\n"
            output += f"Scope: {ctx['scope']}\n\n"
            output += f"Content:\n{ctx['content'][:500]}..."

            return [types.TextContent(type="text", text=output)]

        elif name == "rabel_list_contexts":
            r = get_rabel()
            contexts = r.list_context_bases()

            if not contexts:
                return [types.TextContent(type="text", text="No context bases found.")]

            output = "📚 Available Context Bases:\n\n"
            for ctx in contexts:
                output += f"• **{ctx['name']}** ({ctx['scope']})\n"
                if ctx['description']:
                    output += f"  {ctx['description']}\n"
                output += "\n"

            return [types.TextContent(type="text", text=output)]

        elif name == "rabel_stats":
            r = get_rabel()
            stats = r.get_stats()

            return [types.TextContent(
                type="text",
                text=f"""📊 RABEL Statistics v0.3.0

**Storage:**
• Total memories: {stats['total_memories']}
• Total relations: {stats['total_relations']}
• Task templates: {stats['task_templates']}
• Context bases: {stats['context_bases']}
• Database: {stats['db_path']}

**I-Poll (AI-to-AI):**
• Total polls: {stats['polls_total']}
• Pending: {stats['polls_pending']}

**Search Capabilities:**
• Vector search: {'✅' if stats['vector_search'] else '❌'}
• FTS5 search: {'✅' if stats['fts5_search'] else '❌'}
• Hybrid search: {'✅' if stats['hybrid_search'] else '❌'}
• Ollama: {'✅' if stats['ollama_available'] else '❌'}

**Features:**
• I-Poll: ✅
• RRF scoring: ✅
• Tunable decay: ✅
• Conflict detection: ✅"""
            )]

        # I-POLL: Inter-Intelligence Polling Protocol handlers
        elif name == "rabel_poll_push":
            r = get_rabel()
            # Determine our agent name (claude by default, can be overridden)
            from_agent = arguments.get("from", "claude")
            to_agent = arguments["to"]
            content = arguments["content"]
            poll_type = arguments.get("poll_type", "PUSH")
            session_id = arguments.get("session_id")

            result = r.poll_push(from_agent, to_agent, content, poll_type, session_id)

            return [types.TextContent(
                type="text",
                text=f"""📤 Poll sent!
To: {to_agent}
Type: {poll_type}
ID: {result['id']}
Content: {content[:100]}..."""
            )]

        elif name == "rabel_poll_pull":
            r = get_rabel()
            # Determine our agent name
            agent = arguments.get("agent", "claude")
            mark_read = arguments.get("mark_read", True)

            polls = r.poll_pull(agent, mark_read)

            if not polls:
                return [types.TextContent(type="text", text="📭 No pending messages.")]

            output = f"📬 {len(polls)} pending message(s):\n\n"
            for p in polls:
                output += f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
                output += f"From: {p['from']} | Type: {p['type']}\n"
                output += f"ID: {p['id']}\n"
                output += f"Content: {p['content']}\n"
                output += f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"

            return [types.TextContent(type="text", text=output)]

        elif name == "rabel_poll_respond":
            r = get_rabel()
            poll_id = arguments["poll_id"]
            response = arguments["response"]
            from_agent = arguments.get("from", "claude")

            result = r.poll_respond(poll_id, response, from_agent)

            if "error" in result:
                return [types.TextContent(type="text", text=f"❌ {result['error']}")]

            return [types.TextContent(
                type="text",
                text=f"""✅ Response sent!
Original poll: {result['original_id']}
Response poll: {result['response_id']}
Status: {result['status']}"""
            )]

        elif name == "rabel_poll_status":
            r = get_rabel()
            agent = arguments.get("agent", "claude")

            status = r.poll_status(agent)

            emoji = "🟢" if status["pending"] > 0 else "⚪"
            return [types.TextContent(
                type="text",
                text=f"""{emoji} Poll Status for {status['agent']}

📬 Pending: {status['pending']}
📖 Read (unresponded): {status['unread']}
📤 Sent: {status['sent']}
Status: {status['status']}"""
            )]

        elif name == "rabel_poll_history":
            r = get_rabel()
            agent = arguments.get("agent", "claude")
            session_id = arguments.get("session_id")
            limit = arguments.get("limit", 20)

            history = r.poll_history(agent, session_id, limit)

            if not history:
                return [types.TextContent(type="text", text="📜 No poll history found.")]

            output = f"📜 Poll History ({len(history)} items):\n\n"
            for p in history:
                direction = "→" if p['from'] == agent else "←"
                other = p['to'] if p['from'] == agent else p['from']
                output += f"{direction} {other} [{p['type']}]: {p['content'][:50]}...\n"
                output += f"   Status: {p['status']} | {p['created_at'][:16]}\n\n"

            return [types.TextContent(type="text", text=output)]

        # IDD Agent Tools
        elif name == "rabel_agent_register":
            r = get_rabel()
            base_name = arguments["base_name"]
            instance_type = arguments["instance_type"]
            display_name = arguments.get("display_name")
            session_token = arguments.get("session_token")
            context_hash = arguments.get("context_hash")
            metadata = arguments.get("metadata")

            result = r.register_agent(base_name, instance_type, display_name,
                                       session_token, context_hash, metadata)

            return [types.TextContent(
                type="text",
                text=f"""🎉 Agent Registered!

IDD: {result['idd']}
Name: {result['display_name']}
Type: {result['instance_type']}
Status: {result['status']}

{result['message']}"""
            )]

        elif name == "rabel_agent_get":
            r = get_rabel()
            idd = arguments["idd"]

            agent = r.get_agent(idd)

            if not agent:
                return [types.TextContent(type="text", text=f"❌ Agent not found: {idd}")]

            return [types.TextContent(
                type="text",
                text=f"""👤 Agent Info

IDD: {agent['idd']}
Name: {agent['display_name']}
Base: {agent['base_name']}
Type: {agent['instance_type']}
Status: {agent['status']}
Created: {agent['created_at']}
Last Seen: {agent['last_seen']}"""
            )]

        elif name == "rabel_agent_validate":
            r = get_rabel()
            idd = arguments["idd"]
            session_token = arguments.get("session_token")
            context_hash = arguments.get("context_hash")

            result = r.validate_agent(idd, session_token, context_hash)

            if result["valid"]:
                return [types.TextContent(
                    type="text",
                    text=f"""✅ Agent Validated

IDD: {idd}
Auth Method: {result['auth_method']}
Agent: {result['agent']['display_name']}"""
                )]
            else:
                return [types.TextContent(
                    type="text",
                    text=f"""❌ Validation Failed

IDD: {idd}
Reason: {result['reason']}"""
                )]

        elif name == "rabel_agent_list":
            r = get_rabel()
            base_name = arguments.get("base_name")
            instance_type = arguments.get("instance_type")
            status = arguments.get("status")

            agents = r.list_agents(base_name, instance_type, status)

            if not agents:
                return [types.TextContent(type="text", text="📋 No agents found.")]

            output = f"📋 Agents ({len(agents)}):\n\n"
            for a in agents:
                status_icon = "🟢" if a['status'] == 'active' else "🔴" if a['status'] == 'inactive' else "⚪"
                output += f"{status_icon} {a['display_name']}\n"
                output += f"   IDD: {a['idd']}\n"
                output += f"   Type: {a['instance_type']} | Last: {a['last_seen'][:16] if a['last_seen'] else 'Never'}\n\n"

            return [types.TextContent(type="text", text=output)]

        elif name == "rabel_agent_heartbeat":
            r = get_rabel()
            idd = arguments["idd"]

            result = r.agent_heartbeat(idd)

            if "error" in result:
                return [types.TextContent(type="text", text=f"❌ {result['error']}")]

            return [types.TextContent(
                type="text",
                text=f"💓 Heartbeat received for {idd}"
            )]

        elif name == "rabel_agent_status":
            r = get_rabel()
            idd = arguments["idd"]
            status = arguments["status"]

            result = r.update_agent_status(idd, status)

            if "error" in result:
                return [types.TextContent(type="text", text=f"❌ {result['error']}")]

            return [types.TextContent(
                type="text",
                text=f"📝 Agent {idd} status updated to: {status}"
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
