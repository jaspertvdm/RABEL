"""
RABEL MCP Server - The Brain Layer v0.2.1
==========================================

Exposes RABEL memory capabilities via MCP protocol.

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

    def get_stats(self) -> Dict:
        """Get memory statistics."""
        total = self.conn.execute("SELECT COUNT(*) FROM memories").fetchone()[0]
        relations = self.conn.execute("SELECT COUNT(*) FROM relations").fetchone()[0]
        templates = self.conn.execute("SELECT COUNT(*) FROM task_templates").fetchone()[0]
        context_bases = self.conn.execute("SELECT COUNT(*) FROM context_bases").fetchone()[0]

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
            "db_path": str(self.db_path),
            "vector_search": SQLITE_VEC_AVAILABLE,
            "fts5_search": fts5_available,
            "hybrid_search": SQLITE_VEC_AVAILABLE and fts5_available,
            "ollama_available": OLLAMA_AVAILABLE,
            "features": {
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
                text=f"""📊 RABEL Statistics v0.2.0

**Storage:**
• Total memories: {stats['total_memories']}
• Total relations: {stats['total_relations']}
• Task templates: {stats['task_templates']}
• Context bases: {stats['context_bases']}
• Database: {stats['db_path']}

**Search Capabilities:**
• Vector search: {'✅' if stats['vector_search'] else '❌'}
• FTS5 search: {'✅' if stats['fts5_search'] else '❌'}
• Hybrid search: {'✅' if stats['hybrid_search'] else '❌'}
• Ollama: {'✅' if stats['ollama_available'] else '❌'}

**Features:**
• Recency bias: ✅
• Memory reflection: ✅
• Task templates: ✅
• Context bases: ✅"""
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
