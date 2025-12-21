"""
RABEL MCP Server - Recidive Active Brain Environment Layer
===========================================================

Local-first AI memory for everyone. Mem0 inspired, HumoticaOS evolved.

Install:
    pip install mcp-server-rabel

Add to Claude CLI:
    claude mcp add rabel -- python -m mcp_server_rabel

Features:
- Semantic memory search (SQLite + sqlite-vec + Ollama)
- Graph relations between entities
- Soft pipelines with bilingual guidance (EN/NL)
- Procedure validation ("Handschoen voor muts? FOUT!")
- 100% local - zero cloud dependencies

Inspired by: Mem0 (https://mem0.ai)
Built by: Jasper & Root AI @ HumoticaOS
License: MIT - One love, one fAmIly 💙
"""

from .server import main, mcp

__version__ = "0.1.0"
__all__ = ["main", "mcp"]
