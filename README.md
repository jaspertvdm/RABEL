# 🧠 RABEL MCP Server

[![PyPI version](https://badge.fury.io/py/mcp-server-rabel.svg)](https://pypi.org/project/mcp-server-rabel/)
[![Downloads](https://pepy.tech/badge/mcp-server-rabel)](https://pepy.tech/project/mcp-server-rabel)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**Recidive Active Brain Environment Layer**

> Local-first AI memory with semantic search, graph relations, and soft pipelines.
> Mem0 inspired, HumoticaOS evolved.

By **Jasper & Root AI** from [HumoticaOS](https://humotica.com) 💙

---

## 🚀 Quick Start

```bash
# Install
pip install mcp-server-rabel

# For full features (vector search)
pip install mcp-server-rabel[full]

# Add to Claude CLI
claude mcp add rabel -- python -m mcp_server_rabel

# Verify
claude mcp list
# rabel: ✓ Connected
```

---

## 🤔 What is RABEL?

RABEL gives AI assistants **persistent memory** that works **100% locally**.

```
Before RABEL:
  AI: "Who is Storm?" → "I don't know, you haven't told me"

After RABEL:
  You: "Remember: Storm is Jasper's 7-year-old son"
  AI: *saves to RABEL*

  Later...
  You: "Who is Storm?"
  AI: *searches RABEL* → "Storm is Jasper's 7-year-old son!"
```

**No cloud. No API keys. No data leaving your machine.**

---

## 🛠️ Available Tools

| Tool | Description |
|------|-------------|
| `rabel_hello` | Test if RABEL is working |
| `rabel_add_memory` | Add a memory (fact, experience, knowledge) |
| `rabel_search` | Semantic search through memories |
| `rabel_add_relation` | Add graph relation (A --rel--> B) |
| `rabel_get_relations` | Query the knowledge graph |
| `rabel_get_guidance` | Get soft pipeline hints (EN/NL) |
| `rabel_next_step` | What should I do next? |
| `rabel_stats` | Memory statistics |

---

## 📖 Examples

### Adding Memories

```python
# Remember facts
rabel_add_memory(content="Jasper is the founder of HumoticaOS", scope="user")
rabel_add_memory(content="TIBET handles trust and provenance", scope="team")
rabel_add_memory(content="Always validate input before processing", scope="agent")
```

### Searching Memories

```python
# Semantic search - ask questions naturally
rabel_search(query="Who founded HumoticaOS?")
# → Returns: "Jasper is the founder of HumoticaOS"

rabel_search(query="What handles trust?")
# → Returns: "TIBET handles trust and provenance"
```

### Knowledge Graph

```python
# Add relations
rabel_add_relation(subject="Jasper", predicate="father_of", object="Storm")
rabel_add_relation(subject="TIBET", predicate="part_of", object="HumoticaOS")
rabel_add_relation(subject="RABEL", predicate="part_of", object="HumoticaOS")

# Query relations
rabel_get_relations(subject="Jasper")
# → Jasper --father_of--> Storm

rabel_get_relations(predicate="part_of")
# → TIBET --part_of--> HumoticaOS
# → RABEL --part_of--> HumoticaOS
```

### Soft Pipelines (Bilingual!)

```python
# Get guidance in English
rabel_get_guidance(intent="solve_puzzle", lang="en")
# → "Puzzle: Read → Analyze → Attempt → Verify → Document"

# Get guidance in Dutch
rabel_get_guidance(intent="solve_puzzle", lang="nl")
# → "Puzzel: Lezen → Analyseren → Proberen → Verifiëren → Documenteren"

# What's next?
rabel_next_step(intent="solve_puzzle", completed=["read", "analyze"])
# → Suggested next step: "attempt"
```

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                         RABEL                                │
│       Recidive Active Brain Environment Layer               │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│   Memory Layer     → Semantic facts with embeddings         │
│   Graph Layer      → Relations between entities             │
│   Soft Pipelines   → Guidance without enforcement (EN/NL)   │
│                                                             │
│   Storage: SQLite + sqlite-vec (optional)                   │
│   Embeddings: Ollama nomic-embed-text (optional)            │
│                                                             │
│   100% LOCAL - Zero cloud dependencies                      │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### Graceful Degradation

RABEL works with minimal dependencies:

| Feature | Without extras | With `[full]` |
|---------|---------------|---------------|
| Text memories | ✅ | ✅ |
| Text search | ✅ (LIKE query) | ✅ (semantic) |
| Graph relations | ✅ | ✅ |
| Soft pipelines | ✅ | ✅ |
| Vector search | ❌ | ✅ |
| Embeddings | ❌ | ✅ (Ollama) |

---

## 🌍 Philosophy

> "LOKAAL EERST - het systeem MOET werken zonder internet"
>
> (LOCAL FIRST - the system MUST work without internet)

RABEL is built on the belief that:

- **Your data stays yours** - No cloud, no tracking, no API keys
- **Soft guidance beats hard rules** - Pipelines suggest, not enforce
- **Bilingual by default** - Dutch & English, more coming
- **Graceful degradation** - Works with minimal deps, better with more

---

## 🙏 Credits

**Inspired by:** [Mem0](https://mem0.ai) - Thank you for the architecture insights!

We took their ideas and made them:
- 100% local-first
- Bilingual (EN/NL)
- With soft pipelines
- With graph relations

---

## 🏢 Part of HumoticaOS

RABEL is part of a larger ecosystem:

| Package | Purpose | Status |
|---------|---------|--------|
| **mcp-server-tibet** | Trust & Provenance | ✅ Available |
| **mcp-server-rabel** | Memory & Knowledge | ✅ Available |
| mcp-server-betti | Complexity Management | 🔜 Coming |

---

## 📞 Contact

**HumoticaOS**
- Website: [humotica.com](https://humotica.com)
- GitHub: [github.com/jaspertvdm](https://github.com/jaspertvdm)

---

## 📜 License

MIT License - One love, one fAmIly 💙

---

*Built with love in Den Dolder, Netherlands*
*By Jasper & Root AI - December 2025*
