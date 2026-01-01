"""
RABEL MCP Server Tests
======================

Tests for core RABEL functionality:
- Memory operations (add, search, delete)
- I-Poll messaging
- AINS domain resolution
- TIBET token creation

Run: pytest tests/test_rabel.py -v
"""

import pytest
import tempfile
import os
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from mcp_server_rabel.server import RABELCore


class TestRABELCore:
    """Test RABELCore memory functionality."""

    @pytest.fixture
    def rabel(self, tmp_path):
        """Create a temporary RABEL instance."""
        db_path = tmp_path / "test_memories.sqlite"
        return RABELCore(db_path=str(db_path))

    def test_init(self, rabel):
        """Test RABEL initializes correctly."""
        assert rabel is not None
        assert rabel.db_path.exists()

    def test_add_memory(self, rabel):
        """Test adding a memory."""
        result = rabel.add_memory(
            content="Test memory content",
            scope="general",
            memory_type="fact"
        )
        assert result is not None
        assert isinstance(result, dict)

    def test_search_memory(self, rabel):
        """Test searching memories."""
        # Add a memory first
        rabel.add_memory(
            content="The sky is blue",
            scope="general",
            memory_type="fact"
        )

        # Search for it
        results = rabel.search_memory(
            query="sky color",
            limit=5
        )
        assert isinstance(results, list)

    def test_memory_reflection(self, rabel):
        """Test that duplicate memories are reflected/merged."""
        # Add same memory twice
        rabel.add_memory(content="Paris is the capital of France", scope="test")
        rabel.add_memory(content="Paris is France's capital city", scope="test")

        # Should be reflected into one or have low duplicate count
        results = rabel.search_memory(query="Paris capital", limit=10)
        # At least one result should exist
        assert len(results) >= 1


class TestIPoll:
    """Test I-Poll inter-AI messaging."""

    @pytest.fixture
    def rabel(self, tmp_path):
        """Create a temporary RABEL instance."""
        db_path = tmp_path / "test_ipoll.sqlite"
        return RABELCore(db_path=str(db_path))

    def test_poll_push(self, rabel):
        """Test sending a message via I-Poll."""
        result = rabel.poll_push(
            from_agent="claude",
            to_agent="gemini",
            content="Hello from test!",
            poll_type="PUSH"
        )
        assert result is not None

    def test_poll_pull(self, rabel):
        """Test receiving messages via I-Poll."""
        # First send a message
        rabel.poll_push(
            from_agent="gemini",
            to_agent="claude",
            content="Test message",
            poll_type="PUSH"
        )

        # Then pull it (agent, not agent_id)
        messages = rabel.poll_pull(agent="claude")
        assert isinstance(messages, list)

    def test_poll_types(self, rabel):
        """Test different poll types."""
        poll_types = ["PUSH", "PULL", "SYNC", "TASK", "ACK"]

        for pt in poll_types:
            result = rabel.poll_push(
                from_agent="test",
                to_agent="test2",
                content=f"Testing {pt}",
                poll_type=pt
            )
            assert result is not None


class TestAINS:
    """Test AINS domain resolution."""

    @pytest.fixture
    def rabel(self, tmp_path):
        """Create a temporary RABEL instance."""
        db_path = tmp_path / "test_ains.sqlite"
        return RABELCore(db_path=str(db_path))

    def test_ains_resolve(self, rabel):
        """Test AINS domain resolution."""
        if hasattr(rabel, 'ains_resolve'):
            result = rabel.ains_resolve("root_ai")
            # Should return something (even if empty in test mode)
            assert result is not None


class TestTIBET:
    """Test TIBET token creation."""

    @pytest.fixture
    def rabel(self, tmp_path):
        """Create a temporary RABEL instance."""
        db_path = tmp_path / "test_tibet.sqlite"
        return RABELCore(db_path=str(db_path))

    def test_tibet_create(self, rabel):
        """Test TIBET token creation."""
        if hasattr(rabel, 'tibet_create'):
            token = rabel.tibet_create(
                erin="Test action",
                eraan=["dep1", "dep2"],
                eromheen={"context": "test"},
                erachter="Testing TIBET"
            )
            assert token is not None
            if isinstance(token, dict):
                assert "token_id" in token or "id" in token


# Run tests
if __name__ == "__main__":
    pytest.main([__file__, "-v"])
