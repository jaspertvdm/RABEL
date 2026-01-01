"""
RABEL Test Configuration
"""

import pytest
import sys
from pathlib import Path

# Ensure src is in path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
