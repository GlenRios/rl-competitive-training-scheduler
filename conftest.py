"""
conftest.py — configuración raíz de pytest.

Añade el directorio raíz del proyecto al sys.path para que los imports
absolutos (from src.data.fetcher import ...) funcionen desde cualquier
lugar donde se ejecute pytest.
"""

import sys
from pathlib import Path

# Inserta la raíz del proyecto al inicio del path
sys.path.insert(0, str(Path(__file__).parent))