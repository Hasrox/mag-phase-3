"""Launch the play-side screen. Analytics off. Loopback only. No model."""

from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ["GRADIO_ANALYTICS_ENABLED"] = "False"
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mag.ui import main

if __name__ == "__main__":
    main()
