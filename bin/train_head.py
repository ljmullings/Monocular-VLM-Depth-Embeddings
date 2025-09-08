#!/usr/bin/env python3
"""CLI entrypoint for training distance heads."""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from mvde.pipelines.train_head import main

if __name__ == "__main__":
    main()
