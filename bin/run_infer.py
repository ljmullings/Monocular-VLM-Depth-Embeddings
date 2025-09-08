#!/usr/bin/env python3
"""CLI entrypoint for MVDE inference."""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from mvde.pipelines.infer import main

if __name__ == "__main__":
    main()
