# SPDX-License-Identifier: GPL-3.0-only
# Copyright (C) 2026 LabCoreAI

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
