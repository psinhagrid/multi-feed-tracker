#!/usr/bin/env python3
"""
Starter script for Multi-Feed Tracker
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# ==================== DEMO 1: Video Tracking (Commented) ====================
# from video_processing.video_processor import main as run_video_processor
# if __name__ == "__main__":
#     run_video_processor()

# ==================== DEMO 2: ROI Selector (Active) ====================
from tools.roi_selector import main as run_roi_selector

if __name__ == "__main__":
    run_roi_selector()
