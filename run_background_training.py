#!/usr/bin/env python3
"""
Background AI Training Script

This script runs AI training silently in the background to improve
the ETL pipeline's accuracy without user intervention.
"""

import os
import sys
import subprocess
from pathlib import Path

def run_silent_training():
    """Run AI training silently in the background."""
    try:
        # Set environment to reduce logging
        env = os.environ.copy()
        env['LOG_LEVEL'] = 'ERROR'  # Only show errors
        
        # Run training script silently
        result = subprocess.run(
            [sys.executable, '-m', 'src.ml.train_ai_agent'],
            capture_output=True,
            text=True,
            env=env,
            cwd=Path(__file__).parent
        )
        
        # Only show output if there's an error
        if result.returncode != 0:
            print(f"Training error: {result.stderr}")
        else:
            print("✅ Background AI training completed successfully")
            
    except Exception as e:
        print(f"❌ Background training failed: {e}")

if __name__ == "__main__":
    run_silent_training() 