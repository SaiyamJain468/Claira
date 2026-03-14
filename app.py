# Entry point for deployment platforms.
# This app is built with Streamlit. Run via:
#   streamlit run app.py
# which re-exports the dashboard app.

import runpy
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

# Run the dashboard app
runpy.run_path("dashboard/app.py", run_name="__main__")
