"""
Pytest configuration file.
Ensures proper module imports and paths for testing.
"""
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Remove any local mlflow directory from being treated as a package
# by ensuring it's not in sys.path
mlflow_local_dir = project_root / "mlflow"
if str(mlflow_local_dir) in sys.path:
    sys.path.remove(str(mlflow_local_dir))
