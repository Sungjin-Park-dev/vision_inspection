#!/usr/bin/env python3
"""
Configuration Manager for Vision Inspection Experiments

Provides safe temporary modification of common/config.py during experiments.
Automatically backs up and restores configuration to prevent accidental changes.
"""

import os
import sys
import shutil
from pathlib import Path
from contextlib import contextmanager
from typing import Optional, Dict, Any
import tempfile

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from common import config


class ConfigManager:
    """Manages safe temporary modifications to config.py"""

    def __init__(self):
        self.config_path = Path(__file__).parent.parent / "common" / "config.py"
        self.backup_path: Optional[Path] = None

    def backup(self) -> Path:
        """Create a backup of config.py

        Returns:
            Path to backup file
        """
        if not self.config_path.exists():
            raise FileNotFoundError(f"Config file not found: {self.config_path}")

        # Create backup in temp directory with timestamp
        backup_dir = Path(tempfile.gettempdir()) / "vision_inspection_backups"
        backup_dir.mkdir(exist_ok=True)

        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.backup_path = backup_dir / f"config_backup_{timestamp}.py"

        shutil.copy2(self.config_path, self.backup_path)
        print(f"✓ Config backed up to: {self.backup_path}")

        return self.backup_path

    def restore(self):
        """Restore config.py from backup"""
        if self.backup_path is None or not self.backup_path.exists():
            raise ValueError("No backup found to restore")

        shutil.copy2(self.backup_path, self.config_path)
        print(f"✓ Config restored from: {self.backup_path}")

        # Clean up backup
        self.backup_path.unlink()
        self.backup_path = None

    def modify_value(self, variable_name: str, new_value: Any):
        """Modify a variable in config.py

        Args:
            variable_name: Name of the variable to modify (e.g., 'GLASS_POSITION')
            new_value: New value to set
        """
        # Read current config
        with open(self.config_path, 'r') as f:
            lines = f.readlines()

        # Find and replace the variable
        modified = False
        for i, line in enumerate(lines):
            # Look for variable assignment
            if line.strip().startswith(f"{variable_name} ="):
                # Format the new value
                if isinstance(new_value, str):
                    value_str = f'"{new_value}"'
                elif hasattr(new_value, '__iter__') and not isinstance(new_value, str):
                    # It's an array-like object
                    import numpy as np
                    if isinstance(new_value, np.ndarray):
                        # Convert numpy array to list representation
                        value_str = f"np.array({new_value.tolist()}, dtype=np.float64)"
                    else:
                        value_str = str(new_value)
                else:
                    value_str = str(new_value)

                # Replace line
                lines[i] = f"{variable_name} = {value_str}\n"
                modified = True
                print(f"  Modified {variable_name} = {value_str}")
                break

        if not modified:
            raise ValueError(f"Variable '{variable_name}' not found in config.py")

        # Write modified config
        with open(self.config_path, 'w') as f:
            f.writelines(lines)

    def reload_config(self):
        """Reload the config module to pick up changes"""
        import importlib
        importlib.reload(config)
        print("✓ Config module reloaded")


@contextmanager
def temporary_config(**modifications: Dict[str, Any]):
    """Context manager for temporary config modifications

    Usage:
        with temporary_config(
            GLASS_POSITION=np.array([1.0, 0.0, -0.15]),
            CAMERA_OVERLAP_RATIO=0.75
        ):
            # Run experiments with modified config
            run_pipeline()
        # Config automatically restored after block

    Args:
        **modifications: Variable name and value pairs to modify
    """
    manager = ConfigManager()

    try:
        # Backup original config
        manager.backup()

        # Apply modifications
        print(f"\nApplying temporary config modifications:")
        for var_name, value in modifications.items():
            manager.modify_value(var_name, value)

        # Reload config module
        manager.reload_config()

        yield manager

    finally:
        # Always restore config, even if exception occurs
        try:
            manager.restore()
            manager.reload_config()
        except Exception as e:
            print(f"⚠️  Warning: Failed to restore config: {e}")
            if manager.backup_path and manager.backup_path.exists():
                print(f"   Manual restore required from: {manager.backup_path}")


def main():
    """Test config manager"""
    import numpy as np

    print("Testing ConfigManager...")
    print(f"Original GLASS_POSITION: {config.GLASS_POSITION}")
    print(f"Original CAMERA_OVERLAP_RATIO: {config.CAMERA_OVERLAP_RATIO}")

    # Test temporary modification
    new_position = np.array([1.5, 0.5, -0.2], dtype=np.float64)
    new_overlap = 0.75

    print(f"\nTesting temporary modification...")
    with temporary_config(
        GLASS_POSITION=new_position,
        CAMERA_OVERLAP_RATIO=new_overlap
    ):
        # Reload to see changes
        import importlib
        importlib.reload(config)

        print(f"Modified GLASS_POSITION: {config.GLASS_POSITION}")
        print(f"Modified CAMERA_OVERLAP_RATIO: {config.CAMERA_OVERLAP_RATIO}")

        assert np.allclose(config.GLASS_POSITION, new_position), "Position not modified!"
        assert config.CAMERA_OVERLAP_RATIO == new_overlap, "Overlap not modified!"
        print("✓ Modifications verified")

    # Reload to verify restoration
    import importlib
    importlib.reload(config)

    print(f"\nAfter context exit:")
    print(f"Restored GLASS_POSITION: {config.GLASS_POSITION}")
    print(f"Restored CAMERA_OVERLAP_RATIO: {config.CAMERA_OVERLAP_RATIO}")

    print("\n✓ ConfigManager test passed!")


if __name__ == "__main__":
    main()
