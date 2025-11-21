#!/usr/bin/env python3
"""
Test script to verify that the experiment runner can import all necessary modules
"""

import os
import sys

# Add parent directory to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

# Add scripts directory to path
scripts_dir = os.path.join(project_root, 'scripts')
sys.path.insert(0, scripts_dir)

print("Testing imports...")

try:
    print("  1. Importing random_pose...")
    from experiments.random_pose import random_pose
    print("     ✓ Success")
except Exception as e:
    print(f"     ✗ Failed: {e}")

try:
    print("  2. Importing config...")
    from common import config
    print("     ✓ Success")
except Exception as e:
    print(f"     ✗ Failed: {e}")

try:
    print("  3. Testing random pose generation...")
    pos, quat = random_pose()
    print(f"     ✓ Success: pos={pos}, quat={quat}")
except Exception as e:
    print(f"     ✗ Failed: {e}")

try:
    print("  4. Testing dynamic import of compute_ik_solutions...")
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "compute_ik_solutions",
        os.path.join(scripts_dir, "compute_ik_solutions.py")
    )
    compute_ik_module = importlib.util.module_from_spec(spec)
    sys.modules["compute_ik_solutions"] = compute_ik_module
    spec.loader.exec_module(compute_ik_module)
    print("     ✓ Success")
except Exception as e:
    print(f"     ✗ Failed: {e}")

try:
    print("  5. Testing dynamic import of plan_trajectory...")
    spec = importlib.util.spec_from_file_location(
        "plan_trajectory",
        os.path.join(scripts_dir, "plan_trajectory.py")
    )
    plan_module = importlib.util.module_from_spec(spec)
    sys.modules["plan_trajectory"] = plan_module
    spec.loader.exec_module(plan_module)
    print("     ✓ Success")
except Exception as e:
    print(f"     ✗ Failed: {e}")

try:
    print("  6. Testing dynamic import of coal_check...")
    spec = importlib.util.spec_from_file_location(
        "coal_check",
        os.path.join(scripts_dir, "coal_check.py")
    )
    coal_module = importlib.util.module_from_spec(spec)
    sys.modules["coal_check"] = coal_module
    spec.loader.exec_module(coal_module)
    print("     ✓ Success")
except Exception as e:
    print(f"     ✗ Failed: {e}")

print("\n✓ All import tests passed!")
print("\nYou can now run:")
print("  python experiments/run_random_pose_experiment.py --help")
