import os
import sys
import numpy as np

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from common import config


def random_pose():
    """Generate a random pose near the configured glass position

    Returns:
        Tuple of (position, quaternion)
        - position: [x, y, z] with random offset in XY plane
        - quaternion: [w, x, y, z] with random yaw rotation
    """
    xy_offset = np.random.uniform(-0.02, 0.02, size=2)
    pos = config.GLASS_POSITION.copy()
    pos[:2] += xy_offset

    yaw_rad = np.deg2rad(np.random.uniform(-10.0, 10.0))
    half = yaw_rad / 2.0
    quat = np.array([np.cos(half), 0.0, 0.0, np.sin(half)], dtype=np.float64)
    return pos, quat

poses = [random_pose() for _ in range(10)]
for i, (p, q) in enumerate(poses, 1):
    print(f"Pose {i}: position={p}, orientation={q}")