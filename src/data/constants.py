"""Physical constants, COCO-17 topology, semantic schemas, and bucket definitions."""

from typing import Tuple

DT: float = 0.1
FS: float = 10.0
T_OBS_CHOICES: Tuple[int, ...] = (5, 10, 20)
T_FUT: int = 30
K_MODES: int = 6
NUM_JOINTS: int = 17
TAU_CONF: float = 0.20
MAX_VEL_GAP_SEC: float = 0.50
TEMPORAL_ISOLATION_SEC: float = 6.0

# COCO-17 keypoint indices:
# 0:nose, 1:l_eye, 2:r_eye, 3:l_ear, 4:r_ear, 5:l_sho, 6:r_sho,
# 7:l_elb, 8:r_elb, 9:l_wri, 10:r_wri, 11:l_hip, 12:r_hip,
# 13:l_knee, 14:r_knee, 15:l_ank, 16:r_ank
COCO17_EDGES: Tuple[Tuple[int, int], ...] = (
    (0, 1), (0, 2), (1, 3), (2, 4),
    (0, 5), (0, 6), (5, 6),
    (5, 7), (7, 9), (6, 8), (8, 10),
    (5, 11), (6, 12), (11, 12),
    (11, 13), (13, 15), (12, 14), (14, 16),
)

# Anatomical joint groups for biomechanical analysis & part-aware quality gating:
HEAD_SHOULDER_JOINTS: Tuple[int, ...] = (0, 1, 2, 3, 4, 5, 6)
UPPER_TORSO_ARMS_JOINTS: Tuple[int, ...] = (5, 6, 7, 8, 9, 10, 11, 12)
LOWER_LIMB_GAIT_JOINTS: Tuple[int, ...] = (11, 12, 13, 14, 15, 16)
LEFT_HIP_IDX: int = 11
RIGHT_HIP_IDX: int = 12

# 5 observable dynamic action events (Table 7)
ACTION_NAMES: Tuple[str, ...] = (
    "Start",
    "Stop",
    "Turn-Left",
    "Turn-Right",
    "Maintain",
)
NUM_ACTIONS: int = 5

# 8 heterogeneous traffic participant types (Table 6 right)
AGENT_TYPES: Tuple[str, ...] = (
    "normal_pedestrian",
    "encumbered_pedestrian",
    "standing_pedestrian",
    "bicycle",
    "ebike_scooter",
    "tricycle_cart",
    "passenger_car",
    "shuttle_van",
)
NUM_AGENT_TYPES: int = 8

# Quality buckets (Section 5.3): 0=Clear, 1=Partial, 2=Heavy
BUCKET_NAMES: Tuple[str, ...] = ("Clear", "Partial", "Heavy")
FIXED_SEEDS: Tuple[int, ...] = (42, 1024, 2025, 3407, 4096)
