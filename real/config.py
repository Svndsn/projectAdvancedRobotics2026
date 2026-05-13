# real/config.py
import spatialmath as sm
from .utils import make_tf

ROBOT_IP = "192.168.1.158"

# WSG50 gripper TCP/IP text interface defaults.
WSG50_IP = "192.168.1.20"
WSG50_PORT = 1000
WSG50_OPEN_WIDTH_MM = 100.0
WSG50_PREGRIP_WIDTH_MM = 80.0
WSG50_GRIP_WIDTH_MM = 10.0
WSG50_CLOSE_WIDTH_MM = 0.0
WSG50_GRIP_FALLBACK_TO_MOVE = True
WSG50_GRIP_FORCE_N = 30.0
WSG50_GRIP_SPEED_MM_S = 50.0
WSG50_RELEASE_DISTANCE_MM = 20.0

# UR flange TCP -> active gripper/tool TCP transform.
T_TCP_GRIPPER_TCP = sm.SE3.Tz(0.12)

# world -> robot base transform
UR5E_BASE_POS = [0.0, 0.0, 0.0]
UR5E_BASE_QUAT = [1.0, 0.0, 0.0, 0.0]  # [w, x, y, z]

# optional initial follow pose
T_FOLLOW_INIT = make_tf(
    pos=[0.4, 0.0, 0.2],
    ori=[1.0, 0.0, 0.0, 0.0],
)
