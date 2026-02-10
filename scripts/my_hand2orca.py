import numpy as np
import time
from orca_core import OrcaHand

# ---------- geometry ----------
def angle_3pts(a, b, c):
    """Return angle ABC in radians, with points as np.array([x,y,z])."""
    v1 = a - b # 向量BA
    v2 = c - b # 向量BC
    n1 = np.linalg.norm(v1) # 向量BA模长
    n2 = np.linalg.norm(v2) # 向量BC模长
    if n1 < 1e-8 or n2 < 1e-8:
        return None
    cosang = np.dot(v1, v2) / (n1 * n2)
    cosang = np.clip(cosang, -1.0, 1.0)
    return float(np.arccos(cosang))

def clip01(x):
    return float(np.clip(x, 0.0, 1.0))

def map_rom(theta_h, hmin, hmax, omin, omax, invert=False):
    if theta_h is None:
        return None
    denom = (hmax - hmin)
    if abs(denom) < 1e-8:
        return None
    r = (theta_h - hmin) / denom
    r = clip01(r)
    if invert:
        r = 1.0 - r
    return omin + r * (omax - omin)

# ---------- smoothing / rate limit ----------
class JointFilter:
    def __init__(self, alpha=0.8, max_step_deg=4.0):
        self.alpha = alpha
        self.max_step = np.deg2rad(max_step_deg)
        self.prev = {}  # joint_name -> last value

    def update(self, joint_name, value):
        if value is None:
            return None
        if joint_name not in self.prev or self.prev[joint_name] is None:
            self.prev[joint_name] = value
            return value

        # EMA
        sm = self.alpha * self.prev[joint_name] + (1 - self.alpha) * value

        # rate limit
        delta = sm - self.prev[joint_name]
        if abs(delta) > self.max_step:
            sm = self.prev[joint_name] + np.sign(delta) * self.max_step

        self.prev[joint_name] = sm
        return sm

# ---------- your landmark -> np.array ----------
def lm(landmarks, idx):
    p = landmarks[idx]
    return np.array([p["x"], p["y"], p["z"]], dtype=np.float32)

# ---------- mapping config ----------
# MediaPipe indices:
WRIST = 0
# index: MCP=5, PIP=6, DIP=7, TIP=8
# middle: 9,10,11,12
# ring: 13,14,15,16
# pinky: 17,18,19,20

FINGERS = {
    "index":  {"mcp":5,  "pip":6,  "dip":7,  "tip":8},
    "middle": {"mcp":9,  "pip":10, "dip":11, "tip":12},
    "ring":   {"mcp":13, "pip":14, "dip":15, "tip":16},
    "pinky":  {"mcp":17, "pip":18, "dip":19, "tip":20},
}

# 你需要提前离线得到人手 ROM（建议用 p5/p95）
# 单位：弧度；这里只是示例结构
HUMAN_ROM = {
    "index_mcp":  (0.2, 1.3),
    "index_pip":  (0.2, 1.6),
    "middle_mcp": (0.2, 1.3),
    "middle_pip": (0.2, 1.6),
    "ring_mcp":   (0.2, 1.3),
    "ring_pip":   (0.2, 1.6),
    "pinky_mcp":  (0.2, 1.3),
    "pinky_pip":  (0.2, 1.6),
}

# ORCA ROM 从 config.yaml 读到的是“关节角单位”（看起来是 degree）
# OrcaHand 内部 joint_pos 用的是和 config.yaml 同单位（你这里需确认：是 degree 还是 rad）
# 如果你的 ORCA joint_roms 是 degree，那这里就用 degree；若是 rad，就统一 rad。
# 假设你的 ORCA ROM 是 degree：(-20, 108) 之类
ORCA_ROM = {
    "index_mcp":  (-20, 95),
    "index_pip":  (-20, 108),
    "middle_mcp": (-20, 91),
    "middle_pip": (-20, 107),
    "ring_mcp":   (-20, 91),
    "ring_pip":   (-20, 107),
    "pinky_mcp":  (-20, 98),
    "pinky_pip":  (-20, 108),
}

# 如果 ORCA 关节期望单位是 degree，而你算出来的人手角是 rad：
RAD2DEG = 180.0 / np.pi

# 方向是否需要反转：你做一根手指测试一下就能填
INVERT = {
    "index_mcp": False,
    "index_pip": False,
    "middle_mcp": False,
    "middle_pip": False,
    "ring_mcp": False,
    "ring_pip": False,
    "pinky_mcp": False,
    "pinky_pip": False,
}

def compute_human_angles(landmarks):
    out = {}

    w = lm(landmarks, WRIST)

    for name, ids in FINGERS.items():
        mcp = lm(landmarks, ids["mcp"])
        pip = lm(landmarks, ids["pip"])
        dip = lm(landmarks, ids["dip"])
        tip = lm(landmarks, ids["tip"])

        # MCP flexion proxy: angle(PIP - MCP - WRIST)
        mcp_ang = angle_3pts(pip, mcp, w)
        # PIP flexion proxy: angle(DIP - PIP - MCP)  (也可用 TIP)
        pip_ang = angle_3pts(dip, pip, mcp)

        out[f"{name}_mcp"] = mcp_ang
        out[f"{name}_pip"] = pip_ang

    return out

def human_to_orca(human_angles):
    cmd = {}
    for j, theta_h in human_angles.items():
        if theta_h is None:
            cmd[j] = None
            continue

        hmin, hmax = HUMAN_ROM[j]           # rad
        omin, omax = ORCA_ROM[j]            # deg (assumed)

        # 把 human(rad) -> 映射比例 -> orca(deg)
        theta_orca = map_rom(theta_h, hmin, hmax, omin, omax, invert=INVERT.get(j, False))
        cmd[j] = theta_orca

    return cmd

def main_stream(get_next_landmarks, model_path):
    hand = OrcaHand(model_path)
    ok, msg = hand.connect()
    print(ok, msg)
    if not ok:
        return

    hand.init_joints()  # 包含 calibrate(如需要) + move to neutral
    filt = JointFilter(alpha=0.85, max_step_deg=4.0)

    target_hz = 25
    dt = 1.0 / target_hz
    last = time.time()

    while True:
        landmarks = get_next_landmarks()   # 你自己接 MediaPipe，这里返回 list[21] dict: {"x","y","z"}
        if landmarks is None:
            time.sleep(0.005)
            continue

        human_angles = compute_human_angles(landmarks)
        orca_cmd = human_to_orca(human_angles)

        # smooth + send
        send_dict = {}
        for j, v in orca_cmd.items():
            send_dict[j] = filt.update(j, v)

        hand.set_joint_pos(send_dict, num_steps=1)

        # rate control
        now = time.time()
        sleep_t = dt - (now - last)
        if sleep_t > 0:
            time.sleep(sleep_t)
        last = time.time()

