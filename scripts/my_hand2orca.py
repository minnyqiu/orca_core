import glob, time, yaml
import numpy as np
from orca_core import OrcaHand
import argparse

def angle_abc(A, B, C, eps=1e-9):
    BA = A - B
    BC = C - B
    cosang = np.dot(BA, BC) / (np.linalg.norm(BA)*np.linalg.norm(BC) + eps)
    cosang = np.clip(cosang, -1.0, 1.0)
    return np.degrees(np.arccos(cosang))

def load_landmarks_from_yaml(path):
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    lms = data["hand_frame"]["landmarks"]
    p = np.array([[lm["x"], lm["y"], lm["z"]] for lm in lms], dtype=float)
    return p, data["hand_frame"].get("confidence", 1.0)

def retarget_bend_only(p):
    """
    第一版 只做弯曲 不做abd/wrist。
    返回关节角(度): thumb_mcp/pip/dip + 四指mcp/pip
    """
    joint = {}

    # 伸直180° -> bend≈0；弯曲 -> bend变大
    def bend(A, B, C):
        return 180.0 - angle_abc(A, B, C)

    # Index
    joint["index_mcp"] = bend(p[0], p[5], p[6])
    joint["index_pip"] = bend(p[5], p[6], p[7])

    # Middle
    joint["middle_mcp"] = bend(p[0], p[9], p[10])
    joint["middle_pip"] = bend(p[9], p[10], p[11])

    # Ring
    joint["ring_mcp"] = bend(p[0], p[13], p[14])
    joint["ring_pip"] = bend(p[13], p[14], p[15])

    # Pinky
    joint["pinky_mcp"] = bend(p[0], p[17], p[18])
    joint["pinky_pip"] = bend(p[17], p[18], p[19])

    # Thumb（先简化）
    joint["thumb_mcp"] = bend(p[1], p[2], p[3])
    joint["thumb_pip"] = bend(p[2], p[3], p[4])
    joint["thumb_dip"] = 0.7 * joint["thumb_pip"]

    # abd/wrist先固定
    joint["thumb_abd"] = 0
    joint["index_abd"] = 0
    joint["middle_abd"] = 0
    joint["ring_abd"] = 0
    joint["pinky_abd"] = 0
    joint["wrist"] = 0

    return joint

def main():
    parser = argparse.ArgumentParser(description="Test the ORCA Hand.")  # Added parser
    parser.add_argument('model_path', type=str, nargs='?', default=None, help='Path to the hand model directory')
    args = parser.parse_args()
    hand = OrcaHand(args.model_path)

    status = hand.connect()
    print(status)
    if not status[0]:
        print("Failed to connect to the hand.")
        exit(1)

    hand.enable_torque()
    hand.init_joints(calibrate=False)

    files = sorted(glob.glob("data/hook/left*.yaml")) # /Users/minnyqiu/Documents/orca_core/data/hook/left0.yaml
    print("frames:", len(files))

    # 简单低通滤波，减少抖动
    alpha = 0.3
    last = None

    for fp in files:
        p, conf = load_landmarks_from_yaml(fp)
        if conf < 0.2:
            continue

        joint = retarget_bend_only(p)

        # 低通滤波
        if last is None:
            last = joint
        else:
            for k in joint:
                joint[k] = (1 - alpha) * last[k] + alpha * joint[k]
            last = joint

        # 小步平滑移动（你可以调）
        hand.set_joint_pos(joint, num_steps=3, step_size=0.002)

        time.sleep(0.033)  # 约30fps

if __name__ == "__main__":
    main()
