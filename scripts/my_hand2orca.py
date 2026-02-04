import glob, time, yaml
import numpy as np
from orca_core import OrcaHand
import argparse
import os

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

def verify_angles_only(max_frames=None):
    """验证模式：只计算和显示角度，不连接机械手"""
    files = sorted(glob.glob("data/hook/left*.yaml"))
    if max_frames:
        files = files[:max_frames]
    
    print(f"=== 角度验证模式 ===")
    print(f"共 {len(files)} 帧数据\n")
    
    for i, fp in enumerate(files):
        p, conf = load_landmarks_from_yaml(fp)
        if conf < 0.2:
            print(f"Frame {i}: 置信度过低 ({conf:.2f}), 跳过")
            continue
        
        joint = retarget_bend_only(p)
        
        # 打印每帧的关节角度
        print(f"\n=== Frame {i} ({os.path.basename(fp)}) ===")
        print(f"置信度: {conf:.3f}")
        print("\n手指弯曲角度 (度):")
        print(f"  拇指:  MCP={joint['thumb_mcp']:6.2f}  PIP={joint['thumb_pip']:6.2f}  DIP={joint['thumb_dip']:6.2f}")
        print(f"  食指:  MCP={joint['index_mcp']:6.2f}  PIP={joint['index_pip']:6.2f}")
        print(f"  中指:  MCP={joint['middle_mcp']:6.2f}  PIP={joint['middle_pip']:6.2f}")
        print(f"  无名指: MCP={joint['ring_mcp']:6.2f}  PIP={joint['ring_pip']:6.2f}")
        print(f"  小指:  MCP={joint['pinky_mcp']:6.2f}  PIP={joint['pinky_pip']:6.2f}")
        
        # 添加简单的姿态描述
        avg_bend = np.mean([joint['index_mcp'], joint['middle_mcp'], joint['ring_mcp'], joint['pinky_mcp']])
        if avg_bend < 20:
            gesture = "手掌张开"
        elif avg_bend < 50:
            gesture = "手指轻微弯曲"
        elif avg_bend < 90:
            gesture = "手指半握"
        else:
            gesture = "手指紧握"
        print(f"\n姿态判断: {gesture} (平均弯曲角度: {avg_bend:.1f}°)")
        
        # 每10帧暂停一下
        if (i + 1) % 10 == 0:
            input(f"\n已显示 {i+1} 帧，按回车继续...")

def main():
    parser = argparse.ArgumentParser(description="Test the ORCA Hand.")  # Added parser
    parser.add_argument('model_path', type=str, nargs='?', default=None, help='Path to the hand model directory')
    parser.add_argument('--verify', action='store_true', help='Verify angles without connecting to hand')
    parser.add_argument('--max-frames', type=int, default=None, help='Max frames to process (for testing)')
    args = parser.parse_args()
    
    # 验证模式：只计算和显示角度，不连接机械手
    if args.verify:
        verify_angles_only(args.max_frames)
        return
    
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
