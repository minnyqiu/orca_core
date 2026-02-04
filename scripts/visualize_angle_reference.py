"""
可视化演示：角度定义说明
展示不同角度值对应的手指姿态
"""
import yaml
import numpy as np
import glob

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
    return p, data["hand_frame"].get("confidence", 1.0), data["hand_frame"]["frame_id"]

def analyze_finger_state(p):
    """分析手指状态"""
    def bend(A, B, C):
        return 180.0 - angle_abc(A, B, C)
    
    # 计算食指MCP角度作为示例
    index_mcp_angle = bend(p[0], p[5], p[6])
    
    # 计算所有手指的平均弯曲
    finger_angles = {
        "食指MCP": bend(p[0], p[5], p[6]),
        "食指PIP": bend(p[5], p[6], p[7]),
        "中指MCP": bend(p[0], p[9], p[10]),
        "中指PIP": bend(p[9], p[10], p[11]),
        "无名指MCP": bend(p[0], p[13], p[14]),
        "无名指PIP": bend(p[13], p[14], p[15]),
        "小指MCP": bend(p[0], p[17], p[18]),
        "小指PIP": bend(p[17], p[18], p[19]),
    }
    
    avg_angle = np.mean(list(finger_angles.values()))
    
    # 判断姿态
    if avg_angle < 10:
        state = "✋ 完全伸直 (0° 参考状态)"
    elif avg_angle < 30:
        state = "🖐️ 接近伸直"
    elif avg_angle < 60:
        state = "👋 轻微弯曲"
    elif avg_angle < 90:
        state = "🤚 半握状态"
    elif avg_angle < 120:
        state = "✊ 握拳状态"
    else:
        state = "👊 紧握"
    
    return finger_angles, avg_angle, state

def main():
    files = sorted(glob.glob("data/hook/left*.yaml"))
    
    if not files:
        print("未找到数据文件")
        return
    
    print("="*80)
    print("角度定义说明：")
    print("="*80)
    print("""
在我们的计算中：

  **0° = 手指完全伸直**（关节点A-B-C在一条直线上）
  
      食指示例：
      
      WRIST(A)               伸直状态 (bend ≈ 0°)
         |
         |
      MCP(B) ← 关节点
         |
         |
      PIP(C)
      
      
      WRIST(A)               弯曲状态 (bend > 0°)
         |
         |
      MCP(B) ← 关节点
          \\
           \\
         PIP(C)

弯曲角度越大，手指越弯曲。
""")
    print("="*80)
    print("\n正在分析您的数据中哪些帧最接近0°（伸直状态）...\n")
    
    # 找到最接近伸直的帧
    closest_to_zero = []
    
    for fp in files[:50]:  # 只分析前50帧
        p, conf, frame_id = load_landmarks_from_yaml(fp)
        if conf < 0.5:
            continue
        
        angles, avg_angle, state = analyze_finger_state(p)
        closest_to_zero.append((frame_id, avg_angle, state, fp, angles))
    
    # 按平均角度排序
    closest_to_zero.sort(key=lambda x: x[1])
    
    print("最接近 0° (伸直状态) 的前5帧:")
    print("-" * 80)
    for i, (frame_id, avg_angle, state, fp, angles) in enumerate(closest_to_zero[:5], 1):
        print(f"\n{i}. Frame {frame_id} - 平均弯曲: {avg_angle:.2f}° - {state}")
        print(f"   文件: {fp}")
        print(f"   各手指详情:")
        for joint_name, angle in angles.items():
            print(f"     {joint_name}: {angle:6.2f}°")
    
    print("\n" + "="*80)
    print("最弯曲的前5帧:")
    print("-" * 80)
    for i, (frame_id, avg_angle, state, fp, angles) in enumerate(reversed(closest_to_zero[-5:]), 1):
        print(f"\n{i}. Frame {frame_id} - 平均弯曲: {avg_angle:.2f}° - {state}")
        print(f"   文件: {fp}")
        print(f"   各手指详情:")
        for joint_name, angle in angles.items():
            print(f"     {joint_name}: {angle:6.2f}°")
    
    print("\n" + "="*80)
    print("结论：")
    print("  • 0° = 手指完全伸直（像在说'停'的手势）")
    print("  • 角度增加 = 手指越来越弯曲")
    print("  • 握拳时角度通常在 90°-120° 范围")
    print("="*80)

if __name__ == "__main__":
    main()
