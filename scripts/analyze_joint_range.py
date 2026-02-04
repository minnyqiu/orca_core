"""
分析人手关节数据，获取每个关节的运动范围（最极端位置）
"""
import glob
import yaml
import numpy as np
from collections import defaultdict

def angle_abc(A, B, C, eps=1e-9):
    """计算三点ABC构成的角度"""
    BA = A - B
    BC = C - B
    cosang = np.dot(BA, BC) / (np.linalg.norm(BA)*np.linalg.norm(BC) + eps)
    cosang = np.clip(cosang, -1.0, 1.0)
    return np.degrees(np.arccos(cosang))

def load_landmarks_from_yaml(path):
    """从YAML文件加载关节点数据"""
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    lms = data["hand_frame"]["landmarks"]
    p = np.array([[lm["x"], lm["y"], lm["z"]] for lm in lms], dtype=float)
    return p, data["hand_frame"].get("confidence", 1.0)

def calculate_joint_angles(p):
    """
    计算所有关节的弯曲角度
    
    角度定义说明：
    - 0°  = 手指完全伸直（三个关节点成一条直线）
    - 90° = 手指弯曲90度（形成直角）
    - 180° = 手指完全弯曲回折（极限弯曲）
    
    示意图：
        0°         45°        90°        135°
        |          /          _          C
        |         /          / |        /
        |        /          /  |       /
        B       B          B   |      B
        |                     C         \
        A                                 A
    """
    def bend(A, B, C):
        """
        弯曲角度 = 180° - 三点夹角
        
        A-B-C 三点，B是关节点
        - 伸直时(A-B-C成直线): 夹角≈180° → bend≈0°
        - 弯曲时: 夹角变小 → bend变大
        """
        return 180.0 - angle_abc(A, B, C)
    
    angles = {}
    
    # 食指
    angles["index_mcp"] = bend(p[0], p[5], p[6])
    angles["index_pip"] = bend(p[5], p[6], p[7])
    angles["index_dip"] = bend(p[6], p[7], p[8])
    
    # 中指
    angles["middle_mcp"] = bend(p[0], p[9], p[10])
    angles["middle_pip"] = bend(p[9], p[10], p[11])
    angles["middle_dip"] = bend(p[10], p[11], p[12])
    
    # 无名指
    angles["ring_mcp"] = bend(p[0], p[13], p[14])
    angles["ring_pip"] = bend(p[13], p[14], p[15])
    angles["ring_dip"] = bend(p[14], p[15], p[16])
    
    # 小指
    angles["pinky_mcp"] = bend(p[0], p[17], p[18])
    angles["pinky_pip"] = bend(p[17], p[18], p[19])
    angles["pinky_dip"] = bend(p[18], p[19], p[20])
    
    # 拇指
    angles["thumb_mcp"] = bend(p[1], p[2], p[3])
    angles["thumb_pip"] = bend(p[2], p[3], p[4])
    
    # 手指展开角度（abduction）
    # 食指相对中指的展开
    angles["index_abd"] = angle_abc(p[5], p[0], p[9])
    # 中指相对无名指的展开
    angles["middle_abd"] = angle_abc(p[9], p[0], p[13])
    # 无名指相对小指的展开
    angles["ring_abd"] = angle_abc(p[13], p[0], p[17])
    # 拇指展开（拇指CMC相对手腕和食指MCP的角度）
    angles["thumb_abd"] = angle_abc(p[1], p[0], p[5])
    
    return angles

def analyze_joint_ranges(data_dir="data/hook", min_confidence=0.2):
    """分析所有数据文件，获取每个关节的运动范围"""
    files = sorted(glob.glob(f"{data_dir}/left*.yaml"))
    
    if not files:
        print(f"错误：在 {data_dir} 中没有找到数据文件")
        return
    
    print(f"找到 {len(files)} 个数据文件")
    print(f"最小置信度阈值: {min_confidence}\n")
    
    # 存储每个关节的所有角度值
    joint_values = defaultdict(list)
    valid_frames = 0
    skipped_frames = 0
    
    # 记录每个关节的极值出现在哪一帧
    joint_min_frame = {}
    joint_max_frame = {}
    
    for i, fp in enumerate(files):
        p, conf = load_landmarks_from_yaml(fp)
        
        if conf < min_confidence:
            skipped_frames += 1
            continue
        
        valid_frames += 1
        angles = calculate_joint_angles(p)
        
        for joint_name, angle_value in angles.items():
            joint_values[joint_name].append(angle_value)
            
            # 记录极值帧
            if joint_name not in joint_min_frame or angle_value < min(joint_values[joint_name][:-1] or [float('inf')]):
                joint_min_frame[joint_name] = (i, fp, angle_value)
            if joint_name not in joint_max_frame or angle_value > max(joint_values[joint_name][:-1] or [float('-inf')]):
                joint_max_frame[joint_name] = (i, fp, angle_value)
    
    print(f"有效帧数: {valid_frames}")
    print(f"跳过帧数: {skipped_frames} (置信度过低)\n")
    
    print("="*80)
    print("角度参考说明:")
    print("  0°   = 手指完全伸直")
    print("  30°  = 轻微弯曲")
    print("  60°  = 半握状态")
    print("  90°  = 弯曲成直角")
    print("  120° = 深度弯曲（接近握拳）")
    print("="*80)
    
    # 按手指分组显示结果
    finger_groups = {
        "拇指 (Thumb)": ["thumb_mcp", "thumb_pip", "thumb_abd"],
        "食指 (Index)": ["index_mcp", "index_pip", "index_dip", "index_abd"],
        "中指 (Middle)": ["middle_mcp", "middle_pip", "middle_dip", "middle_abd"],
        "无名指 (Ring)": ["ring_mcp", "ring_pip", "ring_dip", "ring_abd"],
        "小指 (Pinky)": ["pinky_mcp", "pinky_pip", "pinky_dip"]
    }
    
    all_stats = {}
    
    for finger_name, joints in finger_groups.items():
        print(f"\n{finger_name}")
        print("-" * 80)
        
        for joint in joints:
            if joint not in joint_values:
                continue
            
            values = np.array(joint_values[joint])
            min_val = np.min(values)
            max_val = np.max(values)
            mean_val = np.mean(values)
            std_val = np.std(values)
            range_val = max_val - min_val
            
            all_stats[joint] = {
                'min': min_val,
                'max': max_val,
                'mean': mean_val,
                'std': std_val,
                'range': range_val
            }
            
            joint_type = "弯曲" if "abd" not in joint else "展开"
            
            print(f"  {joint:15s} ({joint_type}):")
            print(f"    最小值: {min_val:6.2f}° (帧 {joint_min_frame[joint][0]})")
            print(f"    最大值: {max_val:6.2f}° (帧 {joint_max_frame[joint][0]})")
            print(f"    范围:   {range_val:6.2f}°")
            print(f"    平均值: {mean_val:6.2f}° ± {std_val:.2f}°")
    
    # 总结：识别最活跃的关节
    print("\n" + "="*80)
    print("运动范围排名 (Top 10):")
    print("-" * 80)
    sorted_joints = sorted(all_stats.items(), key=lambda x: x[1]['range'], reverse=True)
    
    for i, (joint, stats) in enumerate(sorted_joints[:10], 1):
        joint_type = "弯曲" if "abd" not in joint else "展开"
        print(f"{i:2d}. {joint:15s} ({joint_type}): {stats['range']:6.2f}° "
              f"[{stats['min']:6.2f}° ~ {stats['max']:6.2f}°]")
    
    print("\n" + "="*80)
    print("建议的机械手关节映射参考值:")
    print("-" * 80)
    print("# 基于人手实际运动范围，建议的映射配置：")
    print("joint_mapping = {")
    for joint, stats in sorted(all_stats.items()):
        print(f"    '{joint}': ({stats['min']:.1f}, {stats['max']:.1f}),  # 范围: {stats['range']:.1f}°")
    print("}")
    
    return all_stats

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="分析人手关节运动范围")
    parser.add_argument('--data-dir', type=str, default='data/hook',
                        help='数据目录路径 (默认: data/hook)')
    parser.add_argument('--min-confidence', type=float, default=0.2,
                        help='最小置信度阈值 (默认: 0.2)')
    
    args = parser.parse_args()
    
    stats = analyze_joint_ranges(args.data_dir, args.min_confidence)
