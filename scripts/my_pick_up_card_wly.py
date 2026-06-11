from orca_core import OrcaHand
import time
import argparse

def main():
    parser = argparse.ArgumentParser(
        description="Demo: Pick up a thin card from a table using thumb and index finger pinch grasp."
    )
    parser.add_argument(
        "model_path",
        type=str,
        nargs="?",
        default=None,
        help="Path to the orcahand model folder (e.g., /path/to/orcahand_v1)"
    )
    args = parser.parse_args()

    # Initialize the hand
    hand = OrcaHand(args.model_path)
    status = hand.connect()
    print(status)

    if not status[0]:
        print("Failed to connect to the hand.")
        exit()

    hand.enable_torque()
    print("Torque enabled")

    print("\n=== Pick Up Card Demo ===")
    print("This will pick up a thin card from a table using thumb and index finger:")
    print("1. Open hand and curl unused fingers away")
    print("2. Pre-pinch position (thumb and index finger open, approaching card)")
    print("3. Pinch the card (close thumb and index finger)")
    print("4. Lift / hold the card")
    print("5. Release the card")
    print("6. Return to neutral\n")

    try:
        # Step 1: Open hand and curl middle, ring, pinky fingers out of the way
        print("Step 1: Preparing hand – curling unused fingers away...")
        prepare_pose = {
            'thumb_mcp': 0,
            'thumb_abd': 35,
            'thumb_pip': 0,
            'thumb_dip': 0,
            'index_mcp': 0,
            'index_pip': 0,
            'index_abd': 0,
            'middle_mcp': 70,
            'middle_pip': 80,
            'ring_mcp': 70,
            'ring_pip': 80,
            'ring_abd': 0,
            'pinky_mcp': 70,
            'pinky_pip': 80,
            'pinky_abd': 0,
            'wrist': 0,
        }
        hand.set_joint_pos(prepare_pose, num_steps=60, step_size=0.02)
        time.sleep(1.5)

        # Step 2: Pre-pinch position – thumb and index finger open wide,
        #         ready to approach a card lying flat on the table
        # Step 2: Pre-pinch position (这是你刚才拖动示教录制的精准位置)
        print("Step 2: Moving to pre-pinch position...")
        pre_pinch_pose = {
            'thumb_mcp': -39.0641,
            'thumb_abd': -4.0388,
            'thumb_pip': 22.7289,
            'thumb_dip': 42.0513,
            'index_abd': 9.8634,
            'index_mcp': 23.9221,
            'index_pip': 42.5020,
            'middle_mcp': 70,
            'middle_pip': 80,
            'ring_abd': 0,
            'ring_mcp': 70,
            'ring_pip': 80,
            'pinky_abd': 0,            
            'pinky_mcp': 70,
            'pinky_pip': 80,
            'wrist': -93.0924,
        }

        hand.set_joint_pos(pre_pinch_pose, num_steps=60, step_size=0.02)
        time.sleep(1.5)

        # Step 3: 主动触碰检测 (Active Contact Detection)
        print("Step 3: Pinching the card (Active Contact Detection)...")

        THUMB_MCP_IDX = 0
        THUMB_PIP_IDX = 2  
        INDEX_MCP_IDX = 5 
        INDEX_PIP_IDX = 6  

        error_threshold = 4.0 

        # --- 3A: 大拇指主动闭合并检测 ---
        print("  -> 3A: Thumb closing until contact...")
        thumb_steps = 40
        
        # 【关键修复】：先读取大拇指当前的真实物理位置作为起点
        start_pose = hand.get_joint_pos()
        base_thumb_mcp = start_pose[THUMB_MCP_IDX] if start_pose else pre_pinch_pose['thumb_mcp']
        base_thumb_pip = start_pose[THUMB_PIP_IDX] if start_pose else pre_pinch_pose['thumb_pip']

        for step in range(thumb_steps):
            progress = (step + 1) / thumb_steps
            # 目标：以【真实起点】为基础，向内压 30 度
            cmd_mcp = base_thumb_mcp + progress * 30
            cmd_pip = base_thumb_pip + progress * 30

            hand.set_joint_pos({
                'thumb_mcp': cmd_mcp,
                'thumb_pip': cmd_pip
            }, num_steps=1)
            time.sleep(0.03)

            actual_pose_list = hand.get_joint_pos()
            if actual_pose_list is not None and len(actual_pose_list) >= 16:
                actual_mcp = actual_pose_list[THUMB_MCP_IDX]
                actual_pip = actual_pose_list[THUMB_PIP_IDX]

                err_mcp = abs(actual_mcp - cmd_mcp)
                err_pip = abs(actual_pip - cmd_pip)

                if err_mcp > error_threshold or err_pip > error_threshold:
                    print(f"     ✅ [检测触发] 大拇指在第 {step} 步接触到卡片！停止闭合。")
                    break 

        time.sleep(0.5)

        # --- 3B: 食指主动闭合并检测 ---
        print("  -> 3B: Index closing until contact...")
        index_steps = 40
        
        # 【关键修复】：先读取食指当前的真实物理位置作为起点
        start_pose = hand.get_joint_pos()
        base_index_mcp = start_pose[INDEX_MCP_IDX] if start_pose else pre_pinch_pose['index_mcp']
        base_index_pip = start_pose[INDEX_PIP_IDX] if start_pose else pre_pinch_pose['index_pip']

        for step in range(index_steps):
            progress = (step + 1) / index_steps
            # 目标：以【真实起点】为基础，向内勾 40 度
            cmd_mcp = base_index_mcp + progress * 40
            cmd_pip = base_index_pip + progress * 40

            hand.set_joint_pos({
                'index_mcp': cmd_mcp,
                'index_pip': cmd_pip
            }, num_steps=1)
            time.sleep(0.03)

            actual_pose_list = hand.get_joint_pos()
            if actual_pose_list is not None and len(actual_pose_list) >= 16:
                actual_mcp = actual_pose_list[INDEX_MCP_IDX]
                actual_pip = actual_pose_list[INDEX_PIP_IDX]

                err_mcp = abs(actual_mcp - cmd_mcp)
                err_pip = abs(actual_pip - cmd_pip)
                
                # 如果你想看误差具体是怎么变大的，可以取消下面这行的注释：
                # print(f"     [Debug] Step {step}: cmd={cmd_mcp:.1f}, actual={actual_mcp:.1f}, err={err_mcp:.1f}")

                if err_mcp > error_threshold or err_pip > error_threshold:
                    print(f"     ✅ [检测触发] 食指在第 {step} 步接触到卡片！停止闭合。")
                    break 

        time.sleep(1.0)

        # Step 4: Lift the card – 仅控制手腕抬起
        print("Step 4: Lifting the card...")
        # 基于 Step 2 的真实手腕位置，往上抬起 120 度
        lift_wrist_angle = pre_pinch_pose['wrist'] + 120 
        hand.set_joint_pos({'wrist': lift_wrist_angle}, num_steps=60, step_size=0.02)
        time.sleep(2.0)
        
        # Step 5: Release the card – open thumb and index finger
        print("Step 5: Releasing the card...")
        hand.set_joint_pos(pre_pinch_pose, num_steps=60, step_size=0.02)
        time.sleep(1.5)

        # Step 6: Return to neutral position
        print("Step 6: Returning to neutral position...")
        hand.set_neutral_position(num_steps=80, step_size=0.02)
        time.sleep(2.0)

        print("\nPick-up card demo completed successfully!")

    except KeyboardInterrupt:
        print("\nDemo interrupted by user.")
    except Exception as e:
        print(f"\nError during demo: {e}")
    finally:
        print("Disabling torque...")
        hand.disable_torque()
        hand.disconnect()
        print("Torque disabled. Exiting.")

if __name__ == "__main__":
    main()
