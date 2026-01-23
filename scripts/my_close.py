from orca_core import OrcaHand
import time
import numpy as np
import argparse

def main():
    parser = argparse.ArgumentParser(
        description="Demo: Close four fingers together, flex, then extend while keeping them adducted."
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

    # Joint ranges of motion (ROMs)
    joint_roms = {
        'thumb_mcp': [-50, 50],
        'thumb_abd': [-20, 42],
        'thumb_pip': [-12, 108],
        'thumb_dip': [-20, 112],
        'index_mcp': [-20, 95],
        'index_pip': [-20, 108],
        'index_abd': [-37, 37],
        'middle_mcp': [-20, 91],
        'middle_pip': [-20, 107],
        'ring_mcp': [-20, 91],
        'ring_pip': [-20, 107],
        'ring_abd': [-37, 37],
        'pinky_mcp': [-20, 98],
        'pinky_pip': [-20, 108],
        'pinky_abd': [-37, 37],
        'wrist': [-50, 30],
    }

    # Step 1: 并拢四指
    close_pose = hand.neutral_position.copy()
    close_pose.update({
        "index_abd": -28,
        "ring_abd": 16,
        "pinky_abd": 35,
    })
    print("Step 1: Closing fingers together...")
    hand.set_joint_pos(close_pose, num_steps=60, step_size=0.02)
    time.sleep(2)

    # Step 2: 缓缓屈曲 MCP + PIP
    print("Step 2: Flexing MCP + PIP joints...")
    flex_pose = close_pose.copy()
    flex_steps = 80
    for step in range(flex_steps):
        progress = (step + 1) / flex_steps
        for joint in ["index_mcp", "index_pip", "middle_mcp", "middle_pip", "ring_mcp", "ring_pip", "pinky_mcp", "pinky_pip"]:
            rom_min, rom_max = joint_roms[joint]
            flex_pose[joint] = (1 - progress) * close_pose[joint] + progress * rom_max
        hand.set_joint_pos(flex_pose, num_steps=1)
        time.sleep(0.05)
    time.sleep(2)

    # Step 3: 伸直 MCP + PIP（保持并拢）
    print("Step 3: Extending MCP + PIP joints back to straight while keeping fingers adducted...")
    extend_pose = close_pose.copy()
    hand.set_joint_pos(extend_pose, num_steps=80, step_size=0.02)
    time.sleep(2)

    print("Sequence complete. Returning to neutral...")
    hand.set_neutral_position(num_steps=80, step_size=0.02)

    hand.disable_torque()
    hand.disconnect()
    print("Done.")

if __name__ == "__main__":
    main()
