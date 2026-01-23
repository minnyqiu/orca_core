from orca_core import OrcaHand
import time
import numpy as np
import argparse

def main():
    parser = argparse.ArgumentParser(
        description="Demo: Perform a grasping motion to pick up an object."
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

    print("\n=== Grasping Motion Demo ===")
    print("This will perform a natural grasping motion:")
    print("1. Open hand (pre-grasp position)")
    print("2. Close fingers to grasp object")
    print("3. Hold the grasp")
    print("4. Release object")
    print("5. Return to neutral\n")

    try:
        # Step 1: Move to pre-grasp position (open hand)
        print("Step 1: Opening hand to pre-grasp position...")
        pre_grasp_position = {
            'thumb_mcp': 20,
            'thumb_abd': 35,
            'thumb_pip': 10,
            'thumb_dip': 10,
            'index_mcp': -15,
            'index_pip': -10,
            'index_abd': 20,
            'middle_mcp': -15,
            'middle_pip': -10,
            'ring_mcp': -15,
            'ring_pip': -10,
            'ring_abd': 0,
            'pinky_mcp': -15,
            'pinky_pip': -10,
            'pinky_abd': 0,
            'wrist': -10,
        }
        hand.set_joint_angles(pre_grasp_position, moving_time=2.0)
        time.sleep(2.5)

        # Step 2: Close fingers to grasp the object
        print("Step 2: Closing fingers to grasp object...")
        
        # Gradual closing motion with interpolation
        grasp_steps = 30
        grasp_duration = 1.5  # seconds
        step_time = grasp_duration / grasp_steps
        
        # Target grasp position
        grasp_position = {
            'thumb_mcp': 25,
            'thumb_abd': 35,
            'thumb_pip': 65,
            'thumb_dip': 75,
            'index_mcp': 60,
            'index_pip': 80,
            'index_abd': 20,
            'middle_mcp': 55,
            'middle_pip': 75,
            'ring_mcp': 50,
            'ring_pip': 70,
            'ring_abd': 0,
            'pinky_mcp': 45,
            'pinky_pip': 65,
            'pinky_abd': 0,
            'wrist': -10,
        }
        
        # Interpolate from pre-grasp to grasp position
        for i in range(grasp_steps + 1):
            t = i / grasp_steps  # 0 to 1
            current_position = {}
            for joint in pre_grasp_position.keys():
                start_val = pre_grasp_position[joint]
                end_val = grasp_position[joint]
                current_position[joint] = start_val + t * (end_val - start_val)
            
            hand.set_joint_angles(current_position, moving_time=step_time * 0.9)
            time.sleep(step_time)

        # Step 3: Hold the grasp
        print("Step 3: Holding object...")
        time.sleep(2.0)

        # Step 4: Release the object
        print("Step 4: Releasing object...")
        hand.set_joint_angles(pre_grasp_position, moving_time=1.5)
        time.sleep(2.0)

        # Step 5: Return to neutral position
        print("Step 5: Returning to neutral position...")
        neutral_position = {
            'thumb_mcp': 0,
            'thumb_abd': 0,
            'thumb_pip': 0,
            'thumb_dip': 0,
            'index_mcp': 0,
            'index_pip': 0,
            'index_abd': 0,
            'middle_mcp': 0,
            'middle_pip': 0,
            'ring_mcp': 0,
            'ring_pip': 0,
            'ring_abd': 0,
            'pinky_mcp': 0,
            'pinky_pip': 0,
            'pinky_abd': 0,
            'wrist': 0,
        }
        hand.set_joint_angles(neutral_position, moving_time=2.0)
        time.sleep(2.5)

        print("\nGrasping demo completed successfully!")

    except KeyboardInterrupt:
        print("\nDemo interrupted by user.")
    except Exception as e:
        print(f"\nError during demo: {e}")
    finally:
        # Ensure torque is disabled before exiting
        print("Disabling torque...")
        hand.disable_torque()
        print("Torque disabled. Exiting.")

if __name__ == "__main__":
    main()
