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
        print("Step 2: Moving to pre-pinch position...")
        pre_pinch_pose = {
            'thumb_mcp': 20,
            'thumb_abd': 35,
            'thumb_pip': 30,
            'thumb_dip': 20,
            'index_mcp': 30,
            'index_pip': 20,
            'index_abd': 0,
            'middle_mcp': 70,
            'middle_pip': 80,
            'ring_mcp': 70,
            'ring_pip': 80,
            'ring_abd': 0,
            'pinky_mcp': 70,
            'pinky_pip': 80,
            'pinky_abd': 0,
            'wrist': -10,
        }
        hand.set_joint_pos(pre_pinch_pose, num_steps=60, step_size=0.02)
        time.sleep(1.5)

        # Step 3: Pinch the card – gradually close thumb and index finger tips together
        print("Step 3: Pinching the card...")
        pinch_pose = {
            'thumb_mcp': 25,
            'thumb_abd': 35,
            'thumb_pip': 70,
            'thumb_dip': 80,
            'index_mcp': 65,
            'index_pip': 75,
            'index_abd': 0,
            'middle_mcp': 70,
            'middle_pip': 80,
            'ring_mcp': 70,
            'ring_pip': 80,
            'ring_abd': 0,
            'pinky_mcp': 70,
            'pinky_pip': 80,
            'pinky_abd': 0,
            'wrist': -10,
        }
        # Use fine-grained interpolation for a gentle pinch
        pinch_steps = 80
        for step in range(pinch_steps):
            progress = (step + 1) / pinch_steps
            interp_pose = {}
            for joint in pre_pinch_pose:
                start_val = pre_pinch_pose[joint]
                end_val = pinch_pose[joint]
                interp_pose[joint] = start_val + progress * (end_val - start_val)
            hand.set_joint_pos(interp_pose, num_steps=1)
            time.sleep(0.03)
        time.sleep(1.0)

        # Step 4: Lift the card – slightly extend the wrist while maintaining pinch
        print("Step 4: Lifting the card...")
        lift_pose = pinch_pose.copy()
        lift_pose['wrist'] = 10
        hand.set_joint_pos(lift_pose, num_steps=60, step_size=0.02)
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
