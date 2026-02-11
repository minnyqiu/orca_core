from orca_core import OrcaHand
import time

def main():
    hand = OrcaHand()
    status = hand.connect()
    hand.enable_torque()
    # hand.calibrate()

    # Set the desired joint positions to 0
    hand.set_joint_pos({joint: 0 for joint in hand.joint_ids})
    time.sleep(0.3)

    print(hand.get_joint_pos())
    
    hand.disable_torque()
    hand.disconnect()


if __name__ == "__main__":
    main()