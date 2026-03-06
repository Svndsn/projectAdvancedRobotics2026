

from pathlib import Path
import pickle
from scripts.config import T_LIFT_INIT, T_FOLLOW_INIT, T_W_BOARD_CORNER
from utils.rtb import make_tf
from .robot import Robot
import spatialmath as sm
from .utils import sm_print

def main() -> None:
    robot = Robot()

    # traj_path = Path("/home/vims/git/mj_wire_mani/data/paths")

    # prim_traj = 0

    # with open(traj_path / "PRIM-2.pickle", "rb") as f:
    #     prim_traj = pickle.load(f)

    # pct = 1.0

    # T_base_target_traj = [robot.T_w_base.inv() @ Ti for Ti in prim_traj[:int(pct*len(prim_traj))]]

    T_w_target = make_tf(pos=T_W_BOARD_CORNER.t, ori = robot.T_base_tcp.R)

    robot.moveL( sm.SE3.Tz(0.2) @ robot.T_w_base.inv() @ T_w_target)

    # robot.moveL(T_base_target_traj[0])

    # key = input("continue?")
    # if key.lower() == "n":
    #     quit()

    # for Ti in T_base_target_traj:
    #     robot.servoL(Ti)


    # while True:

    #     robot.hande.move_and_wait_for_pos(0,pct=True)

    #     key = input("continue?")
    #     if key.lower() == "n":
    #         quit()

    #     robot.moveJ_IK(T_FOLLOW_INIT)
        
    #     key = input("continue?")
    #     if key.lower() == "n":
    #         quit()

    #     robot.hande.move_and_wait_for_pos(0.93)

    #     key = input("continue?")
    #     if key.lower() == "n":
    #         quit()

    #     T_w_target = sm.SE3.Tz(0.3) @ sm.SE3.Tx(0.3) @ robot.T_w_tcp

    #     T_base_target = robot.T_w_base.inv() @ T_w_target

    #     robot.moveL(T_base_target)

    #     key = input("continue?")
    #     if key.lower() == "n":
    #         quit()

    # robot.moveL(robot.T_base_tcp @ sm.SE3.Tz(0.1))

    
    # sm_print(robot.T_base_tcp)

if __name__ == "__main__":
    main()