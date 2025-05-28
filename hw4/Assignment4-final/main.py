import argparse
from typing import Optional, Tuple, List
import numpy as np
import cv2
from pyapriltags import Detector

from src.type import Grasp
from src.utils import to_pose
from src.sim.wrapper_env import WrapperEnvConfig, WrapperEnv
from src.sim.wrapper_env import get_grasps
from src.test.load_test import load_test_data
import algo




'''
我们现在急需确定，如果狗不转身，靠近机器人后，箱子的距离机器人能不能够得着，如果不行，我们可能需要调整控制狗的策略
目前还有一个问题，头低到最低，在狗接近时也看不到 Tag，我们不放让狗先转 180度，然后倒着靠近。
'''



np.set_printoptions(precision=4, suppress=True)







def detect_driller_pose(img, depth, camera_matrix, camera_pose, *args, **kwargs):
    """
    Detects the pose of driller, you can include your policy in args
    """
    # implement the detection logic here
    # 
    
    pose = np.eye(4)
    return pose

def detect_marker_pose(
        detector: Detector, 
        img: np.ndarray, 
        camera_params: tuple, 
        camera_pose: np.ndarray,
        tag_size: float = 0.12
    ) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    # implement
    trans_marker_world = None
    rot_marker_world = None
    gray_image = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    detections_list = detector.detect(gray_image, 
                                    estimate_tag_pose=True, 
                                    camera_params=camera_params, 
                                    tag_size=tag_size) # 使用之前定义的 tag_physical_size

    if len(detections_list) == 0:
        return None,None
    else:
        tag_detection=detections_list[0]
        T_cam_tag = np.eye(4)
        T_cam_tag[0:3, 0:3] = tag_detection.pose_R
        # print("tag_detection.pose_R",tag_detection.pose_R)
        T_cam_tag[0:3, 3] = tag_detection.pose_t[:, 0]
        # inv_camera_pose = np.linalg.inv(obs_head.camera_pose)
        # print("T cam tag : ",T_cam_tag)
        # print("camera pose:", T_to_pose7d(obs_head.camera_pose))
        T_world_tag = np.dot(camera_pose, T_cam_tag) 
        trans_marker_world = T_world_tag[:3,3]
        rot_marker_world = T_world_tag[:3,:3]

    return trans_marker_world, rot_marker_world

def forward_quad_policy(pose, target_pose, *args, **kwargs):
    """ guide the quadruped to position where you drop the driller """
    # implement
    action = np.array([0,0,0])
    return action

def backward_quad_policy(pose, target_pose, *args, **kwargs):
    """ guide the quadruped back to its initial position """
    # implement
    action = np.array([0,0,0])
    return action

def plan_grasp(env: WrapperEnv, grasp: Grasp, grasp_config, *args, **kwargs) -> Optional[List[np.ndarray]]:
    """Try to plan a grasp trajectory for the given grasp. The trajectory is a list of joint positions. Return None if the trajectory is not valid."""
    # implement
    reach_steps = grasp_config['reach_steps']
    lift_steps = grasp_config['lift_steps']
    delta_dist = grasp_config['delta_dist']

    traj_reach = []
    traj_lift = []
    succ = False
    if not succ: return None

    return [np.array(traj_reach), np.array(traj_lift)]

def plan_move(env: WrapperEnv, begin_qpos, begin_trans, begin_rot, end_trans, end_rot, steps = 50, *args, **kwargs):
    """Plan a trajectory moving the driller from table to dropping position"""
    # implement
    traj = []

    succ = False
    if not succ: return None
    return traj

def open_gripper(env: WrapperEnv, steps = 10):
    for _ in range(steps):
        env.step_env(gripper_open=1)
def close_gripper(env: WrapperEnv, steps = 10):
    for _ in range(steps):
        env.step_env(gripper_open=0)
def plan_move_qpos(begin_qpos, end_qpos, steps=50) -> np.ndarray:
    delta_qpos = (end_qpos - begin_qpos) / steps
    cur_qpos = begin_qpos.copy()
    traj = []
    
    for _ in range(steps):
        cur_qpos += delta_qpos
        traj.append(cur_qpos.copy())
    
    return np.array(traj)
def execute_plan(env: WrapperEnv, plan):
    """Execute the plan in the environment."""
    for step in range(len(plan)):
        env.step_env(
            humanoid_action=plan[step],
        )


def head_move_policy(tag_world,current_head_pose,camera_pose):
    #TODO:
    return [0,0]

TESTING = True
DISABLE_GRASP = False
DISABLE_MOVE = False

def main():
    parser = argparse.ArgumentParser(description="Launcher config - Physics")
    parser.add_argument("--robot", type=str, default="galbot")
    parser.add_argument("--obj", type=str, default="power_drill")
    parser.add_argument("--ctrl_dt", type=float, default=0.02)
    parser.add_argument("--headless", type=int, default=1) # 暂时不显式
    parser.add_argument("--reset_wait_steps", type=int, default=100)
    parser.add_argument("--test_id", type=int, default=0)

    args = parser.parse_args()

    detector = Detector(
        families="tagStandard52h13",
        nthreads=1,
        quad_decimate=1.0,
        quad_sigma=0.0,
        refine_edges=1,
        decode_sharpening=0.25,
        debug=0
    )

    env_config = WrapperEnvConfig(
        humanoid_robot=args.robot,
        obj_name=args.obj,
        headless=args.headless,
        ctrl_dt=args.ctrl_dt,
        reset_wait_steps=args.reset_wait_steps,
    )


    env = WrapperEnv(env_config)






    if TESTING:
        data_dict = load_test_data(args.test_id)
        env.set_table_obj_config(
            table_pose=data_dict['table_pose'],
            table_size=data_dict['table_size'],
            obj_pose=data_dict['obj_pose']
        )
        env.set_quad_reset_pos(data_dict['quad_reset_pos'])

    env.launch()
    env.reset(humanoid_qpos=env.sim.humanoid_robot_cfg.joint_init_qpos)
    humanoid_init_qpos = env.sim.humanoid_robot_cfg.joint_init_qpos[:7]
    Metric = {
        'obj_pose': False,
        'drop_precision': False,
        'quad_return': False,
    }
    
    head_init_qpos = np.array([0.0,0.0]) # you can adjust the head init qpos to find the driller

    env.step_env(humanoid_head_qpos=head_init_qpos)
    
    observing_qpos = humanoid_init_qpos + np.array([0.01,0,0,0,0,0,0]) # you can customize observing qpos to get wrist obs
    # init_plan = plan_move_qpos(env, humanoid_init_qpos, observing_qpos)
    # execute_plan(env, init_plan)

    '''
    Priviledge infomation in the environment: You can't use them directly, only used to debug and evaluation
    '''
    container_dog = np.array([0.09, 0, 0.115, 0, 0, 0, 1])
    # tag_pose_dog = np.array([-0.22, 0, 0.09, 0.7071, 0.7071, 0, 0])
    tag_pose_dog = np.array([-0.22, 0, 0.09, 0, 0, 0, 1])
    dog_world = env.sim.mj_data.qpos[:7]
    real_tag_pose = algo.from_dog_frame_to_world(dog_pose=dog_world,obj_local=tag_pose_dog)
    # real_tag_pose = algo.pose7d_to_T(real_tag_pose)
    real_container_pose = algo.from_dog_frame_to_world(dog_pose=dog_world,obj_local=container_dog)
    # real_container_pose = algo.pose7d_to_T(real_container_pose)
    print("Dog pose in the world is : ",dog_world)


    obs_head = env.get_obs(camera_id=0) # head camera
    obs_wrist = env.get_obs(camera_id=1) # wrist camera

    env.debug_save_obs(obs_head, 'data/obs_head') # obs has rgb, depth, and camera pose
    env.debug_save_obs(obs_wrist, 'data/obs_wrist')
    # --------------------------------------step 1: move quadruped to dropping position--------------------------------------
    if not DISABLE_MOVE:
        forward_steps = 1000 # number of steps that quadruped walk to dropping position
        steps_per_camera_shot = 5 # number of steps per camera shot, increase this to reduce the frequency of camera shots and speed up the simulation
        head_camera_matrix = env.sim.humanoid_robot_cfg.camera_cfg[0].intrinsics
        head_camera_params = (head_camera_matrix[0, 0],head_camera_matrix[1, 1],head_camera_matrix[0, 2],head_camera_matrix[1, 2])

        # implement this to guide the quadruped to target dropping position
        #
        target_container_pose = [] 
        def is_close(pose1, pose2, threshold): # Only judge x,y,z
            # pos_1 = pose1[:3,3]
            # pos_2 = pose2[:3,3]
            # if np.norm()
            # TODO:
            return True
        for step in range(forward_steps):
            move_head = False
            if step % steps_per_camera_shot == 0:
                obs_head = env.get_obs(camera_id=0) # head camera
                trans_marker_world, rot_marker_world = detect_marker_pose(
                    detector, 
                    obs_head.rgb, 
                    head_camera_params,
                    obs_head.camera_pose,
                    tag_size=0.12
                )
                
                

                if trans_marker_world is not None: # in the world 
                    # the container's pose is given by follows:
                    trans_container_world = rot_marker_world @ np.array([0,0.31,0.02]) + trans_marker_world
                    rot_container_world = rot_marker_world
                    pose_container_world = to_pose(trans_container_world, rot_container_world)
                    
                    '''Check if detect mark in the world is true'''
                    tag_world = to_pose(trans_marker_world,rot_marker_world)
                    print(f"   pred Tag in the world {algo.T_to_pose7d(tag_world)}, groudtruth Tag in the world: {real_tag_pose}")
                    print(f"   pred Container in the world {algo.T_to_pose7d(pose_container_world)}, groudtruth container in the world: {real_container_pose}")
                else: # No AprilTag detected: Try to change human head pose
                    print("No AprilTag detected: Try to change human head pose")
                    move_head = True

            # 根据 AprilTag 先确定好狗的位置
            # quad_command = forward_quad_policy(pose_container_world, target_container_pose)
            if move_head:
                # if you need moving head to track the marker, implement this
                head_qpos = [0,0]
                quad_command = [0. ,0.,0.]
                '''也许需要动态的调整'''
                head_qpos = head_move_policy(tag_world=tag_world,current_head_pose=obs_head.camera_pose)
                env.step_env(
                    humanoid_head_qpos=head_qpos,
                    quad_command=quad_command
                )
            else:
                quad_command = forward_quad_policy(pose_container_world, target_container_pose)
                env.step_env(quad_command=quad_command)
            if is_close(pose_container_world, target_container_pose, threshold=0.05): break


    # --------------------------------------step 2: detect driller pose------------------------------------------------------
    if not DISABLE_GRASP:
        obs_wrist = env.get_obs(camera_id=1) # wrist camera
        rgb, depth, camera_pose = obs_wrist.rgb, obs_wrist.depth, obs_wrist.camera_pose
        wrist_camera_matrix = env.sim.humanoid_robot_cfg.camera_cfg[1].intrinsics
        driller_pose = detect_driller_pose(rgb, depth, wrist_camera_matrix, camera_pose[:3, 3])
        # metric judgement
        Metric['obj_pose'] = env.metric_obj_pose(driller_pose)


    # --------------------------------------step 3: plan grasp and lift------------------------------------------------------
    if not DISABLE_GRASP:
        obj_pose = driller_pose.copy()
        grasps = get_grasps(args.obj) 
        grasps0_n = Grasp(grasps[0].trans, grasps[0].rot @ np.diag([-1,-1,1]), grasps[0].width)
        grasps2_n = Grasp(grasps[2].trans, grasps[2].rot @ np.diag([-1,-1,1]), grasps[2].width)
        valid_grasps = [grasps[0], grasps0_n, grasps[2], grasps2_n] # we have provided some grasps, you can choose to use them or yours
        grasp_config = dict( 
            reach_steps=0,
            lift_steps=0,
            delta_dist=0, 
        ) # the grasping design in assignment 2, you can choose to use it or design yours

        for obj_frame_grasp in valid_grasps:
            robot_frame_grasp = Grasp(
                trans=obj_pose[:3, :3] @ obj_frame_grasp.trans
                + obj_pose[:3, 3],
                rot=obj_pose[:3, :3] @ obj_frame_grasp.rot,
                width=obj_frame_grasp.width,
            )
            grasp_plan = plan_grasp(env, robot_frame_grasp, grasp_config)
            if grasp_plan is not None:
                break
        if grasp_plan is None:
            print("No valid grasp plan found.")
            env.close()
            return
        reach_plan, lift_plan = grasp_plan

        pregrasp_plan = plan_move_qpos(env, observing_qpos, reach_plan[0], steps=50) # pregrasp, change if you want
        execute_plan(env, pregrasp_plan)
        open_gripper(env)
        execute_plan(env, reach_plan)
        close_gripper(env)
        execute_plan(env, lift_plan)


    # --------------------------------------step 4: plan to move and drop----------------------------------------------------
    if not DISABLE_GRASP and not DISABLE_MOVE:
        # implement your moving plan
        #
        move_plan = plan_move(
            env=env,
        ) 
        execute_plan(env, move_plan)
        open_gripper(env)


    # --------------------------------------step 5: move quadruped backward to initial position------------------------------
    if not DISABLE_MOVE:
        # implement
        #
        backward_steps = 1000 # customize by yourselves
        for step in range(backward_steps):
            # same as before, please implement this
            #
            quad_command = backward_quad_policy()
            env.step_env(
                quad_command=quad_command
            )
        

    # test the metrics
    Metric["drop_precision"] = Metric["drop_precision"] or env.metric_drop_precision()
    Metric["quad_return"] = Metric["quad_return"] or env.metric_quad_return()

    print("Metrics:", Metric) 

    print("Simulation completed.")
    env.close()

if __name__ == "__main__":
    main()