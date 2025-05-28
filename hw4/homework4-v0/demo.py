import argparse
import numpy as np
import cv2
from pyapriltags import Detector
import os
from src.utils import to_pose
from src.sim.wrapper_env import WrapperEnvConfig, WrapperEnv

from scipy.spatial.transform import Rotation as R

'''临时设置'''
np.set_printoptions(precision=4, suppress=True)

'''全部使用 x,y,z,w 代表 quaternion'''

def detect_AT_pose(detector, rgb_img, camera_matrix, tag_size=0.12):
    gray_image = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2GRAY)
        
    # 正确获取相机内参
    # 对于头部相机 (camera_id=0)
    head_intrinsic_matrix = camera_matrix
    fx = head_intrinsic_matrix[0, 0]
    fy = head_intrinsic_matrix[1, 1]
    cx = head_intrinsic_matrix[0, 2]
    cy = head_intrinsic_matrix[1, 2]
    correct_camera_params = (fx, fy, cx, cy)
    
    # 调用 detect 方法
    detections_list = detector.detect(gray_image, 
                                            estimate_tag_pose=True, 
                                            camera_params=correct_camera_params, 
                                            tag_size=tag_size) 
    # 使用之前定义的 tag_physical_size
    if detections_list is None:
        raise("No AprilTag detected!")
    tag_detection = detections_list[0]
    if tag_detection.pose_R is not None and tag_detection.pose_t is not None:
        T_cam_tag = np.eye(4)
        T_cam_tag[0:3, 0:3] = tag_detection.pose_R
        T_cam_tag[0:3, 3] = tag_detection.pose_t[:, 0]
        return T_cam_tag
            # T_world_tag = np.dot(obs_head.camera_pose, T_cam_tag) # obs_head.camera_pose 是 T_world_cam
            # tag_world_position = T_world_tag[:3, 3]

def pose7d_to_mat(pose7d):
    t = np.array(pose7d[:3])
    q = np.array(pose7d[3:7])  # 输入四元数 [x,y,z,w]

    rot = R.from_quat(q).as_matrix()  # scipy默认四元数格式是 [x,y,z,w]
    T = np.eye(4)
    T[:3, :3] = rot
    T[:3, 3] = t
    return T

def mat_to_pose7d(T):
    t = T[:3, 3]
    rot_mat = T[:3, :3]
    q = R.from_matrix(rot_mat).as_quat()  # 输出 [x,y,z,w]
    return np.concatenate([t, q])

def pred_dog_pose_in_world(tag_pose_world, tag_pose_dog):
    """
    计算狗在世界坐标系下的位姿:
    输入tag_pose_world和tag_pose_dog均为7D，格式：
    [x,y,z, qx, qy, qz, qw]
    """
    tag_pose_world_T = pose7d_to_mat(tag_pose_world)
    tag_pose_dog_T = pose7d_to_mat(tag_pose_dog)

    tag_pose_dog_inv = np.linalg.inv(tag_pose_dog_T)
    dog_pose_world = np.dot(tag_pose_world_T, tag_pose_dog_inv)

    return mat_to_pose7d(dog_pose_world)

def from_dog_frame_to_world(dog_pose, obj_local):
    """
    将物体在狗坐标系下的7D位姿转换到世界坐标系。
    dog_pose和obj_local均为7D格式：[x,y,z, qx,qy,qz,qw]
    """
    p_dog = dog_pose[:3]
    q_dog = dog_pose[3:]  # [x,y,z,w]
    p_obj_local = obj_local[:3]
    q_obj_local = obj_local[3:]  # [x,y,z,w]
    r_dog = R.from_quat(q_dog)
    r_obj_local = R.from_quat(q_obj_local)

    # 位置变换：将obj_local的位置旋转到世界坐标系，再加上狗的位置
    p_world = p_dog + r_dog.apply(p_obj_local)

    # 姿态变换：世界坐标系下的旋转 = 狗的旋转 * 物体在狗坐标系的旋转
    r_world = r_dog * r_obj_local
    q_world = r_world.as_quat()  # [x,y,z,w]
    return np.concatenate([p_world, q_world])


def demo_sim():
    # nearly all the functions of the simulation is implemeted in this demo
    parser = argparse.ArgumentParser(description="Launcher config - Physics")
    parser.add_argument("--robot", type=str, default="galbot")
    parser.add_argument("--obj", type=str, default="power_drill")
    parser.add_argument("--ctrl_dt", type=float, default=0.02)
    parser.add_argument("--headless", type=int, default=1)
    parser.add_argument("--customize_scene", type=int, default=0)
    parser.add_argument("--reset_wait_steps", type=int, default=100)
    

    args = parser.parse_args()
    
    # april tag is used for detection, tag used in the simulation has size of 0.12
    apriltag_detector = Detector(
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
    if args.customize_scene:
        # print(object)
        # Customize the table and object for testing with random environment
        table_pose = to_pose(trans=np.array([0.6, 0.35, 0.72]))
        table_size = np.array([0.68, 0.36, 0.02])
        # 如果将物体的位置设置为箱子的位置，就可以在箱子里面,例如 [0.6807103  -0.20000001  0.24536407]
        # 
        # env.sim.mj_data.qpos[:3] = np.array([0.65,-0.2,0.278])
        obj_trans = np.array([0.5, 0.3, 0.82])
        # print(obj_trans)
        obj_rot = np.eye(3)
        obj_pose = to_pose(obj_trans, obj_rot)
        env.set_table_obj_config(
            table_pose=table_pose,
            table_size=table_size,
            obj_pose=obj_pose
        )

    env.launch()
    env.reset() 
    
    # 这里初始化是没有用的 env.sim.mj_data.qpos[:3] = np.array([0.65,-0.2,0.278])


    ''' 
    Calculate the groundtruth in the world frame 
    obj on the dog, xyz in dog frame:
    -------------------------------
    container: -0.09, 0, 0.115
    AprilTag: 0.22, 0, 0.09
    '''
    container_dog = np.array([0.09, 0, 0.115, 0, 0, 0, 1])
    tag_pose_dog = np.array([-0.22, 0, 0.09, 0, 0, 0, 1])
    dog_world = env.sim.mj_data.qpos[:7]
    # groundtruth of the container pose
    container_pose=from_dog_frame_to_world(dog_pose=dog_world,obj_local=container_dog)
    tag_pose = from_dog_frame_to_world(dog_pose=dog_world,obj_local=tag_pose_dog)
    # groundtruth of the AprilTag pose
    print(f"Groundtruth pose in the world frame:\n Dog: {dog_world}\ncontainer pose (3D): {container_pose}\nAprilTag 7D pose: {tag_pose[:3]}")

    ''' Heap qpos
    default: np.array([-0.05, 0.35])
    horizontal:[-1.57,1.57], negative -> turn right
    vertical: [-0.366,0.366], negative -> up
    '''
    
    # head_init_qpos = np.array([-0.1, 0.3])  # [horizontal, vertical]
    head_init_qpos = np.array([0., 0.36])


    humanoid_init_qpos = env.sim.humanoid_robot_cfg.joint_init_qpos

    env.step_env(
        humanoid_head_qpos=head_init_qpos, # head joint qpos is for adjusting the camera pose
        humanoid_action=humanoid_init_qpos[:8],
        quad_command=[0,0,0]
    )
    obs_head = env.get_obs(camera_id=0) # head camera
    obs_wrist = env.get_obs(camera_id=1) # wrist camera



    env.debug_save_obs(obs_head, 'data/obs_head') # obs has rgb, depth, and camera pose
    env.debug_save_obs(obs_wrist, 'data/obs_wrist')


    '''
    Predict the AprilTag with Detector:
    ----------------------------
    进一步需要知道 Tag 的大小: 0.24
    相机的内参： camera_id = 0 for head camera
    pred -> AprilTag in the world.
    calculate: container pose in the world
    '''
    tag_size = 0.12
    pred_tag_pose_cam = detect_AT_pose(apriltag_detector,rgb_img=obs_head.rgb,camera_matrix=env.humanoid_robot_cfg.camera_cfg[0].intrinsics,tag_size=tag_size)
    camera_pose = obs_head.camera_pose

    p_cam = camera_pose @ np.array([0,0,0,1])

    print("p_cam: ",p_cam)
    print("Tag in the camera frame is: ",pred_tag_pose_cam)
    pred_tag_pose_world = np.dot(camera_pose, pred_tag_pose_cam)
    print(f"camera pose {camera_pose}\nPredicted pose in the world frame:\nTag in the world(decect): {pred_tag_pose_world[:3,3]}")
    ''''''


    '''暂时设置: 之后需要更改为 实际预测的 Tag pose''' 
    pred_tag_pose_world = tag_pose
    # print("\n\npredict tag pose is : ",pred_tag_pose_world[:3])
    pred_dog_pose = pred_dog_pose_in_world(tag_pose_world=pred_tag_pose_world,tag_pose_dog=tag_pose_dog)
    # print("predict dog pose is : ",pred_dog_pose)
    pred_contrainer_pose = from_dog_frame_to_world(dog_pose=pred_dog_pose,obj_local=container_dog)

    # print(f"Predicted container pose is : xyz:{pred_contrainer_pose[:3]}")

    trans, rot = env.humanoid_robot_model.fk_link(humanoid_init_qpos, env.humanoid_robot_cfg.link_eef)
    succ, qpos = env.humanoid_robot_model.ik(trans=trans, rot=rot)
    if succ:
        print("IK success")
        print("qpos:", qpos)
    
    env.sim.debug_vis_pose(to_pose(trans, rot)) # visualize the pose of the end effector
    for i in range(50):
        env.step_env(
            humanoid_head_qpos=head_init_qpos,
            humanoid_action=humanoid_init_qpos[:8],
            quad_command=[0,0,0]
        )

    humanoid_curr_qpos = humanoid_init_qpos[:8].copy()
    for i in range(100):
        quad_command = np.array([0.0, -0.0 ,0.3]) # quad robot move forward
        delta_qpos = np.random.uniform(-0.01, 0.01, size=7) # humanoid robot random action
        
        
        # Use delta_qpos to control the humanoid robot
        humanoid_curr_qpos[:7] += delta_qpos
        env.step_env(
            humanoid_head_qpos=head_init_qpos,
            humanoid_action=humanoid_curr_qpos,
            quad_command=quad_command
        )
    
    print("Simulation completed.")
    env.close()

if __name__ == "__main__":
    demo_sim()