import argparse
import numpy as np
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from src.sim.wrapper_env import WrapperEnvConfig, WrapperEnv
import os
import cv2
from pyapriltags import Detector
import algo 
from algo import pose7d_to_T,T_to_pose7d


'''
世界坐标系中：桌子的位置 xyz 都为正
所以对机器人来说：x 向前；y 向左，z 向下
'''

R_align = np.array([
    [ -1, 0,  0],  # x_apriltag → -x_env
    [ 0,  0, 1],  # y_apriltag → z_env
    [ 0,  1,  0],  # z_apriltag → +x_env
])

T_align = np.eye(4)
T_align[:3, :3] = R_align

np.set_printoptions(precision=4, suppress=True)

def detect_pose(img, camera_matrix, camera_pose):
    """
    Detects the pose of driller
    """
    # implement the detection logic here
    # 
    pose = np.eye(4)
    return pose

def dummy_policy(obs):
    """
    A dummy policy that does nothing.
    You may need to modify this function to implement your own policy.
    """
    # implement the policy logic here
    #
    release_flag = 0
    humanoid_action = np.zeros(8)

    obj_pose = detect_pose(...)
    
    # if release set release flag is 1

    return humanoid_action, release_flag



def main():
    # implement your policy here
    parser = argparse.ArgumentParser(description="Launcher config - Physics")
    parser.add_argument("--robot", type=str, default="galbot")
    parser.add_argument("--obj", type=str, default="power_drill")
    parser.add_argument("--ctrl_dt", type=float, default=0.02)
    parser.add_argument("--headless", type=int, default=1)
    parser.add_argument("--reset_wait_steps", type=int, default=100)

    args = parser.parse_args()

    env_config = WrapperEnvConfig(
        humanoid_robot=args.robot,
        obj_name=args.obj,
        headless=args.headless,
        ctrl_dt=args.ctrl_dt,
        reset_wait_steps=args.reset_wait_steps,
    )

    env = WrapperEnv(env_config)
    env.launch()
    env.reset()

    head_init_qpos = np.array([-0.05, 0.35])  # [horizontal, vertical] 确保这个角度能看到tag
    humanoid_init_qpos = env.sim.humanoid_robot_cfg.joint_init_qpos

    env.step_env(
        humanoid_head_qpos=head_init_qpos, 
        humanoid_action=humanoid_init_qpos[:8],
        quad_command=[0,0,0] # 初始让四足机器人静止，方便检测
    )
    apriltag_detector = Detector(
        families="tagStandard52h13", # 确保这个家族与你场景中的AprilTag一致
        nthreads=1,
        quad_decimate=1.0,
        quad_sigma=0.0,
        refine_edges=1,
        decode_sharpening=0.25,
        debug=0
    )
    # 初始获取一次观测并尝试检测，以便调试头部初始姿态
    obs_head_initial = env.get_obs(camera_id=0)
    env.debug_save_obs(obs_head_initial, 'data/obs_head_initial_setup') 

    Metric = {
        'obj_pose': False,
        'drop_precision': False,
        'quad_return': False,
    }
    trans, rot = env.humanoid_robot_model.fk_link(humanoid_init_qpos, env.humanoid_robot_cfg.link_eef)
    succ, qpos = env.humanoid_robot_model.ik(trans=trans, rot=rot)
    if succ:
        print("IK success")
        # print("qpos:", qpos) # qpos很长，可以按需打印


    '''
    parameters
    '''
    total_steps = 1 # number of steps to run the simulation, you can change this according to your needs
    #steps_per_camera_shot = 5 # number of steps per camera shot, increase this to reduce the frequency of camera shots and speed up the simulation
    #humanoid_init_qpos = env.sim.humanoid_robot_cfg.joint_init_qpos
    humanoid_curr_qpos = humanoid_init_qpos[:8].copy()
    tag_physical_size = 0.12
    target_pos = np.array([0.45, -0.06])   # 目标点
    arrive_thresh = 0.05   # 认为到达的距离阈值
    max_speed = 0.3        # 最大速度
    head_init_qpos = np.array([-0.35, 0.35])
    quad_command = np.array([2, -0.1 ,0]) # quad robot move forward
    target_back=np.array([1.761, -0.216]) # 目标点的世界坐标系位置
    initial_location=np.array([0,0]) # 初始位置
    flag=0


    '''
    Priviledge infomation in the environment: You can't use them directly, only used to debug and evaluation
    '''
    container_dog = np.array([-0.09, 0, 0.115, 0, 0, 0, 1])
    # tag_pose_dog = np.array([-0.22, 0, 0.09, 0.7071, 0.7071, 0, 0])
    tag_pose_dog = np.array([-0.22, 0, 0.09, 0, 0, 0, 1])
    dog_world = env.sim.mj_data.qpos[:7]
    real_tag_pose = algo.from_dog_frame_to_world(dog_pose=dog_world,obj_local=tag_pose_dog)
    real_tag_pose = algo.pose7d_to_T(real_tag_pose)
    real_container_pose = algo.from_dog_frame_to_world(dog_pose=dog_world,obj_local=container_dog)

    '''
    Variables need to update (4x4)
    '''
    pred_container_pose = np.eye(4)
    human_action = None



    for step in range(total_steps):
        obs = []
        '''
        if step % steps_per_camera_shot == 0:
            obs_head = env.get_obs(camera_id=0) # head camera
            obs_wrist = env.get_obs(camera_id=1) # wrist camera
            obs = [obs_head, obs_wrist]
        '''
        
        
        #state = env.get_state() # get the state (you can only get humanoid qpos here)
        #head_qpos, humanoid_action, quad_command = dummy_policy(step, obs, state)
        
        # delta_qpos = np.random.uniform(-0.01, 0.01, size=7) # humanoid robot random action
        # humanoid_curr_qpos[:7] += delta_qpos
        obs_head = env.get_obs(camera_id=0) # head camera
        obs_wrist = env.get_obs(camera_id=1) # wrist camera
        env.debug_save_obs(obs_head, 'data/obs_head') # obs has rgb, depth, and camera pose
        env.debug_save_obs(obs_wrist, 'data/obs_wrist')
        # test the metrics
        gray_image = cv2.cvtColor(obs_head.rgb, cv2.COLOR_RGB2GRAY)
        
        # 正确获取相机内参
        # 对于头部相机 (camera_id=0)
        head_intrinsic_matrix = env.humanoid_robot_cfg.camera_cfg[0].intrinsics
        fx = head_intrinsic_matrix[0, 0]
        fy = head_intrinsic_matrix[1, 1]
        cx = head_intrinsic_matrix[0, 2]
        cy = head_intrinsic_matrix[1, 2]
        correct_camera_params = (fx, fy, cx, cy)
        
        # 调用 detect 方法
        detections_list = apriltag_detector.detect(gray_image, 
                                             estimate_tag_pose=True, 
                                             camera_params=correct_camera_params, 
                                             tag_size=tag_physical_size) # 使用之前定义的 tag_physical_size
        
         
        print(f"Step {step}, Detected {len(detections_list)} tags with head camera.")
        if detections_list:
            for idx, tag_detection in enumerate(detections_list):
                print(f"  --- Tag {idx+1} (ID: {tag_detection.tag_id}) ---")
                # print(f"    Tag Family: {tag_detection.tag_family.decode()}")
                print(f"    Center (pixels): ({tag_detection.center[0]:.2f}, {tag_detection.center[1]:.2f})")
                
                if tag_detection.pose_R is not None and tag_detection.pose_t is not None:

                    # 计算并打印世界坐标系下的位姿
                    T_cam_tag = np.eye(4)
                    T_cam_tag[0:3, 0:3] = tag_detection.pose_R
                    # print("tag_detection.pose_R",tag_detection.pose_R)
                    T_cam_tag[0:3, 3] = tag_detection.pose_t[:, 0]
                    M = np.array([
                    [0, -1, 0],
                    [-1, 0, 0],
                    [0, 0, -1]
                    ])

# 转成4x4
                    A = np.eye(4)
                    A[:3, :3] = M

# 变换后的新坐标系
                    T_cam_tag = T_cam_tag @ A
                    inv_camera_pose = np.linalg.inv(obs_head.camera_pose)
                    print("T cam tag : ",T_cam_tag)
                    print("camera pose:", T_to_pose7d(obs_head.camera_pose))
                    T_world_tag = np.dot(obs_head.camera_pose, T_cam_tag) # obs_head.camera_pose 是 T_world_cam
                    tag_world_position = T_world_tag[:3, 3]

                    #  print("ground truth tag in cam : ",((inv_camera_pose@real_tag_pose)[:3,:3])@(np.linalg.inv(tag_detection.pose_R)))

                    print(f"    Tag Position in World (x,y,z): [{tag_world_position[0]:.3f}, {tag_world_position[1]:.3f}, {tag_world_position[2]:.3f}] m")
                else:
                    print("    3D Pose (R, t) not estimated for this tag.")

                # 最后需要注释掉
                tag_dog_T = algo.pose7d_to_T(tag_pose_dog) 
                # print(tag_dog_T)
                tag_pose_dog_inv = np.linalg.inv(tag_dog_T) # tag 坐标系下狗的坐标
                # print(tag_pose_dog_inv)
                pred_dog_pose = np.dot(T_world_tag, tag_pose_dog_inv) # 世界坐标系下狗的坐标
                # print(T_world_tag)
                con_dog_T = algo.pose7d_to_T(container_dog) # 狗的坐标系下的 container 的坐标
                pred_container_pose = np.dot(pred_dog_pose,con_dog_T)
                print(f"predicted Tag xyz {T_to_pose7d(T_world_tag)}---- \ngroundtruth Tag xyz {T_to_pose7d(real_tag_pose)}\n")
                print(f"predicted dog xyz {T_to_pose7d(pred_dog_pose)}---- groundtruth dog xyz {dog_world}")
                print(f"predicted container xyz {pred_container_pose[:3,3]}---- groundtruth container xyz {real_container_pose[:3]}")

        else:
            head_init_qpos[1]+=0.1 # 低头 TODO: 极限为 0.366 可能需要调整

        
        if initial_location[0] == 0 and initial_location[1] == 0:
            initial_location = tag_world_position[:2]
            print(f"Initial location set to: {initial_location}")
        distance_to_target = np.linalg.norm(target_pos - tag_world_position[:2])
        if distance_to_target < arrive_thresh and flag==0:
            print(f"Target position reached within threshold: {distance_to_target:.3f} m")
            quad_command = np.array([0, 0, 0])  # 停止四足机器人
            target_pos = initial_location # 更新目标位置为初始位置
            flag=1
        elif distance_to_target < arrive_thresh and flag==1:
            print(f"Target position reached within threshold: {distance_to_target:.3f} m")
            quad_command = np.array([0, 0, 0])
        else:
            vx= 3*np.clip(-(target_pos[0] - tag_world_position[0]) / 2, -max_speed, max_speed)
            vy= 3*np.clip(-(target_pos[1] - tag_world_position[1]) / 2, -max_speed, max_speed)

            if abs(vx) < 0.15 and abs(vy) < 0.15:
                vx=1* np.sign(vx) * 0.15
                vy=1* np.sign(vy) * 0.15
            quad_command = np.array([vx, vy, 0])  # 四足机器人移动到目标位置
            #quad_command = np.array([0.3, -0.1, 0])  # 四足机器人移动到目标位置    


        if flag == 1: # call our policy to grasp:
            _,human_action,_ = dummy_policy(obs=[obs_head,obs_wrist])



        env.step_env(
            humanoid_head_qpos=head_init_qpos,
            humanoid_action=humanoid_curr_qpos,
            quad_command=quad_command
        )
        # --- End AprilTag Detection Logic ---
        '''
        Metric["drop_precision"] = Metric["drop_precision"] or env.metric_drop_precision()
        Metric["quad_return"] = Metric["quad_return"] or env.metric_quad_return()
        '''
    print("Metrics:", Metric) 

    print("Simulation completed.")
    env.close()

if __name__ == "__main__":
    main()