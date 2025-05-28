import argparse
import numpy as np

from src.sim.wrapper_env import WrapperEnvConfig, WrapperEnv

def detect_pose(img, camera_matrix, camera_pose):
    """
    Detects the pose of driller
    """
    # TODO: implement the detection logic here
    # 
    pose = np.eye(4)
    return pose

def dummy_policy(step, obs, state):
    """
    A dummy policy that does nothing.
    You may need to modify this function to implement your own policy.
    """
    # TODO:implement the policy logic here
    #
    head_qpos = np.zeros(2)
    humanoid_action = np.zeros(8)
    quad_command = np.zeros(3) 



    # Whats the form of "quad_command"
    return head_qpos, humanoid_action, quad_command

def main():
    # implement your policy here
    parser = argparse.ArgumentParser(description="Launcher config - Physics")
    parser.add_argument("--robot", type=str, default="galbot")
    parser.add_argument("--obj", type=str, default="power_drill")
    parser.add_argument("--ctrl_dt", type=float, default=0.02)
    parser.add_argument("--headless", type=int, default=0)
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

    head_init_qpos = np.array([0.0, 0.0]) # you can adjust the head init qpos to find the driller

    # just turn the robort's head at x and y axis to find the driller
    env.step_env(
        humanoid_head_qpos=head_init_qpos,
        humanoid_action=env.sim.humanoid_robot_cfg.joint_init_qpos[:8],
        quad_command=[0, 0, 0]
    )

    Metric = {
        'obj_pose': False,
        'drop_precision': False,
        'quad_return': False,
    }

    obs_wrist = env.get_obs(camera_id=1) # wrist camera
    rgb, depth, camera_pose = obs_wrist.rgb, obs_wrist.depth, obs_wrist.camera_pose
    
    '''Detect the pose of driller: Part 1'''
    driller_pose = detect_pose(rgb, camera_pose[:3, :3], camera_pose[:3, 3])

    # test the pose detection
    Metric['obj_pose'] = env.metric_obj_pose(driller_pose)

    total_steps = 800 # number of steps to run the simulation, you can change this according to your needs
    steps_per_camera_shot = 5 # number of steps per camera shot, increase this to reduce the frequency of camera shots and speed up the simulation
    
    for step in range(total_steps):
        obs = []
        if step % steps_per_camera_shot == 0:
            obs_head = env.get_obs(camera_id=0) # head camera
            obs_wrist = env.get_obs(camera_id=1) # wrist camera
            obs = [obs_head, obs_wrist]
        state = env.get_state() # get the state (you can only get humanoid qpos here)

        '''Apply our policy to get humanoid_head_qpos, humanoid_action, quad_command: Part 2'''
        head_qpos, humanoid_action, quad_command = dummy_policy(step, obs, state)

        env.step_env(
            humanoid_head_qpos=head_qpos,
            humanoid_action=humanoid_action,
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