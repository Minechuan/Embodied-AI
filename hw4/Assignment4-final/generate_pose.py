import os
import numpy as np
import cv2
import shutil
import random
from pathlib import Path
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
from src.sim.wrapper_env import WrapperEnvConfig, WrapperEnv
from src.utils import get_pc_from_rgbd
from src.utils import to_pose
from transforms3d.quaternions import quat2mat,mat2quat
from src.constants import OBJ_INIT_TRANS, OBJ_RAND_RANGE, TABLE_HEIGHT,DEPTH_IMG_SCALE

def randomize_object_pose(env):
    try:
        # 以OBJ_INIT_TRANS为中心，OBJ_RAND_RANGE为边长的正方形区域
        x = np.random.uniform(
            OBJ_INIT_TRANS[0] - OBJ_RAND_RANGE / 2,
            OBJ_INIT_TRANS[0] + OBJ_RAND_RANGE / 2
        )
        y = np.random.uniform(
            OBJ_INIT_TRANS[1] - OBJ_RAND_RANGE / 2,
            OBJ_INIT_TRANS[1] + OBJ_RAND_RANGE / 2
        )
        # z略高于桌面
        z = TABLE_HEIGHT + 0.1 + np.random.uniform(-0.01, 0.01)
        yaw = np.random.uniform(-np.pi, np.pi)
        trans = np.array([x, y, z])
        rot_z = np.array([
            [np.cos(yaw), -np.sin(yaw), 0],
            [np.sin(yaw),  np.cos(yaw), 0],
            [0, 0, 1]
        ])
        quat = mat2quat(rot_z)
        new_qpose = np.concatenate([trans, quat])
        if hasattr(env.sim, 'set_driller_pose'):
            env.sim.set_driller_pose(new_qpose)
        elif hasattr(env.sim, '_CombinedSim__set_driller_pose'):
            env.sim._CombinedSim__set_driller_pose(new_qpose)
        return True
    except Exception as e:
        print(f"Warning: Failed to randomize object pose: {e}")
        return False
    

def generate_object_segmentation(rgb, depth, object_pose, camera_pose, camera_intrinsics):
    """
    生成物体分割掩码 (简单版本)
    
    Parameters:
    -----------
    rgb : np.ndarray
        RGB图像 (H, W, 3)
    depth : np.ndarray
        深度图像 (H, W)
    object_pose : np.ndarray
        物体在世界坐标系的位姿 (4, 4)
    camera_pose : np.ndarray
        相机在世界坐标系的位姿 (4, 4)
    camera_intrinsics : np.ndarray
        相机内参矩阵 (3, 3)
        
    Returns:
    --------
    seg_mask : np.ndarray
        分割掩码 (H, W), 0为背景，255为物体
    """
    try:
        # 方法1: 基于深度和颜色的启发式分割
        h, w = depth.shape
        seg_mask = np.zeros((h, w), dtype=np.uint8)
        
        # 过滤有效深度值
        valid_depth = (depth > 0.1) & (depth < 3.0)
        
        if not np.any(valid_depth):
            return seg_mask
        
        # 基于深度的粗略分割
        # 假设桌面高度约0.8m，物体在桌面上方
        table_height = 0.8
        object_height_range = [table_height, table_height + 0.3]  # 物体高度范围
        
        # 将深度图转换为世界坐标系中的Z值（高度）
        # 这里需要根据相机姿态进行坐标变换
        camera_height = camera_pose[2, 3]  # 相机在世界坐标系中的高度
        # 简化版本：基于深度值的阈值分割
        # 选择距离相机适中的区域作为可能的物体区域
        object_depth_range = [0.3, 1.5]  # 物体可能的深度范围
        depth_mask = (depth >= object_depth_range[0]) & (depth <= object_depth_range[1])
        
        # 基于颜色的筛选（排除背景颜色）
        # 假设背景主要是桌面（褐色/木色）和墙面（白色/灰色）
        rgb_float = rgb.astype(np.float32) / 255.0
        
        # 计算颜色特征
        brightness = np.mean(rgb_float, axis=2)
        color_variance = np.var(rgb_float, axis=2)
        
        # 排除过亮（墙面）和过暗（阴影）的区域
        brightness_mask = (brightness > 0.1) & (brightness < 0.8)
        
        # 排除颜色单调的区域（可能是背景）
        variance_mask = color_variance > 0.01
        
        # 组合所有mask
        combined_mask = valid_depth & depth_mask & brightness_mask & variance_mask
        
        # 形态学操作去噪
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        combined_mask = cv2.morphologyEx(combined_mask.astype(np.uint8), cv2.MORPH_OPEN, kernel)
        combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel)
        
        # 连通域分析，保留最大的连通区域
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(combined_mask, connectivity=8)
        
        if num_labels > 1:  # 除了背景外还有其他连通域
            # 找到最大的连通域（除了背景）
            largest_component = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
            seg_mask = (labels == largest_component).astype(np.uint8) * 255
        else:
            seg_mask = combined_mask * 255
            
        return seg_mask
        
    except Exception as e:
        print(f"Warning: Failed to generate segmentation mask: {e}")
        # 返回空mask
        return np.zeros((rgb.shape[0], rgb.shape[1]), dtype=np.uint8)

def generate_episode_data(args):
    """单个episode的数据生成函数"""
    episode_start, episode_end, save_dir, process_id = args
    
    env_config = WrapperEnvConfig(
        humanoid_robot="galbot",
        obj_name="power_drill",
        headless=1,
        ctrl_dt=0.02,
        reset_wait_steps=20,
    )
    
    env = WrapperEnv(env_config)
    env.launch()
    
    for episode in range(episode_start, episode_end):
        try:
            env.launch()
            env.reset()
            # randomize_object_pose(env)
            # 更大范围的头部姿态采样
            head_qpos = np.random.uniform([-0.7, -0.4], [0.7, 0.4], 2)
            env.step_env(
                humanoid_head_qpos=head_qpos,
                humanoid_action=env.sim.humanoid_robot_cfg.joint_init_qpos[:7],
                quad_command=[0, 0, 0]
            )
            
            # 减少稳定步数
            for _ in range(3):
                env.step_env(
                    humanoid_head_qpos=head_qpos,
                    humanoid_action=env.sim.humanoid_robot_cfg.joint_init_qpos[:7],
                    quad_command=[0, 0, 0]
                )
            
            # 获取多个视角的数据
            for view_idx in range(1):
                if view_idx > 0:
                    # 微调头部姿态
                    head_qpos += np.random.uniform(-0.1, 0.1, 2)
                    head_qpos = np.clip(head_qpos, [-0.7, -0.4], [0.7, 0.4])
                    env.step_env(
                        humanoid_head_qpos=head_qpos,
                        humanoid_action=env.sim.humanoid_robot_cfg.joint_init_qpos[:7],
                        quad_command=[0, 0, 0]
                    )
                
                obs_wrist = env.get_obs(camera_id=1)
                rgb, depth, camera_pose = obs_wrist.rgb, obs_wrist.depth, obs_wrist.camera_pose
                
                if rgb is None or depth is None:
                    continue
                
                # qpose = env.sim.get_driller_pose()
                # trans = qpose[:3]
                # quat = qpose[3:7]
                # rot = quat2mat(quat)
                # obj_pose = to_pose(trans, rot)
                obj_pose = env.get_driller_pose()
                
                # 获取相机内参
                try:
                    camera_intrinsics = env.humanoid_robot_cfg.camera_cfg[1].intrinsics
                except:
                    print(f"Warning: no camera intrinsics found, using default.")
                    camera_intrinsics = np.array([
                        [525.0, 0.0, 320.0],
                        [0.0, 525.0, 240.0], 
                        [0.0, 0.0, 1.0]
                    ])
                
                # 生成物体分割掩码
                # obj_seg = generate_object_segmentation(rgb, depth, obj_pose, camera_pose, camera_intrinsics)
                
                # 保存数据，每个视角单独保存
                episode_dir = os.path.join(save_dir, f"episode_{episode:06d}_view_{view_idx}")
                os.makedirs(episode_dir, exist_ok=True)
                
                # 保存为PNG格式
                # RGB: 转换为BGR并保存
                cv2.imwrite(os.path.join(episode_dir, "rgb.png"), 
                           cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
                
                depth_scaled = (np.clip(depth, 0, 4.0) * 16384).astype(np.uint16)  # 0~4米映射到0~65535
                depth_scaled[depth <= 0] = 0  # 无效深度设为0
                cv2.imwrite(os.path.join(episode_dir, "depth.png"), depth_scaled)

                # depth_vis.png
                if np.any(depth_scaled > 0):
                    dmin, dmax = depth_scaled[depth_scaled > 0].min(), depth_scaled[depth_scaled > 0].max()
                    depth_vis = np.zeros_like(depth_scaled, dtype=np.uint8)
                    depth_vis[depth_scaled > 0] = ((depth_scaled[depth_scaled > 0] - dmin) / (dmax - dmin) * 255).astype(np.uint8)
                    depth_vis = cv2.applyColorMap(depth_vis, cv2.COLORMAP_JET)
                    cv2.imwrite(os.path.join(episode_dir, "depth_vis.png"), depth_vis)
                else:
                    depth_vis = np.zeros_like(depth_scaled, dtype=np.uint8)
                    cv2.imwrite(os.path.join(episode_dir, "depth_vis.png"), depth_vis)
                
                # 物体分割掩码
                # cv2.imwrite(os.path.join(episode_dir, "obj_seg.png"), obj_seg)
               
                # 相机位姿和物体位姿仍用npy保存（更精确）
                np.save(os.path.join(episode_dir, "camera_pose.npy"), camera_pose)
                np.save(os.path.join(episode_dir, "object_pose.npy"), obj_pose)
                np.save(os.path.join(episode_dir, "camera_intrinsics.npy"), camera_intrinsics)
                
        except Exception as e:
            print(f"Process {process_id} Episode {episode} failed: {e}")
            continue
    
    env.close()
    return episode_end - episode_start

def split_train_val(data_dir, train_ratio=0.8, val_ratio=0.2, random_seed=42):
    """
    将生成的数据划分为训练集和验证集
    """
    print("Starting train/val split...")
    
    # 设置随机种子
    random.seed(random_seed)
    np.random.seed(random_seed)
    
    # 获取所有episode目录
    data_path = Path(data_dir)
    all_episodes = [d for d in data_path.iterdir() if d.is_dir() and d.name.startswith('episode_')]
    
    print(f"Found {len(all_episodes)} episodes")
    
    # 随机打乱
    random.shuffle(all_episodes)
    
    # 计算分割点
    train_split = int(len(all_episodes) * train_ratio)
    val_split = train_split + int(len(all_episodes) * val_ratio)
    
    train_episodes = all_episodes[:train_split]
    val_episodes = all_episodes[train_split:val_split]
    
    print(f"Train episodes: {len(train_episodes)}")
    print(f"Val episodes: {len(val_episodes)}")
    
    # 创建训练和验证目录
    train_dir = data_path / "train"
    val_dir = data_path / "val"
    
    train_dir.mkdir(exist_ok=True)
    val_dir.mkdir(exist_ok=True)
    
    # 移动训练集
    print("Moving training data...")
    for episode in train_episodes:
        dest = train_dir / episode.name
        if not dest.exists():
            shutil.move(str(episode), str(dest))
    
    # 移动验证集
    print("Moving validation data...")
    for episode in val_episodes:
        dest = val_dir / episode.name
        if not dest.exists():
            shutil.move(str(episode), str(dest))
    
    # 生成数据集统计信息
    generate_dataset_info(data_dir, len(train_episodes), len(val_episodes))
    
    print("Train/val split completed!")

def generate_dataset_info(data_dir, num_train, num_val):
    """
    生成数据集信息文件
    """
    info = {
        'dataset_name': 'power_drill_pose_estimation',
        'total_samples': num_train + num_val,
        'train_samples': num_train,
        'val_samples': num_val,
        'train_ratio': num_train / (num_train + num_val),
        'val_ratio': num_val / (num_train + num_val),
        'data_format': {
            'rgb': 'PNG image (H, W, 3) BGR format',
            'depth': 'PNG image (H, W) uint16, values in millimeters',
            'obj_seg': 'PNG image (H, W) uint8, 0=background, 255=object',
            'camera_pose': 'numpy array (4, 4) float32',
            'object_pose': 'numpy array (4, 4) float32',
            'camera_intrinsics': 'numpy array (3, 3) float32'
        },
        'generation_config': {
            'head_qpos_range': [[-0.7, -0.4], [0.7, 0.4]],
            'views_per_episode': 3,
            'stabilization_steps': 5,
            'depth_scale': 1000,  # depth values in mm
            'segmentation_method': 'heuristic_depth_color'
        }
    }
    
    import json
    with open(os.path.join(data_dir, 'dataset_info.json'), 'w') as f:
        json.dump(info, f, indent=2)

def create_data_loaders_info(data_dir):
    """
    创建用于训练的数据加载器信息文件
    """
    train_dir = os.path.join(data_dir, 'train')
    val_dir = os.path.join(data_dir, 'val')
    
    # 获取所有样本路径
    train_samples = []
    val_samples = []
    
    if os.path.exists(train_dir):
        train_samples = [os.path.join(train_dir, d) for d in os.listdir(train_dir) 
                        if os.path.isdir(os.path.join(train_dir, d))]
    
    if os.path.exists(val_dir):
        val_samples = [os.path.join(val_dir, d) for d in os.listdir(val_dir) 
                      if os.path.isdir(os.path.join(val_dir, d))]
    
    # 保存样本路径列表
    np.save(os.path.join(data_dir, 'train_samples.npy'), train_samples)
    np.save(os.path.join(data_dir, 'val_samples.npy'), val_samples)
    
    print(f"Saved train samples list: {len(train_samples)} samples")
    print(f"Saved val samples list: {len(val_samples)} samples")

def verify_dataset(data_dir):
    """
    验证数据集的完整性
    """
    print("Verifying dataset...")
    
    train_dir = os.path.join(data_dir, 'train')
    val_dir = os.path.join(data_dir, 'val')
    
    for split_name, split_dir in [('train', train_dir), ('val', val_dir)]:
        if not os.path.exists(split_dir):
            print(f"Warning: {split_name} directory not found!")
            continue
            
        episodes = [d for d in os.listdir(split_dir) 
                   if os.path.isdir(os.path.join(split_dir, d))]
        
        valid_episodes = 0
        invalid_episodes = []
        
        for episode in episodes:
            episode_path = os.path.join(split_dir, episode)
            required_files = ['rgb.png', 'depth.png', 'obj_seg.png', 
                            'camera_pose.npy', 'object_pose.npy']
            
            if all(os.path.exists(os.path.join(episode_path, f)) for f in required_files):
                valid_episodes += 1
            else:
                invalid_episodes.append(episode)
        
        print(f"{split_name}: {valid_episodes}/{len(episodes)} valid episodes")
        if invalid_episodes:
            print(f"Invalid episodes in {split_name}: {invalid_episodes[:5]}...")

def generate_training_data_parallel(total_episodes=3000, save_dir="data/power_drill_train", 
                                  num_processes=4, train_ratio=0.8, val_ratio=0.2):
    """并行生成训练数据并自动划分训练/验证集"""
    os.makedirs(save_dir, exist_ok=True)
    
    # 分配任务给不同进程
    episodes_per_process = total_episodes // num_processes
    tasks = []
    
    for i in range(num_processes):
        start = i * episodes_per_process
        end = (i + 1) * episodes_per_process if i < num_processes - 1 else total_episodes
        tasks.append((start, end, save_dir, i))
    
    # 使用进程池并行执行
    print("Starting parallel data generation...")
    with ProcessPoolExecutor(max_workers=num_processes) as executor:
        results = list(executor.map(generate_episode_data, tasks))
    
    total_generated = sum(results)
    print(f"Parallel data generation completed. Total episodes: {total_generated}")
    
    # 划分训练/验证集
    split_train_val(save_dir, train_ratio, val_ratio)
    
    # 创建数据加载器信息
    create_data_loaders_info(save_dir)
    
    # 验证数据集
    verify_dataset(save_dir)
    
    print(f"\nDataset generation and split completed!")
    print(f"Dataset location: {save_dir}")
    print(f"Train data: {save_dir}/train")
    print(f"Val data: {save_dir}/val")

if __name__ == "__main__":
    # 可以调整参数
    generate_training_data_parallel(
        total_episodes=10000,
        save_dir="data/power_drill_train",
        num_processes=1,
        train_ratio=0.9,
        val_ratio=0.1
    )