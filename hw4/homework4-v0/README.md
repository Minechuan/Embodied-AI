# Assignment 4 (12pt)

In this assignment you will need to complete the combined task consisting a quadruped robot and a humanoid robot.

## Environment

You can install the environment as follows:

```sh
conda create -n hw4 python=3.10
conda activate hw4
conda install conda-forge::roboticstoolbox-python==1.0.3

# if you can use Nvidia GPU
conda install pytorch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 pytorch-cuda=12.4 -c pytorch -c nvidia 
# you might need to change the cuda version here
# if you can only use CPU
conda install pytorch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 cpuonly -c pytorch 

pip install tqdm h5py wandb plotly pygments ikpy transforms3d pyyaml pillow matplotlib trimesh dm_control
pip install numpy==1.26.4 scipy==1.11.4 sympy==1.13.1 opencv-python==4.11.0.86
pip install onnxruntime pyapriltags

```

## Simulation Task Details

The task requires you to implement the policies both for the quadruped robot (go2) and the humanoid robot (galbot). The whole pipeline consists of three steps: 1.guiding the quadruped to a position near table, 2.controlling the humanoid to grasp the object (a driller) and drop in the box which is on the quadruped's back and 3.guiding the quadruped going back to its initial position.

An apriltag is attached on quadruped's head for location. You don't need to implement the policy of quadruped's locomotion. Instead, you just need to give command of its velocity in xy plane and angular velocity around z axis. For tracking the apriltag in case it runs out of sight, you are allowed to adjust the humanoid head pose by controlling the qpos of head joints.

The grasping driller is almost the same as Assignment 2. But you cannot directly set the arm qpos to pre-grasp state, since it is not feasible in real world. Please implement a whole policy of planning to grasp and lift, and moving the driller to drop it in the box. Note that the simulation environment this time is different from that in Assignment2, and your may have to generate your own dataset and re-train the detection network if it no longer works.

The simulation task will judge your whole policy by three metrics. See metrics below for details.

We have provided a demo code for the simulation guidance, you can run ```python demo.py``` to see the demo scene.
The main code template is in ```main.py```. You will implement your code here.


## Metric I: Object Pose Estimation (4pt)

You are required to estimate the pose of driller which is placed on the table. The ground truth of driller is the pose of the object in simulation. You can access only rgb and depth of the wrist camera. In ```main.py```, ```Metric['obj_pose']``` will judge the precision of pose estimation.

You will get one score if the mean translation error is smaller than 0.025 m and the mean rotation error is smaller than 0.25 rad. There will be four test cases.


## Metric II: Dropping Precision (4pt)

The dropping precision metric is whether the humanoid drops the driller into the box. The process is judged success if the position of driller enters the box area in simulation. In ```main.py```, ```Metric['drop_precision']``` will judge the precision of pose estimation. 

You will get one score for each success case. There will be four test cases. 


## Metric III: Quadruped Returning (4pt)

This metric will detect whether the quadruped robot take the box to the initial position. The judging is triggered after the dropping success. In ```main.py```, ```Metric['quad_return']``` will judge the precision of pose estimation. The process is judged success if the box's position distance from its initial position is smaller than 0.1m.  

You will get one score for each success case. There will be four test cases.


## Precautions

You are NOT allowed to: (1) read the pose of object in simulation, (2) read the pose of quadruped robot or the box in the simulation, (3) diretly set pose of object, set pose of robot root or set qpos of robot in simulation


## Testing & Uploading

Currently you can use mujoco viewer to visualize whether your implementation works. The testing code and example test cases will be release later on.
After testing is released, we will provide submission details.