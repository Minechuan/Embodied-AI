# 规划

## 实现代码

有两个部分需要完成：
1. 通过视觉估计物体的 pose。
2. 训练一个 policy：需要包括机器人的 Action，头部的旋转方向，四足狗的指令。

已有的 observation：
1. 头部相机
2. 手腕相机
3. 


需要注意的问题：
1.在作业二中只需要返回物体的 pose，只需要将相机坐标系中的物体 pose 转移到物体坐标系下，不需要对 Action 进行估计。这里需要逐个 Step 的 Action 进行预测，可以参考 hw2 的 codebase。


机器人的抓取和放置策略没有训练数据？
1. 仅仅使用传统的路径规划算法
2. 使用强化学习算法进行训练，并采集数据，然后用这些 trajoctory 训练没有 privilege obs 的 policy。



可能的进行方式：
1. 首先训练一个 detector 检测物体的 pose。
2. 同时训练一个 policy 预测 Action。（暂时先将 groundtruth object pose 作为输入）
3. 如果把任务视为两阶段 task，第一阶段 robot 只需要将看着桌面，拿着物体；第二阶段将视线转向 quadruped robot，然后再预测箱子的 pose，执行放置策略。（所以这样只需要控制 quadruped robot 每次停在一个特定的范围内）。


注意事项：
1. 为了提高训练的稳定性，可以将物体的 pose 转换到 robot root 的坐标系下。
2. 固定抓取后物体相对 robot root 的位置。




1. Image -> hw2 数据 
2. ik 抓和放？ 箱子的 6D pose —— 
3. 世界坐标系—— 建立以 robot 为原点

我的任务：训练对箱子的位姿估计！
实现步骤：确定一个可以抓到的位置——一个范围：在范围中初始化一些狗的位置——将箱子的坐标设置为 groundtruth，然后训练一个网络

In src\sim\combined_sim.py change the quadruped robot's pose.
store the point cloud image and the box center (x,y,z) 
这里 container: pos="-0.09 0 0.115" 就是箱子相对于 base body（也就是狗背上的那个 body）的局部平移

### 任务分工：

由于有明确的接口，大家先在自己的电脑上调 work:
**Function 1** 
driller pose estimation(zjl)
Input: image, camera parameters
(包含一个网络)
Output: driller pose

**Function 2** 
AprilTag pose estimation (mc)
Input: image, camera parameters
detect AprilTag
Retuen: humanoid_head_qpos, AprilTag pose

**Function 3** 
Dog Commend: (gxc)
Input: AprilTag pose, goal position(只有两个：最佳放置位置和狗的初始位置)
Output: commands sequence

**Function 4** 
Solve IK problem to plan actions for grasp. (ljl)
Input: target pose
Output: robot actions sequence



### reset implementation

wrapper_env.py: reset call self.sim.reset (in combined_sim.py); in this reset function, initial dog pose.

### Head Joint
```html
    <joint name="head_joint1" type="revolute">
        <limit effort="4.0" velocity="1.5" lower="-1.570796327" upper="1.570796327"/>
    </joint>
    <joint name="head_joint2" type="revolute">
        <limit effort="4.0" velocity="1.5" lower="-0.366519143" upper="0.366519143"/>
    </joint>
```


