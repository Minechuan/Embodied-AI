## 运行 part 1 PPO
python -m hw_py.part1_ppo

## Part 2

os.environ['MUJOCO_GL'] = 'egl'
但是不行

如果使用端口转发：
Fatal server error:
(EE) Server is already active for display 1

If this server is no longer running, remove /tmp/.X1-lock and start again.


### Kill 
sudo pkill X