# Test the Blackjack agent with Q-table

import os
import pickle
import gym
import numpy as np
from collections import defaultdict

def default_array():
    return np.zeros(2)

# 获取当前文件路径
current_path = os.path.dirname(os.path.realpath(__file__))

# 列出目录下所有的q_table文件
q_table_files = [f for f in os.listdir(current_path) if f.startswith('q_table_') and f.endswith('.pkl')]

if len(q_table_files) == 0:
    print("没有找到Q-table文件!")
    exit()

# 按文件名排序(最新的在最后)
q_table_files.sort()
latest_q_table = q_table_files[-1]

print(f"正在读取最新的Q-table文件: {latest_q_table}")

# 读取Q-table
with open(os.path.join(current_path, latest_q_table), 'rb') as f:
    q_table = pickle.load(f)

print(f"Q-table已加载，包含 {len(q_table)} 个状态")

# 创建测试环境
# env = gym.make('Blackjack-v1', render_mode='human')
env = gym.make('Blackjack-v1')
NUM_EPISODES = 10000

# 统计数据
total_rewards = 0
wins = 0
losses = 0
draws = 0

for episode in range(NUM_EPISODES):
    state, _ = env.reset()
    done = False
    episode_reward = 0
    
    while not done:
        # 将状态转换为元组以用作字典键
        state_key = tuple(state) if isinstance(state, (list, np.ndarray)) else state
        
        # 选择动作
        action = np.argmax(q_table[state_key])
        
        # 执行动作
        state, reward, done, _, _ = env.step(action)
        episode_reward += reward
    
    total_rewards += episode_reward
    
    # 统计游戏结果
    if episode_reward > 0:
        wins += 1
    elif episode_reward < 0:
        losses += 1
    else:
        draws += 1

# 打印统计结果
print("\n测试结果统计:")
print(f"总场数: {NUM_EPISODES}")
print(f"胜率: {wins/NUM_EPISODES*100:.2f}%")
print(f"负率: {losses/NUM_EPISODES*100:.2f}%")
print(f"平局率: {draws/NUM_EPISODES*100:.2f}%")
print(f"平均奖励: {total_rewards/NUM_EPISODES:.3f}")

env.close()
