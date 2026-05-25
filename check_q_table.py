import pickle
import os
import numpy as np

def default_array():
    return np.zeros(2)  # 21点游戏中只有两个动作：要牌或停牌

# 获取当前文件路径
current_path = os.path.dirname(os.path.realpath(__file__))

# 列出目录下所有的q_table文件
q_table_files = [f for f in os.listdir(current_path) if f.startswith('q_table_') and f.endswith('.pkl')]

if len(q_table_files) == 0:
    print("没有找到Q-table文件!")
else:
    # 按文件名排序(最新的在最后)
    q_table_files.sort()
    latest_q_table = q_table_files[-1]
    
    print(f"正在读取最新的Q-table文件: {latest_q_table}")
    
    # 读取Q-table
    with open(os.path.join(current_path, latest_q_table), 'rb') as f:
        q_table = pickle.load(f)
    
    print(f"Q-table已加载，包含 {len(q_table)} 个状态")
    print(q_table)
