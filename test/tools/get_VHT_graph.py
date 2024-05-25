"""
    选择不同的环境 env ，将 graph 保存为 pkl，方便 VirtualHomeText 环境导入不同的环境
"""
from btgym.envs.virtualhome.simulation.unity_simulator.comm_unity import UnityCommunication
from btgym.utils import ROOT_PATH
import subprocess
import pickle
import json

file_name = f'{ROOT_PATH}\\..\\simulators\\virtualhome\\windows\\VirtualHome.exe'
print(file_name)
# 启动 exe 文件
process = subprocess.Popen(file_name, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

print('Starting Unity...')
comm = UnityCommunication()

print('Starting scene...')
simulator_launched = False
env_index=6# 0-49
while not simulator_launched:
    try:
        comm.reset(env_index) # 0-49
        simulator_launched = True
    except:
        pass

_, env_g = comm.environment_graph()

# 将 env_g 保存到
# graph_path = f'{ROOT_PATH}\\envs\\virtualhometext\\graphs\\simulation\\graph_{env_index}.pkl'
# print(graph_path)
# 将字典保存到 graph_input.pkl 文件中
# with open(graph_path, "wb") as pkl_file:
#     pickle.dump(env_g, pkl_file)

# 将字典保存到 JSON 文件中
graph_path = f'{ROOT_PATH}\\envs\\virtualhometext\\graphs\\graph_{env_index}.json'
print(graph_path)
with open(graph_path, 'w') as json_file:
    json.dump(env_g, json_file, indent=4)