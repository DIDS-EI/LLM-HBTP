import time
import re
import os
import random
import numpy as np
import pandas as pd
from itertools import chain
from btgym import BehaviorTree, ExecBehaviorLibrary
import btgym
from btgym.utils import ROOT_PATH
from btgym.algos.llm_client.llms.gpt3 import LLMGPT3
from btgym.algos.bt_autogen.main_interface import BTExpInterface

from sympy import symbols, Not, Or, And, to_dnf, simplify_logic
from btgym.utils.read_dataset import read_dataset
from btgym.algos.llm_client.tools import goal_transfer_str, act_str_process, act_format_records
from btgym.utils.tools import collect_action_nodes,extract_objects
from tools import execute_algorithm
# from btgym.behavior_tree.utils.bt.draw import render_dot_tree

def show_bt(algo):
    ptml_string, cost, expanded_num = algo.post_process()  # 后处理
    print("Expanded Conditions: ",expanded_num)
    print("planning_time_total:", planning_time_total)
    print("cost_total:", cost)
    file_name = "Theoretical_Bounds"
    file_path = f'./{file_name}.btml'
    with open(file_path, 'w') as file:
        file.write(ptml_string)

    # 读取执行
    bt = BehaviorTree(file_name + ".btml", env.behavior_lib)
    bt.print()
    bt.draw()

# from btgym.envs.VirtualHome.exec_lib._base.VHAction import VHAction
# env = btgym.make("VH-PutMilkInFridge")
# cur_cond_set = env.agents[0].condition_set = {"IsRightHandEmpty(self)", "IsLeftHandEmpty(self)", "IsStanding(self)"}
# cur_cond_set |= {f'IsClose({arg})' for arg in VHAction.CanOpenPlaces}
# cur_cond_set |= {f'IsSwitchedOff({arg})' for arg in VHAction.HasSwitchObjects}
# big_actions = collect_action_nodes(env.behavior_lib)
#
# file_name="VS"
# data_path = f"{ROOT_PATH}/../test/SCENES_EXP/{file_name}.txt"
# output_path = f"{ROOT_PATH}/../test/SCENES_EXP/{file_name}_processed_data.txt"
# output_csv_path = f"{ROOT_PATH}/../test/SCENES_EXP/{file_name}_processed_h=1.csv"
# data1 = read_dataset(data_path)
# len_data = len(data1)
# print(f"导入 {len_data} 条数据")
# print(data1[0])

# # ================== RW ===============
# name = "RW"
# dataset = read_dataset(f"{name}_test_50.txt")
# from btgym.envs.RoboWaiter.exec_lib._base.RWAction import RWAction
# env = btgym.make("RWEnv")
# cur_cond_set = env.agents[0].condition_set = {'RobotNear(Bar)','Holding(Nothing)' }
# cur_cond_set |= {f'Exists({arg})' for arg in RWAction.all_object-{'Coffee', 'Water', 'Dessert'}}
# print(f"共收集到 {len(RWAction.all_object)} 个物体")


# # ================== RHS ===============
# name = "RHS"
# dataset = read_dataset(f"{name}_test_50.txt")
# from btgym.envs.RobotHow_Small.exec_lib._base.RHSAction import RHSAction
# env = btgym.make("VHT-Small")
# cur_cond_set = env.agents[0].condition_set = {"IsRightHandEmpty(self)", "IsLeftHandEmpty(self)", "IsStanding(self)"}
# cur_cond_set |= {f'IsClose({arg})' for arg in RHSAction.CAN_OPEN}
# cur_cond_set |= {f'IsUnplugged({arg})' for arg in RHSAction.HAS_PLUG}
# cur_cond_set |= {f'IsSwitchedOff({arg})' for arg in RHSAction.HAS_SWITCH}
# big_actions = collect_action_nodes(env.behavior_lib)


# ================== VH ===============
# name = "VH"
# dataset = read_dataset(f"{name}_test_50.txt")
# from btgym.envs.VirtualHome.exec_lib._base.VHAction import VHAction
# env = btgym.make("VH-PutMilkInFridge")
# cur_cond_set = env.agents[0].condition_set = {"IsRightHandEmpty(self)", "IsLeftHandEmpty(self)", "IsStanding(self)"}
# cur_cond_set |= {f'IsClose({arg})' for arg in VHAction.CanOpenPlaces}
# cur_cond_set |= {f'IsSwitchedOff({arg})' for arg in VHAction.HasSwitchObjects}
# big_actions = collect_action_nodes(env.behavior_lib)


# ================== RHB ===============
name = "RHB"
dataset = read_dataset(f"{name}_test_50.txt")
from btgym.envs.RobotHow.exec_lib._base.RHAction import RHAction as RHB
env = btgym.make("VHT-WatchTV")
cur_cond_set = env.agents[0].condition_set = {"IsRightHandEmpty(self)", "IsLeftHandEmpty(self)", "IsStanding(self)"}
cur_cond_set |= {f'IsClose({arg})' for arg in RHB.CAN_OPEN}
cur_cond_set |= {f'IsSwitchedOff({arg})' for arg in RHB.HAS_SWITCH}
cur_cond_set |= {f'IsUnplugged({arg})' for arg in RHB.HAS_PLUG}
big_actions = collect_action_nodes(env.behavior_lib)


# Initialize accumulators for lengths
total_priority_act_length = 0
total_algo_actions_length = 0
num_entries = 0
planning_time_total_all = 0

record_data = []

for n,d in enumerate(dataset):
    goal_str = ' & '.join(d["Goals"])
    act_str = ', '.join(d["Optimal Actions"])

    goal_set = goal_transfer_str(goal_str)
    print("goal_set:", goal_set)
    priority_act_ls = act_str_process(act_str)
    print("priority_act_ls:", priority_act_ls)

    # key_predicates = extract_objects(priority_act_ls)
    # 提取动作谓词
    key_predicates = []
    for act in priority_act_ls:
        key_predicates.append(act.split('(')[0])
    # print("key_predicates:", key_predicates)
    key_predicates = list(set(key_predicates))

    priority_obj_ls = []
    objects = set()
    pattern = re.compile(r'\((.*?)\)')
    for expr in chain(goal_set[0], priority_act_ls):
        match = pattern.search(expr)
        if match:
            objects.update(match.group(1).split(','))
    priority_obj_ls += list(objects)
    print("priority_obj_ls:", priority_obj_ls)

    algo = BTExpInterface(env.behavior_lib, cur_cond_set=cur_cond_set,
                          priority_act_ls=priority_act_ls, key_predicates=key_predicates,
                          key_objects=priority_obj_ls,
                          selected_algorithm="opt", mode="small-predicate-objs", #mode=""
                          llm_reflect=False, time_limit=180,
                          heuristic_choice=0)




    expanded_num, planning_time_total, cost, error, act_num, current_cost, record_act_ls = \
        execute_algorithm(algo, goal_set, cur_cond_set)
    time_limit_exceeded = algo.algo.time_limit_exceeded
    success = not error and not time_limit_exceeded

    # show bt
    # show_bt(algo)

    GREEN = "\033[32m"
    RED = "\033[31m"
    RESET = "\033[0m"
    print("\n")
    if success:
        print(f"{GREEN}- ID: {n}  Goal:{goal_str}  Success  -{RESET}")
    else:
        print(f"{RED}- ID: {n}  Goal:{goal_str}  Failed  -{RESET}")
        # break

    if time_limit_exceeded:
        RED = "\033[31m"
        RESET = "\033[0m"
        print(f"{RED}- ID: {n}  Goal:{goal_str}  Time Out  -{RESET}")
        planning_time_total=180
    print("\n")
    planning_time_total_all += planning_time_total
    print("time:",planning_time_total)

    # Update accumulators
    total_priority_act_length += len(priority_act_ls)
    total_algo_actions_length += len(algo.actions)
    num_entries += 1

    # 把每个数据的 priority_act_ls, len(priority_act_ls) 和 len(algo.actions) 分为3列 写入 csv
    data = {
        "priority_act_ls": priority_act_ls,
        "priority_act_ls_length": len(priority_act_ls),
        "algo_actions_length": len(algo.actions)
    }
    record_data.append(data)


df = pd.DataFrame(record_data)
df.to_csv(f"{name}_theoretical_bounds.csv", index=False)

# Calculate averages
average_priority_act_length = total_priority_act_length / num_entries
average_algo_actions_length = total_algo_actions_length / num_entries

# Print averages
print(f"Optimal path length (|p∗|): {average_priority_act_length}")
print(f"Size of the action space (|A∗|): {average_algo_actions_length}")
print(f"Planning Time Total: {planning_time_total_all/len(dataset)}")