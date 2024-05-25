import time
import re
from btgym import BehaviorTree
from btgym import ExecBehaviorLibrary
import btgym
from btgym.utils import ROOT_PATH
from btgym.algos.llm_client.llms.gpt3 import LLMGPT3
from btgym.algos.bt_autogen.main_interface import BTExpInterface
from btgym.envs.virtualhometext.exec_lib._base.VHTAction import VHTAction
from btgym.algos.bt_autogen.tools import state_transition
from sympy import symbols, Not, Or, And, to_dnf
from sympy import symbols, simplify_logic
from btgym.algos.llm_client.tools import goal_transfer_str, act_str_process
from btgym.algos.llm_client.llm_ask_tools import extract_llm_from_instr,llm_reflect

import random
import numpy as np
import re
seed = 0
random.seed(seed)
np.random.seed(seed)

env = btgym.make("VHT-PutMilkInFridge")

# 初始环境
cur_cond_set = env.agents[0].condition_set = {"IsRightHandEmpty(self)", "IsLeftHandEmpty(self)", "IsStanding(self)"}
cur_cond_set |= {f'IsClose({arg})' for arg in VHTAction.CAN_OPEN}
cur_cond_set |= {f'IsSwitchedOff({arg})' for arg in VHTAction.HAS_SWITCH}
cur_cond_set |= {f'IsUnplugged({arg})' for arg in VHTAction.HAS_PLUG}

# LLM
llm = LLMGPT3()
default_prompt_file = f"{ROOT_PATH}\\algos\\llm_client\\prompt_VHT.txt"
# # instuction = "Put the bowl in the dishwasher and wash it."
# # instuction="Put the milk and chicken in the fridge."
# # instuction="Put the milk or chicken in the fridge."
# instuction="Turn on the computer, TV, and lights, then put the bowl in the dishwasher and wash it"
# instuction="Prepare for a small birthday party by setting the dining table with candles, plates, and wine glasses. "+\
#     "Then, bake a cake using the oven, ensure the candles are switched on."+\
#     "Finally, make sure the kitchen counter is clean."
# instuction = "Prepare for a small birthday party by baking a cake using the oven, ensure the candles are switched on." + \
#              "Finally, make sure the kitchen counter is clean."

instuction = "4: Wash the bananas, cut the bananas and put it in the fridge"
goal_set, priority_act_ls, key_predicates, key_objects,messages = \
    extract_llm_from_instr(llm,default_prompt_file,instuction,cur_cond_set,\
                                choose_database = True)

# goal_set = [{'IsIn(cupcake,fridge)'}]
# goal_set = [{'IsOn(cutleryknife,kitchentable)', 'IsIn(cupcake,fridge)'}]
# priority_act_ls = {'PlugIn(fridge)', 'Open(fridge)', 'RightPut(cutleryknife,kitchentable)', 'Walk(cutleryknife)',
#                    'Walk(fridge)', 'Walk(cupcake)', 'RightGrab(cutleryknife)', 'RightGrab(cupcake)',
#                    'Walk(kitchentable)', 'RightPutIn(cupcake,fridge)'}

# todo: BTExp:process
MAX_TIME = 1
for try_time in range(MAX_TIME):

    # 在小动作空间里搜索
    # 需要重新写一下，小动作空间是 key_predicate 和 key_objects 的组合
    # 这个函数提供四种模式： 大动作空间、指定的动作空间、由物体组成的小动作空间、key_predicate和key_objects的小动作空间、
    algo = BTExpInterface(env.behavior_lib,cur_cond_set=cur_cond_set,priority_act_ls=priority_act_ls,\
                          key_predicates=key_predicates,key_objects=key_objects,\
                          selected_algorithm="opt", mode="small-predicate-objs", \
                          llm_reflect=False)


    start_time = time.time()
    algo.process(goal_set)
    end_time = time.time()

    ptml_string, cost, expanded_num = algo.post_process()  # 后处理
    print("Expanded Conditions: ", expanded_num)
    planning_time_total = (end_time - start_time)
    print("planning_time_total:", planning_time_total)
    print("cost_total:", cost)
    file_name = "grasp_milk"
    file_path = f'./{file_name}.btml'
    with open(file_path, 'w') as file:
        file.write(ptml_string)

    # 读取执行
    bt = BehaviorTree(file_name + ".btml", env.behavior_lib)
    # bt.print()
    # bt.draw()

    env.agents[0].bind_bt(bt)
    env.reset()
    env.print_ticks = True

    # simulation and test
    print("\n================ ")
    goal = goal_set[0]
    state = cur_cond_set
    error, state, act_num, current_cost, record_act_ls = algo.execute_bt(goal, state, verbose=True)
    print(f"一定运行了 {act_num - 1} 个动作步")
    print("current_cost:", current_cost)
    print("================ ")

    # 大模型反馈
    if not error:
        break
    print("大模型重推荐......")
    reflect_prompt = ""
    reflect_prompt += (
            "The list of actions, predicates, and objects you provided is insufficient to accomplish the Goals, "
            "as it seems to have overlooked the dependencies between actions, "
            "such as the need to plug in an electrical appliance before using it, "
            "the requirement of a rag for wiping, and the necessity of a cutlery knife for cutting.")

    # messages.append({"role": "user", "content": prompt})
    # answer = llm.request(message=messages)
    # messages.append({"role": "assistant", "content": answer})
    # print("answer:",answer)
    # act_str = answer.split("Actions:")[1]
    # # act_str = re.sub(r'\s+|[\[\]\(\)\n]', '', act_str)
    # # priority_act_ls = act_str_process(act_str)
    # priority_act_ls = [action.replace(" ", "") for action in act_str.split(",")]
    # print(priority_act_ls)

    # goal_set, priority_act_ls, key_predicates, key_objects, messages = \
    #     llm_reflect(llm, messages, reflect_prompt)

    # priority_act_ls, priority_obj_ls
    # 问题：小动作空间，是不是不是分级启发式，priority_obj_ls限制动作空间，priority_act_ls采用计数的启发式
    # 问题：大模型重新推荐，是不是全部都要重新推荐

# env.close()
