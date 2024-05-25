from btgym.algos.llm_client.llm_ask_tools import extract_llm_from_instr_goal, extract_llm_from_reflect, \
    convert_conditions, format_example
from tools import execute_algorithm, load_dataset, setup_default_env
from tools import find_from_small_act, load_dataset_and_cost
import time
import random
import numpy as np
import pandas as pd
import btgym
from btgym.utils import ROOT_PATH
from btgym.algos.llm_client.llms.gpt3 import LLMGPT3
from btgym.algos.bt_autogen.main_interface import BTExpInterface
from btgym.algos.llm_client.tools import goal_transfer_str, act_str_process, act_format_records
from btgym.utils.tools import collect_action_nodes, extract_objects
from btgym.algos.llm_client.vector_database_env_goal import search_nearest_examples
from btgym.utils.read_dataset import read_dataset

# Set random seed
random_seed = 0
np.random.seed(random_seed)
random.seed(random_seed)

# Initialize the LLM
llm = LLMGPT3()


def convert_set_to_str(string_set):
    return ", ".join(f'\"{s}\"' for s in string_set)


def get_or_default(value, default):
    return value if value is not None else default


def perform_test(env, chosen_goal, database_index_path, reflect_time=0, train=False, choose_database=True):
    cur_cond_set = env.agents[0].condition_set
    priority_act_ls, llm_key_pred, llm_key_obj, messages, distances, parsed_fail = \
        extract_llm_from_instr_goal(llm, default_prompt_file, 1, chosen_goal, verbose=False,
                                    choose_database=choose_database, database_index_path=database_index_path)
    if priority_act_ls != None:
        _priority_act_ls, pred, obj = act_format_records(priority_act_ls)
        key_predicates = list(set(llm_key_pred + pred))
        key_objects = list(set(llm_key_obj + obj))
        # goal里的目标也要加进去
        algo = BTExpInterface(env.behavior_lib, cur_cond_set=cur_cond_set,
                              priority_act_ls=priority_act_ls, key_predicates=key_predicates,
                              key_objects=key_objects,
                              selected_algorithm="opt", mode="small-predicate-objs",
                              llm_reflect=False, time_limit=10,
                              heuristic_choice=0)
        goal_set = goal_transfer_str(' & '.join(chosen_goal))
        expanded_num, planning_time_total, cost, error, act_num, current_cost, record_act_ls = \
            execute_algorithm(algo, goal_set, cur_cond_set)
        time_limit_exceeded = algo.algo.time_limit_exceeded
        success = not error and not time_limit_exceeded

        act_space = len(algo.actions)

        return success, np.mean(distances), priority_act_ls, key_predicates, key_objects, \
            act_num, error, time_limit_exceeded, current_cost, expanded_num, planning_time_total, act_space, parsed_fail
    else:
        success = False
        goal_set, priority_act_ls, key_predicates, key_objects, \
            act_num, error, time_limit_exceeded, current_cost, expanded_num, planning_time_total, act_space = \
            None, None, None, None, None, None, None, None, None, None, None

    # 搜索小动作空间得到一个解
    if not success and train:
        success, priority_act_ls, key_predicates, key_objects, \
            act_num, error, time_limit_exceeded, current_cost, expanded_num, planning_time_total, act_space = \
            find_from_small_act(chosen_goal)

    if not choose_database:
        distances = 0
    return success, np.mean(distances), priority_act_ls, key_predicates, key_objects, \
        act_num, error, time_limit_exceeded, current_cost, expanded_num, planning_time_total, act_space, parsed_fail


# Function to validate a single goal
def validate_goal(env, chosen_goal, n, database_index_path=None, round_num=None, database_num=None, reflect_time=0,
                  choose_database=True):
    print(f"test:{n}", chosen_goal)

    test_result = perform_test(env, chosen_goal, database_index_path, reflect_time=reflect_time,
                               choose_database=choose_database)
    success, avg_distance, priority_act_ls, key_predicates, key_objects, \
        act_num, error, time_limit_exceeded, current_cost, expanded_num, planning_time_total, \
        act_space, parsed_fail  = test_result

    fail = 0
    while not success:
        fail += 1
        RED = "\033[31m"
        RESET = "\033[0m"
        print(f"{RED}---- ID: {n}  Goal:{chosen_goal}    Fail Times: {fail} -----{RESET}")

        if fail > 6:
            # 跑全空间
            print(f"{RED}---- ID: {n}  Goal:{chosen_goal}  mode=\"big\" -----{RESET}")
            algo = BTExpInterface(env.behavior_lib, cur_cond_set=cur_cond_set,
                                  priority_act_ls=priority_act_ls, key_predicates=key_predicates,
                                  key_objects=key_objects,
                                  selected_algorithm="opt", mode="big",
                                  llm_reflect=False, time_limit=10,
                                  heuristic_choice=0)
            goal_set = goal_transfer_str(' & '.join(chosen_goal))
            expanded_num, planning_time_total, cost, error, act_num, current_cost, record_act_ls = \
                execute_algorithm(algo, goal_set, cur_cond_set)
            time_limit_exceeded = algo.algo.time_limit_exceeded
            success = not error and not time_limit_exceeded
            act_space = len(algo.actions)
            break

        # 在这里添加例子，加入到
        # 如果这里面把例子中的pred和obj也加进去
        nearest_examples, distances = search_nearest_examples(database_index_path, llm, chosen_goal,
                                                              top_n=5 * 2 ** (fail - 1))
        ex_preds = set()
        ex_objs = set()
        for ex in nearest_examples:
            ex_preds |= set(ex['value']['Vital Action Predicates'].replace(" ", "").split(","))
            ex_objs |= set(ex['value']['Vital Objects'].replace(" ", "").split(","))
        key_predicates = list(set(key_predicates) | ex_preds)
        key_objects = list(set(key_objects) | ex_objs)

        algo = BTExpInterface(env.behavior_lib, cur_cond_set=cur_cond_set,
                              priority_act_ls=priority_act_ls, key_predicates=key_predicates,
                              key_objects=key_objects,
                              selected_algorithm="opt", mode="small-predicate-objs",
                              llm_reflect=False, time_limit=30,
                              heuristic_choice=0)
        goal_set = goal_transfer_str(' & '.join(chosen_goal))
        expanded_num, planning_time_total, cost, error, act_num, current_cost, record_act_ls = \
            execute_algorithm(algo, goal_set, cur_cond_set)
        time_limit_exceeded = algo.algo.time_limit_exceeded
        success = not error and not time_limit_exceeded
        act_space = len(algo.actions)

        # 疑问：动作空间如何每次计算？没加一个要遍历所有动作？加了物体又怎么判别呢
        # 感觉可以加5个example、10个example、20个example....

        # VD 实验， 简单、依赖、长序列、混合？ 成功率和expanded？ 要扩张吗
        # 感觉体现向量数据库学习过程，可以不扩张，单纯看 四种数据集上的学习能力

    print(f"\033[92mtest:{n} {chosen_goal} {act_num}\033[0m")
    return {
        'round': round_num, 'id': n, 'goals': ' & '.join(chosen_goal), 'priority_act_ls': priority_act_ls,
        'key_predicates': key_predicates, 'key_objects': key_objects, 'act_num': act_num, 'error': error,
        'time_limit_exceeded': time_limit_exceeded, 'act_space': act_space, 'expanded_num': expanded_num, 'current_cost': current_cost,
        'planning_time_total': planning_time_total, 'average_distance': avg_distance, 'database_size': database_num
    }, success, avg_distance, fail,parsed_fail


# ============= VH ================
name = "VH"
default_prompt_file = f"prompt_{name}_no_example.txt"
dataset = read_dataset(f"{name}_test_50.txt")
database_index_path = f"{ROOT_PATH}/../test/SCENES_EXP/DATABASE/{name}_100_env_goal_vectors.index"
from btgym.envs.virtualhome.exec_lib._base.VHAction import VHAction
env = btgym.make("VH-PutMilkInFridge")
cur_cond_set = env.agents[0].condition_set = {"IsRightHandEmpty(self)", "IsLeftHandEmpty(self)", "IsStanding(self)"}
cur_cond_set |= {f'IsClose({arg})' for arg in VHAction.CanOpenPlaces}
cur_cond_set |= {f'IsSwitchedOff({arg})' for arg in VHAction.HasSwitchObjects}
big_actions = collect_action_nodes(env.behavior_lib)


# name = "RHS"
# default_prompt_file = f"prompt_VH.txt"
# dataset = read_dataset(f"RHS.txt")
# # dataset = load_dataset_and_cost(f"RHS.txt")
# # env, _ = setup_default_env()
# from btgym.envs.virtualhometextsmall.exec_lib._base.VHTAction import VHTAction as RHS
# env = btgym.make("VHT-Small")
# cur_cond_set = env.agents[0].condition_set = {"IsRightHandEmpty(self)", "IsLeftHandEmpty(self)", "IsStanding(self)"}
# cur_cond_set |= {f'IsClose({arg})' for arg in RHS.CAN_OPEN}
# cur_cond_set |= {f'IsSwitchedOff({arg})' for arg in RHS.HAS_SWITCH}
# cur_cond_set |= {f'IsUnplugged({arg})' for arg in RHS.HAS_PLUG}
# print(f"共收集到 {len(RHS.AllObject)} 个物体")

# =========================  RHS ======================
# name = "RHS"
# default_prompt_file = f"prompt_{name}_no_example.txt"
# dataset = read_dataset(f"{name}_test_50.txt")
# database_index_path = f"{ROOT_PATH}/../test/SCENES_EXP/DATABASE/{name}_100_env_goal_vectors.index"
# from btgym.envs.virtualhometextsmall.exec_lib._base.VHTAction import VHTAction
# env = btgym.make("VHT-Small")
# cur_cond_set = env.agents[0].condition_set = {"IsRightHandEmpty(self)", "IsLeftHandEmpty(self)", "IsStanding(self)"}
# cur_cond_set |= {f'IsClose({arg})' for arg in VHTAction.CAN_OPEN}
# cur_cond_set |= {f'IsUnplugged({arg})' for arg in VHTAction.HAS_PLUG}
# cur_cond_set |= {f'IsSwitchedOff({arg})' for arg in VHTAction.HAS_SWITCH}
# big_actions = collect_action_nodes(env.behavior_lib)




vaild_num = 50

# Initialize accumulators and counters
test_results = []
test_success_count = 0
total_expanded_num = 0
total_planning_time_total = 0
total_cost_ratio = 0
total_current_cost = 0
total_fail_count = 0
total_act_space = 0
total_parsed_fail = 0
# Dataframe to store metrics for each round
metrics_df = pd.DataFrame(columns=[
    "Test Success Rate", "Average Expanded Num", "Average Planning Time Total", "Average Current Cost"
])

# ========================= 并行 ========================
# with concurrent.futures.ThreadPoolExecutor() as executor:
#     futures = [executor.submit(validate_goal, env, d['Goals'], database_index_path, round_num, n, database_num,
#                                reflect_time=reflect_time,choose_database=True) \
#                for n, d in enumerate(vaild_dataset_test)]
#     for future in concurrent.futures.as_completed(futures):
#         result, success, _ = future.result()
# ========================= 并行 ========================

# ========================= 串行========================
vaild_dataset = dataset[:vaild_num]
for n, d in enumerate(vaild_dataset):
    result, success, avg_distance, fail, parsed_fail = validate_goal(env, d['Goals'], n, choose_database=True,
                                                                     database_index_path=database_index_path)
    test_results.append(result)
    # 计算一次成功率
    if success and fail == 0:
        test_success_count += 1
    total_expanded_num += result.get('expanded_num')
    total_planning_time_total += result.get('planning_time_total')
    total_fail_count += fail
    total_parsed_fail += parsed_fail
    total_act_space += result.get("act_space")
    total_cost_ratio += 0
    current_cost = 0
    total_current_cost += 0

# Calculate metrics
num_entries = len(vaild_dataset)
success_rate = test_success_count / num_entries
average_fail_count = total_fail_count / num_entries if num_entries else 0
average_act_space = total_act_space / num_entries if num_entries else 0
average_expanded_num = total_expanded_num / num_entries if num_entries else 0
average_planning_time_total = total_planning_time_total / num_entries if num_entries else 0
average_current_cost = total_current_cost / num_entries if num_entries else 0
average_parsed_fail = total_parsed_fail / num_entries if num_entries else 0

# Append metrics to dataframe
round_metrics = pd.DataFrame([{
    "Test Success Rate": success_rate,
    "Average Fail Count": average_fail_count,
    "Average Act Space": average_act_space,
    "Average Expanded Num": average_expanded_num,
    "Average Planning Time Total": average_planning_time_total,
    "Average Current Cost": average_current_cost,
    "Average Parsed Fail": average_parsed_fail
}])

metrics_df = pd.concat([metrics_df, round_metrics], ignore_index=True)

# Output metrics
print(f"Test Success Rate: {success_rate}")
print(f"Average Parsed Fail: {average_parsed_fail}")
print(f"Average Fail Count: {average_fail_count}")
print(f"Average Act Space: {average_act_space}")
print(f"Average Expanded Num: {average_expanded_num}")
print(f"Average Planning Time Total: {average_planning_time_total}")
print(f"Average Current Cost: {average_current_cost}")

# Save daily detailed results and metrics to CSV
time_str = time.strftime('%Y%m%d', time.localtime())
details_filename = f'output_{name}_details_{time_str}.csv'
details_df = pd.DataFrame(test_results)
details_df.to_csv(details_filename, index=False)

metrics_filename = f'output_{name}_metrics_{time_str}.csv'
metrics_df.to_csv(metrics_filename, index=False)

# Set display options to ensure the entire DataFrame is printed
# pd.set_option('display.max_rows', None)
# pd.set_option('display.max_columns', None)
# # Print the entire metrics dataframe
# print(metrics_df)
