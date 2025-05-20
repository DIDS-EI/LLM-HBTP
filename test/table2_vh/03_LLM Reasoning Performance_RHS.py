from btgym.algos.llm_client.llm_ask_tools import extract_llm_from_instr_goal, extract_llm_from_reflect
from tools import execute_algorithm
from tools import find_from_small_act, load_dataset_and_cost
import time
import random
import numpy as np
import pandas as pd
import btgym
from ordered_set import OrderedSet
from btgym.utils import ROOT_PATH
from btgym.algos.llm_client.llms.gpt3 import LLMGPT3
from btgym.algos.bt_autogen.main_interface import BTExpInterface
from btgym.algos.llm_client.tools import goal_transfer_str, act_str_process, act_format_records
from btgym.utils.tools import collect_action_nodes, extract_objects
from btgym.algos.llm_client.vector_database_env_goal import search_nearest_examples
from btgym.utils.read_dataset import read_dataset
from btgym.algos.llm_client.llm_ask_tools import extract_llm_from_instr_goal, convert_conditions, \
    extract_llm_from_reflect
from btgym import BehaviorTree
import os

from btgym.envs.RobotHow_Small.exec_lib._base.RHSAction import RHSAction

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

def show_bt(algo,dir=None,id=None):
    ptml_string, cost, expanded_num = algo.post_process()  # 后处理

    if dir is None:
        file_name = "Theoretical_Bounds"
    else:
        file_name = id
    file_path = f'{ROOT_PATH}/../test/table2_vh/{dir}/{file_name}.btml'
    with open(file_path, 'w') as file:
        file.write(ptml_string)

    # 读取执行
    bt = BehaviorTree(file_path, env.behavior_lib)
    bt.print()
    bt.draw(file_name=f'{file_name}',target_directory = f'{ROOT_PATH}/../test/table2_vh/{dir}')


def perform_test(env, chosen_goal, priority_act_ls, llm_key_pred, llm_key_obj,mode="small-predicate-objs"):
    cur_cond_set = env.agents[0].condition_set

    if priority_act_ls != None:
        _priority_act_ls, pred, obj = act_format_records(priority_act_ls)
        key_predicates = list(set(llm_key_pred + pred))
        key_objects = list(set(llm_key_obj + obj))
        # put the objs in goal
        algo = BTExpInterface(env.behavior_lib, cur_cond_set=cur_cond_set,
                            priority_act_ls=priority_act_ls, key_predicates=key_predicates,
                            key_objects=key_objects,
                            selected_algorithm="opt", mode=mode,
                            llm_reflect=False, time_limit=10,
                            heuristic_choice=0,get_bt_summary = True)

        # algo = BTExpInterface(env.behavior_lib, cur_cond_set=cur_cond_set,
        #                       priority_act_ls=priority_act_ls, key_predicates=key_predicates,
        #                       key_objects=key_objects,
        #                       selected_algorithm="opt", mode="small-predicate-objs",  # mode=""
        #                       llm_reflect=False, time_limit=180,
        #                       heuristic_choice=0)

        goal_set = goal_transfer_str(' & '.join(chosen_goal))



        expanded_num, planning_time_total, cost, error, act_num, current_cost, record_act_ls = \
            execute_algorithm(algo, goal_set, cur_cond_set)
        time_limit_exceeded = algo.algo.time_limit_exceeded
        success = not error and not time_limit_exceeded

        act_space = len(algo.actions)
        bt_summary = algo.algo.bt_summary
    else:
        success = False
        goal_set, priority_act_ls, key_predicates, key_objects, \
            act_num, error, time_limit_exceeded, current_cost, expanded_num, planning_time_total, act_space, bt_summary,record_act_ls,algo = \
            None, None, None, None, None, None, None, None, None, None, None,None,None,None


    return success,  priority_act_ls, key_predicates, key_objects, \
        act_num, error, time_limit_exceeded, current_cost, expanded_num, planning_time_total, act_space,bt_summary,record_act_ls,algo


def convert_set_to_str(string_set):
    return ", ".join(f'\"{s}\"' for s in string_set)

# Function to validate a single goal
def validate_goal(env, chosen_goal, n,grounding_objs=set()):

    result_entry = {}

    print(f"test:{n}", chosen_goal)
    goal_set = goal_transfer_str(' & '.join(chosen_goal))
    cur_cond_set = env.agents[0].condition_set

    reflect_time = 0
    success = False
    bt_summary = []
    messages = []

    priority_act_ls = []
    key_predicates = []
    key_objects = []

    while not success and reflect_time <= 3:

        # using llm to reason
        # get prompt,messages
        if reflect_time == 0:
            llm_priority_act_ls, llm_key_pred, llm_key_obj, messages, parsed_fail = \
                extract_llm_from_instr_goal(llm, default_prompt_file, chosen_goal, cur_cond_set=cur_cond_set, verbose=False, messages=[])
            priority_act_ls = llm_priority_act_ls
            key_predicates = llm_key_pred
            key_objects = llm_key_obj
        else:
            # reflect
            # 用黄色打印
            print(f"\033[93mReflect Time: {reflect_time}; Goals: {chosen_goal} \033[0m")
            print(f"\033[93mbt_summary: {bt_summary} \033[0m")
            # reflect_prompt = ""

            # 获取全部的 predict 和 object
            TOOLS = RHSAction.TOOLS
            ACT_PREDICATES = {"Walk", "RightGrab", "LeftGrab", "RightPut", "LeftPut", "RightPutIn", "LeftPutIn", \
                              "Open", "Close", "SwitchOn", "SwitchOff", "Wipe", "PlugIn", "PlugOut", "Cut", "Wash"}

            # 查询还有哪些动作谓词没有用到
            not_use_pred = ACT_PREDICATES - set(key_predicates)
            not_use_pred_str = convert_set_to_str(not_use_pred)
            not_use_obj = TOOLS - set(key_objects)
            if not_use_obj != set():
                not_use_obj_str = ", and the tools/objects you have not used are: " + convert_set_to_str(
                    not_use_obj) + ""
            else:
                not_use_obj_str = ""

            reflect_prompt = (
                "The list of actions, predicates, and objects you provided is insufficient to accomplish the specified goals: \"{goals}\". "
                "Specifically, these only allow for the completion of the \"{have_finished}\", while failing to address the \"{not_finished}\".\n"

                "Note that you have not used the following Action Predicates and Objects: \"{not_use_pred_str}\",\"{not_use_obj_str}\"."
                "In regards to the unfinished goals \"{not_finished}\", check if these unused action predicates and objects are important and helpful for completing the goals. Please try to include any vital missing action predicates and objects.\n"
                
                "Please note some common-sense practices, such as plugging in the device before turning it on, using it, or placing objects inside it."
                
                "Please re-analyze the specified goal to identify the optimal actions, essential action predicates, and key objects necessary for achieving the goals. "
                "Use the same format as previously used, beginning with 'Optimal Actions:', 'Vital Action Predicates:', and 'Vital Objects:' respectively. Do not provide any additional explanations."
            )

            not_finished = set()
            for _g in d["Goals"]:
                #  _g = 'IsCut_pear'
                algo_tmp = BTExpInterface(env.behavior_lib, cur_cond_set=cur_cond_set,
                                          priority_act_ls=priority_act_ls, key_predicates=key_predicates,
                                          key_objects=key_objects,
                                          selected_algorithm="opt", mode="small-predicate-objs",
                                          llm_reflect=False, time_limit=3,
                                          heuristic_choice=0)
                gset = goal_transfer_str(_g)  # gset = [{'IsCut(pear)'}]
                algo_tmp.process(gset)
                state_g = cur_cond_set
                g = gset[0]  # g='IsCut(pear)'
                error_g, state_g, _, _, _ = algo_tmp.execute_bt(g, state_g, verbose=False)
                if error_g:
                    not_finished |= g

            have_finished = goal_set[0] - not_finished
            # goal_set[0] = {[{'IsClean(nightstand)','IsCut(pear)','IsOpen(window)','IsPlugged(wallphone)','IsSwitchedOn(wallphone)'}]}
            have_finished_str = convert_conditions(have_finished)
            not_finished_str = convert_conditions(not_finished)

            reflect_prompt = reflect_prompt.format(goals=d['Goals'], have_finished=have_finished_str,
                                                   not_finished=not_finished_str,
                                                   not_use_pred_str=not_use_pred_str, not_use_obj_str=not_use_obj_str)
            messages.append({"role": "user", "content": reflect_prompt})


            # get other information
            # priority_act_ls_new, llm_key_pred_new, llm_key_obj_new, messages, parsed_fail = \
            #     extract_llm_from_reflect(llm, messages)
            priority_act_ls_new, llm_key_pred_new, llm_key_obj_new, messages = \
                extract_llm_from_reflect(llm, messages)

            cyan = "\033[36m"
            reset = "\033[0m"
            print(f"{cyan}--- ADD ---{reset}")
            print(f"{cyan}priority_act_ls: {', '.join(set(priority_act_ls_new)-set(priority_act_ls))}{reset}")
            print(f"{cyan}key_predicates: {', '.join(set(llm_key_pred_new)-set(key_predicates))}{reset}")
            print(f"{cyan}key_objects: {', '.join(set(llm_key_obj_new)-set(key_objects))}{reset}")

            # expand action space
            priority_act_ls = list(OrderedSet(priority_act_ls + priority_act_ls_new))
            llm_key_pred = list(OrderedSet(key_predicates + llm_key_pred_new))
            llm_key_obj = list(OrderedSet(key_objects + llm_key_obj_new))
            _, pred, obj = act_format_records(priority_act_ls)
            key_predicates = list(set(llm_key_pred + pred))
            key_objects = list(set(llm_key_obj + obj))

        # print("priority_act_ls:",priority_act_ls,"\nllm_key_pred:",llm_key_pred,"\nllm_key_obj:",llm_key_obj)

        # test bt and get result
        test_result = perform_test(env, chosen_goal, priority_act_ls, key_predicates, key_objects)
        success,  _, _, _, \
            act_num, error, time_limit_exceeded, current_cost, expanded_num, planning_time_total, act_space,bt_summary,record_act_ls,algo = test_result


        # result_entry.update({
        #     'reflect': reflect_time,
        #     'Timeout': 1 if time_limit_exceeded else 0,
        #     'err': 1 if error else 0,
        #     'exp': expanded_num,
        #     'time': planning_time_total,
        #     'cost': current_cost,
        #     'act': record_act_ls
        # })

        result_entry.update({
            f"{reflect_time}_result": {
                                "success":success,
                                "time_limit_exceeded":time_limit_exceeded,
                                "priority_act_ls":priority_act_ls,
                                "key_predicates":key_predicates,
                                "key_objects":key_objects,
                                "record_act_ls":record_act_ls
                                }
        })


        # if fail, using llm to reason
        if success:
            result_entry.update({
                "reflect_time":reflect_time,
                "success": success,
                "time_limit_exceeded": time_limit_exceeded,

                "action_space":act_space,
                "expanded_num":expanded_num,
                "planning_time_total":planning_time_total,

                "actions_path_len":len(record_act_ls)
            })
            # save bt
            show_bt(algo,dir=f"bt_trees/{name}",id=f"{n}")
            break
        reflect_time += 1

    if not success:
        # big action space
        # test_result = perform_test(env, chosen_goal, priority_act_ls, llm_key_pred, llm_key_obj,mode="big")
        # success,  priority_act_ls, key_predicates, key_objects, \
        #     act_num, error, time_limit_exceeded, current_cost, expanded_num, planning_time_total, act_space, parsed_fail,record_act_ls ,algo= test_result
        # if fail, put in all predicate , revanlt objs and tools

        all_key_pred = list(set(llm_key_pred) | ACT_PREDICATES)
        all_obj_pred = list(set(llm_key_obj) | TOOLS | set(grounding_objs))

        test_result = perform_test(env, chosen_goal, priority_act_ls, all_key_pred, all_obj_pred)
        success,  priority_act_ls, key_predicates, key_objects, \
            act_num, error, time_limit_exceeded, current_cost, expanded_num, planning_time_total, act_space, parsed_fail,record_act_ls ,algo= test_result

        result_entry.update({
            "reflect_time": reflect_time,
            "success": success,
            "time_limit_exceeded": time_limit_exceeded,

            "action_space": act_space,
            "expanded_num": expanded_num,
            "planning_time_total": planning_time_total,

            "actions_path_len":len(record_act_ls)
        })

    return {
        'id': n, 'goals': ' & '.join(chosen_goal), 'priority_act_ls': priority_act_ls,
        'key_predicates': key_predicates, 'key_objects': key_objects, 'act_num': act_num, 'error': error,
        'time_limit_exceeded': time_limit_exceeded, 'act_space': act_space, 'expanded_num': expanded_num, 'current_cost': current_cost,
        'planning_time_total': planning_time_total,"success":success,"parsed_fail":parsed_fail,"reflect_time":reflect_time,
        "result_entry":result_entry,"record_act_ls":record_act_ls
    }




name = "RHB"

default_prompt_file = f"{ROOT_PATH}/../test/SCENES_EXP/prompt_{name}.txt"
dataset = read_dataset(f"{name}_test_50.txt")


dataset = read_dataset(f"{name}_test_50.txt")
# env = btgym.make("VHT-Small")
# cur_cond_set = env.agents[0].condition_set = {"IsRightHandEmpty(self)", "IsLeftHandEmpty(self)", "IsStanding(self)"}
# cur_cond_set |= {f'IsClose({arg})' for arg in RHSAction.CAN_OPEN}
# cur_cond_set |= {f'IsUnplugged({arg})' for arg in RHSAction.HAS_PLUG}
# cur_cond_set |= {f'IsSwitchedOff({arg})' for arg in RHSAction.HAS_SWITCH}
# all_obj = RHSAction.AllObject

# from btgym.envs.RobotHow.exec_lib._base.RHAction import RHAction
from btgym.envs.RobotHow.exec_lib._base.RHAction import RHAction as RHB
env = btgym.make("VHT-WatchTV")
cur_cond_set = env.agents[0].condition_set = {"IsRightHandEmpty(self)", "IsLeftHandEmpty(self)", "IsStanding(self)"}
cur_cond_set |= {f'IsClose({arg})' for arg in RHB.CAN_OPEN}
cur_cond_set |= {f'IsSwitchedOff({arg})' for arg in RHB.HAS_SWITCH}
cur_cond_set |= {f'IsUnplugged({arg})' for arg in RHB.HAS_PLUG}




big_actions = collect_action_nodes(env.behavior_lib)
all_pred = env.behavior_lib["Action"].values()




vaild_num = 50 #50

# Initialize accumulators and counters
test_results = []
reflect_record_reflect = []
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
# vaild_dataset = dataset[7:8]
for n, d in enumerate(vaild_dataset):

    # grounding_objs
    grounding_objs= d["Vital Objects"]
    # start
    result = validate_goal(env, d['Goals'], n,grounding_objs)
    test_results.append(result)
    # 计算一次成功率
    success = result["success"]
    fail = 0 if success else 1

    if success and fail == 0:
        test_success_count += 1
    total_expanded_num += result.get('expanded_num')
    total_planning_time_total += result.get('planning_time_total')
    total_fail_count += fail
    total_act_space += result.get("act_space")
    total_cost_ratio += 0
    current_cost = 0
    total_current_cost += 0



    # record
    instruction = d['Instruction']
    goals = d['Goals']
    d['Optimal Actions'] = act_str_process(d['Optimal Actions'], already_split=True)

    result_entry = result.get("result_entry")
    result_entry.update({
        'id': n,
        'Instruction': instruction,
        'Goals': goals,
        'Optimal Actions': d['Optimal Actions'],
        'Vital Action Predicates': d['Vital Action Predicates'],
        'Vital Objects': d['Vital Objects'],
        'actions_path_len':len(result.get("record_act_ls"))
    })

    reflect_record_reflect.append(result_entry)



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

time_str = time.strftime('%Y%m%d_%H%M', time.localtime())
details_filename = f'{ROOT_PATH}/../test/table2_vh/saved/output_{name}_details_{time_str}.csv'
details_df = pd.DataFrame(test_results)
details_df.to_csv(details_filename, index=False)

metrics_filename = f'{ROOT_PATH}/../test/table2_vh/saved/output_{name}_metrics_{time_str}.csv'
metrics_df.to_csv(metrics_filename, index=False)

# Set display options to ensure the entire DataFrame is printed
# pd.set_option('display.max_rows', None)
# pd.set_option('display.max_columns', None)
# # Print the entire metrics dataframe
# print(metrics_df)



    
# record
from analyze_results import analyze_reflect_results, print_analysis
analysis_results = analyze_reflect_results(name,reflect_record_reflect)
print_analysis(analysis_results)
