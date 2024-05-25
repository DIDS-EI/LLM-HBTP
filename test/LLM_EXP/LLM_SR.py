import pandas as pd
import btgym
import time
from btgym import BehaviorTree
from btgym.algos.bt_autogen.main_interface import BTExpInterface
from btgym.envs.virtualhometext.exec_lib._base.VHTAction import VHTAction
from btgym.utils import ROOT_PATH
from btgym.utils.read_dataset import read_dataset
from btgym.algos.llm_client.tools import goal_transfer_str, act_str_process, act_format_records
from btgym.algos.llm_client.llms.gpt3 import LLMGPT3
from btgym.algos.llm_client.llms.gpt4 import LLMGPT4
from btgym.algos.llm_client.llm_ask_tools import extract_llm_from_instr_goal
def load_result_csv(file_name):
    # Load the results from the previously saved CSV file
    # file_name = "llm_40.csv"
    results_df = pd.read_csv(file_name)
    results = results_df.to_dict(orient='records')  # Convert DataFrame back to list of dictionaries
    # Check that the data is successfully loaded
    print(f"Loaded {len(results)} results from '{file_name}'")


def load_dataset(data_path):
    data1 = read_dataset(data_path)
    len_data = len(data1)
    print(f"导入 {len_data} 条数据")
    return data1


# 导入大模型的结果
llm=LLMGPT3()
# default_prompt_file = f"{ROOT_PATH}\\algos\\llm_client\\prompt_VHT_just_goal.txt"
default_prompt_file = f"{ROOT_PATH}\\algos\\llm_client\\prompt_VHT_just_goal_old_example=0.425.txt"

# 导入数据集 真实值
dataset1 = load_dataset(f"{ROOT_PATH}/../test/dataset/data1_env1_40_test_reflect.txt")

results = []
error_count = 0  # Initialize error count
start_time = time.time()
for id, d in enumerate(dataset1): # 5可以
    print("\n== ID:", id, "  ", d['Instruction'])

    instruction = d['Instruction']
    goals = d['Goals']
    d['Optimal Actions'] = act_str_process(d['Optimal Actions'], already_split=True)
    # 大模型推荐的结果
    priority_act_ls, llm_key_pred, llm_key_obj, messages ,_= \
        extract_llm_from_instr_goal(llm, default_prompt_file, instruction, goals, verbose=False)
    # 增加板块，解析错误，直接重来，3次以上认定为错误

    print("Rec Act:", priority_act_ls)
    print("Rec Pred", llm_key_pred)
    print("Rec Obj:", llm_key_obj)

    # key_predicates 和 key_objects 要将推荐的 priority_act_ls 补充进来
    _, pred, obj = act_format_records(priority_act_ls)
    key_predicates = list(set(llm_key_pred + pred))
    key_objects = list(set(llm_key_obj + obj))

    result_entry = {
        'id': id,
        'Instruction': d['Instruction'],
        'Goals': d['Goals'],
        'Optimal Actions': d['Optimal Actions'],
        'Vital Action Predicates': d['Vital Action Predicates'],
        'Vital Objects': d['Vital Objects']
    }

    env = btgym.make("VHT-PutMilkInFridge")
    cur_cond_set = env.agents[0].condition_set = {"IsRightHandEmpty(self)", "IsLeftHandEmpty(self)", "IsStanding(self)"}
    cur_cond_set |= {f'IsClose({arg})' for arg in VHTAction.CAN_OPEN}
    cur_cond_set |= {f'IsSwitchedOff({arg})' for arg in VHTAction.HAS_SWITCH}
    cur_cond_set |= {f'IsUnplugged({arg})' for arg in VHTAction.HAS_PLUG}

    algo = BTExpInterface(env.behavior_lib, cur_cond_set=cur_cond_set, \
                          priority_act_ls=priority_act_ls, key_predicates=key_predicates,
                          key_objects=key_objects, \
                          selected_algorithm="opt", mode="small-predicate-objs", \
                          llm_reflect=False, time_limit=10,
                          heuristic_choice=0)

    start_time = time.time()
    goal_set = goal_transfer_str(' & '.join(d["Goals"]))
    algo.process(goal_set)
    end_time = time.time()

    ptml_string, cost, expanded_num = algo.post_process()
    print("Expanded Conditions: ", expanded_num)
    planning_time_total = (end_time - start_time)
    print("planning_time_total:", planning_time_total)
    print("cost_total:", cost)

    time_limit_exceeded = algo.algo.time_limit_exceeded

    # Simulation and test
    print("\n================ ")
    goal = goal_set[0]
    state = cur_cond_set
    error, state, act_num, current_cost, record_act_ls = algo.execute_bt(goal, state, verbose=False)
    print(f"Executed {act_num - 1} action steps")
    print("current_cost:", current_cost)
    print("================ ")

    # Add the results to the entry with appropriate prefixes
    result_entry[f'Timeout'] = 1 if time_limit_exceeded == True else 0
    result_entry[f'err'] = 1 if error == True else 0
    error_count += result_entry[f'err']
    result_entry[f'exp'] = expanded_num
    result_entry[f'time'] = planning_time_total
    result_entry[f'cost'] = cost
    result_entry[f'act'] = record_act_ls

    results.append(result_entry)


df = pd.DataFrame(results)
time_str = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime()).replace("-", "").replace(":", "")
# df.to_csv(f"LLM_40_time={time_str}.csv", index=False)
# print(f"Results have been saved to LLM_40_time={time_str}.csv")
df.to_csv(f"LLM_40.csv", index=False)
print(f"Results have been saved to LLM_40.csv")


# Calculate and print success rate
total_count = len(results)
success_rate = (total_count - error_count) / total_count
print(f"\033[92mSuccess Rate: {success_rate * 100:.2f}%\033[0m")

end_time = time.time()
time_total = (end_time - start_time)
print("Total time:", time_total)
