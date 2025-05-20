
from btgym.utils.read_dataset import read_dataset
import btgym
import time
from btgym.envs.virtualhometext.exec_lib._base.VHTAction import VHTAction
from tools import *


def setup_default_env():
    env = btgym.make("VHT-PutMilkInFridge")
    cur_cond_set = env.agents[0].condition_set = {"IsRightHandEmpty(self)", "IsLeftHandEmpty(self)", "IsStanding(self)"}
    cur_cond_set |= {f'IsClose({arg})' for arg in VHTAction.CAN_OPEN}
    cur_cond_set |= {f'IsSwitchedOff({arg})' for arg in VHTAction.HAS_SWITCH}
    cur_cond_set |= {f'IsUnplugged({arg})' for arg in VHTAction.HAS_PLUG}
    return env, cur_cond_set

def execute_algorithm(algo, goal_set, cur_cond_set):


    start_time = time.time()
    algo.process(goal_set)
    end_time = time.time()
    planning_time_total = end_time - start_time

    ptml_string, cost, expanded_num = algo.post_process()
    error, state, act_num, current_cost, record_act_ls = algo.execute_bt(\
        goal_set[0], cur_cond_set, verbose=False)

    return expanded_num, planning_time_total, cost, error, act_num, current_cost, record_act_ls


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
    print(f"load {len_data} data")
    return data1


def load_dataset_and_cost(file_path):
    dataset = []

    with open(file_path, 'r') as file:
        lines = file.readlines()

    current_entry = {}
    for line in lines:
        line = line.strip()
        if line.isdigit():
            if current_entry:
                dataset.append(current_entry)
                current_entry = {}
            current_entry["id"] = int(line)
        elif line.startswith("Environment:"):
            current_entry["Environment"] = int(line.split(":")[1])
        elif line.startswith("Instruction:"):
            try:
                current_entry["Instruction"] = line.split(": ", 1)[1]
            except:
                current_entry["Instruction"] = ""
        elif line.startswith("Goals:"):
            goals = line.split(": ", 1)[1]
            current_entry["Goals"] = [goal.strip() for goal in goals.split("&")]
        elif line.startswith("Optimal Actions:"):
            actions = line.split(": ", 1)[1]
            current_entry["Optimal Actions"] = [action.strip() for action in actions.split(",")]
        elif line.startswith("Vital Action Predicates:"):
            predicates = line.split(": ", 1)[1]
            current_entry["Vital Action Predicates"] = [predicate.strip() for predicate in predicates.split(",")]
        elif line.startswith("Vital Objects:"):
            objects = line.split(": ", 1)[1]
            current_entry["Vital Objects"] = [obj.strip() for obj in objects.split(",")]
        elif line.startswith("cost:"):
            current_entry["cost"] = int(line.split(": ")[1])

    if current_entry:
        dataset.append(current_entry)

    return dataset

def count_accuracy(expected, actual):
    correct = 0
    incorrect = 0

    for item in actual:
        if item in expected:
            correct += 1
        else:
            incorrect += 1

    total = correct + incorrect
    accuracy = (correct / len(expected)) * 100
    error_rate = (incorrect / len(expected)) * 100

    return correct, incorrect, accuracy, error_rate


def identify_and_print_diffs(d, priority_act_ls, key_predicates, key_objects):
    """
    Identifies and prints missing or incorrectly predicted actions, predicates, and objects.
    Args:
        d (dict): Dictionary containing 'Actions', 'Vital Action Predicates', and 'Vital Objects'.
        priority_act_ls (list): List of priority actions.
        key_predicates (list): List of key predicates.
        key_objects (list): List of key objects.
    """
    diff=False


    missing_act = set(d['Optimal Actions']) - set(priority_act_ls)
    incorrect_act = set(priority_act_ls) - set(d['Optimal Actions'])

    missing_predicate = set(d['Vital Action Predicates']) - set(key_predicates)
    incorrect_predicate = set(key_predicates) - set(d['Vital Action Predicates'])

    missing_obj = set(d['Vital Objects']) - set(key_objects)
    incorrect_obj = set(key_objects) - set(d['Vital Objects'])

    if missing_act:
        print("Missing Actions:", missing_act)
    if incorrect_act:
        print("Incorrectly Predicted Actions:", incorrect_act)

    if missing_predicate:
        print("Missing Predicates:", missing_predicate)
    if incorrect_predicate:
        print("Incorrectly Predicted Predicates:", incorrect_predicate)

    if missing_obj:
        print("Missing Objects:", missing_obj)
    if incorrect_obj:
        print("Incorrectly Predicted Objects:", incorrect_obj)

    if missing_act or incorrect_act or missing_predicate or incorrect_predicate or missing_obj or incorrect_obj:
        diff = True

    return diff


# d = {
#     'Actions': ['move', 'pick', 'turn'],
#     'Vital Action Predicates': ['is_moving', 'is_holding', 'is_rotating'],
#     'Vital Objects': ['box', 'ball', 'cup']
# }
# priority_act_ls = ['move', 'pick', 'jump']
# key_predicates = ['is_moving', 'is_holding', 'is_running']
# key_objects = ['box', 'ball', 'cone']
#
# identify_and_print_diffs(d, priority_act_ls, key_predicates, key_objects)


import pandas as pd


def analyze_data_tabular(data1, counts):
    print('----------  Data Info ----------------')
    # Initialize results for each group
    num_groups = len(counts)
    group_results = [{'Optimal Actions Count': 0,
                      'PlugIn': 0, 'Open': 0, 'Close': 0, 'Cut': 0, 'Wipe': 0, 'Wash': 0,
                      'kitchenknife': 0, 'faucet': 0, 'rag': 0,
                      'Total Records': count}
                     for count in counts]

    # Process data for each group
    start = 0
    for i, count in enumerate(counts):
        for j in range(start, start + count):
            d = data1[j]

            # Count Optimal Actions
            optimal_actions = d['Optimal Actions']
            group_results[i]['Optimal Actions Count'] += len(optimal_actions)

            # Count specific predicates in 'Vital Action Predicates'
            for predicate in ['PlugIn', 'Open', 'Close', 'Cut', 'Wipe', 'Wash']:
                group_results[i][predicate] += d['Vital Action Predicates'].count(predicate)

            # Count specific objects in 'Vital Objects'
            for obj in ['kitchenknife', 'faucet', 'rag']:
                group_results[i][obj] += d['Vital Objects'].count(obj)

        # Update starting index for the next group
        start += count

    # Calculate averages and prepare data for DataFrame
    data = []
    for i, result in enumerate(group_results):
        data.append([
            result['Total Records'],  # Add record count
            result['Optimal Actions Count'] / result['Total Records'],
            result['PlugIn'] / result['Total Records'],
            result['Open'] / result['Total Records'],
            result['Close'] / result['Total Records'],
            result['Cut'] / result['Total Records'],
            result['Wipe'] / result['Total Records'],
            result['Wash'] / result['Total Records'],
            result['kitchenknife'] / result['Total Records'],
            result['faucet'] / result['Total Records'],
            result['rag'] / result['Total Records']
        ])

    # Calculate overall averages
    overall_result = {
        'Optimal Actions Count': sum([r['Optimal Actions Count'] for r in group_results]),
        'PlugIn': sum([r['PlugIn'] for r in group_results]),
        'Open': sum([r['Open'] for r in group_results]),
        'Close': sum([r['Close'] for r in group_results]),
        'Cut': sum([r['Cut'] for r in group_results]),
        'Wipe': sum([r['Wipe'] for r in group_results]),
        'Wash': sum([r['Wash'] for r in group_results]),
        'kitchenknife': sum([r['kitchenknife'] for r in group_results]),
        'faucet': sum([r['faucet'] for r in group_results]),
        'rag': sum([r['rag'] for r in group_results]),
        'Total Records': sum(counts)
    }

    overall_averages = [
        overall_result['Total Records'],  # Add total record count
        overall_result['Optimal Actions Count'] / overall_result['Total Records'],
        overall_result['PlugIn'] / overall_result['Total Records'],
        overall_result['Open'] / overall_result['Total Records'],
        overall_result['Close'] / overall_result['Total Records'],
        overall_result['Cut'] / overall_result['Total Records'],
        overall_result['Wipe'] / overall_result['Total Records'],
        overall_result['Wash'] / overall_result['Total Records'],
        overall_result['kitchenknife'] / overall_result['Total Records'],
        overall_result['faucet'] / overall_result['Total Records'],
        overall_result['rag'] / overall_result['Total Records']
    ]

    # Add overall averages to data
    data.append(overall_averages)

    # Create DataFrame with the new column for record counts
    df = pd.DataFrame(
        data,
        index=[f'Group{i + 1}' for i in range(num_groups)] + ['Overall'],
        columns=['Record Count', 'Optimal Actions Count', 'PlugIn', 'Open', 'Close', 'Cut', 'Wipe', 'Wash',
                 'kitchenknife', 'faucet', 'rag']
    )

    # Set pandas options to print the entire DataFrame
    pd.set_option('display.max_columns', None)
    pd.set_option('display.expand_frame_repr', False)

    # Print table
    print(df)
    print('------------------------------------')

# Example usage
# analyze_data_tabular(data1, [10, 10, 10, 10])



def find_from_small_act(goal):
    from btgym.envs.virtualhometext.exec_lib._base.VHTAction_small import VHTAction_small
    from btgym.utils.tools import collect_action_nodes
    from btgym.algos.bt_autogen.main_interface import BTExpInterface
    from btgym.algos.llm_client.tools import goal_transfer_str, act_format_records

    env = btgym.make("VHT-Small")
    cur_cond_set = env.agents[0].condition_set = {"IsRightHandEmpty(self)", "IsLeftHandEmpty(self)", "IsStanding(self)"}
    cur_cond_set |= {f'IsClose({arg})' for arg in VHTAction.CAN_OPEN}
    cur_cond_set |= {f'IsSwitchedOff({arg})' for arg in VHTAction.HAS_SWITCH}
    cur_cond_set |= {f'IsUnplugged({arg})' for arg in VHTAction.HAS_PLUG}
    big_actions = collect_action_nodes(env.behavior_lib)

    algo = BTExpInterface(env.behavior_lib, cur_cond_set=cur_cond_set,
                          priority_act_ls=[], key_predicates=[],
                          key_objects=[],
                          selected_algorithm="opt", mode="big",
                          llm_reflect=False, time_limit=10,
                          heuristic_choice=0)
    goal_set = goal_transfer_str(' & '.join(goal))
    expanded_num, planning_time_total, cost, error, act_num, current_cost, record_act_ls = \
        execute_algorithm(algo, goal_set, cur_cond_set)
    time_limit_exceeded = algo.algo.time_limit_exceeded

    success = not error and not time_limit_exceeded

    _priority_act_ls, key_predicates, key_objects = act_format_records(record_act_ls)
    priority_act_ls = record_act_ls

    BLUE = "\033[94m"
    RESET = "\033[0m"

    print(f"{BLUE}Try to use big space...{RESET}")

    print(f"{BLUE}success:{RESET}", success)
    print(f"{BLUE}goal:{RESET}", ' & '.join(goal))
    print(f"{BLUE}_priority_act_ls:{RESET}", _priority_act_ls)
    print(f"{BLUE}act_num:{RESET}", act_num)
    print(f"{BLUE}planning_time_total:{RESET}", planning_time_total)
    print(f"{BLUE}expanded_num:{RESET}", expanded_num)
    print(f"{BLUE}current_cost:{RESET}", current_cost)

    # print("key_predicates:", key_predicates)
    # print("key_objects:", key_objects)
    # print("cost:", cost)
    # print("priority_act_ls:", priority_act_ls)
    # print("act_num:", act_num)
    # print("error:", error)
    # print("time_limit_exceeded:", time_limit_exceeded)
    # print("current_cost:", current_cost)
    # print("expanded_num:", expanded_num)
    # print("planning_time_total:", planning_time_total)
    return success, priority_act_ls, key_predicates, key_objects, \
        act_num, error, time_limit_exceeded, current_cost, expanded_num, planning_time_total,len(algo.actions)

# find_from_small_act(['IsIn_apple_microwave & IsClose_microwave & IsSwitchedOn_microwave']) #,'IsCut_breadslice'
# find_from_small_act(['IsIn_apple_microwave & IsIn_cutlets_microwave']) #,'IsCut_breadslice'
# find_from_small_act(['IsOn_kitchenknife_kitchentable'])