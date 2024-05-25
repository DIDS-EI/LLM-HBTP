import random
import numpy as np
import pandas as pd
import pickle


def count_accuracy(expected, actual):
    correct = 0
    incorrect = 0

    # 统计正确个数和错误个数
    for item in actual:
        if item in expected:
            correct += 1
        else:
            incorrect += 1

    # 计算正确率和错误率
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

    # 识别缺失和预测错误的动作/谓词/对象
    missing_act = set(d['Optimal Actions']) - set(priority_act_ls)
    incorrect_act = set(priority_act_ls) - set(d['Optimal Actions'])

    missing_predicate = set(d['Vital Action Predicates']) - set(key_predicates)
    incorrect_predicate = set(key_predicates) - set(d['Vital Action Predicates'])

    missing_obj = set(d['Vital Objects']) - set(key_objects)
    incorrect_obj = set(key_objects) - set(d['Vital Objects'])

    # 打印结果，如果集合不为空
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

# 示例使用
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
    print('----------  数据信息 ----------------')
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


# Function to create a custom action list based on action_space_size and provided action list `a`
def generate_custom_action_list(big_actions,action_space_size, a):
    # Get all available actions from a larger behavior library
    # big_actions = collect_action_nodes(behavior_lib)

    # Map action names to action objects for faster lookup
    name_to_action = {action.name: action for action in big_actions}

    # Find the action objects in `big_actions` that correspond to names in `a`
    selected_actions = []
    for action_name in a:
        if action_name not in name_to_action:
            raise ValueError(f"Action '{action_name}' is not available in the behavior library.")
        selected_actions.append(name_to_action[action_name])

    # Randomly select actions from the big action space excluding those already in `a`
    remaining_actions = [action for action in big_actions if action not in a]
    sorted_remaining_actions = sorted(remaining_actions, key=lambda x: x.name)

    # 这个地方 不排序就随机
    random_actions = random.sample(sorted_remaining_actions, action_space_size - len(a))

    # Combine selected actions with randomly chosen ones
    final_action_list = selected_actions + random_actions

    return final_action_list


def generate_custom_action_list_from_obj(big_actions,action_space_size, a,obj_ls):
    # Get all available actions from a larger behavior library
    # big_actions = collect_action_nodes(behavior_lib)

    # Map action names to action objects for faster lookup
    name_to_action = {action.name: action for action in big_actions}

    # Find the action objects in `big_actions` that correspond to names in `a`
    selected_actions = []
    for action_name in a:
        if action_name not in name_to_action:
            raise ValueError(f"Action '{action_name}' is not available in the behavior library.")
        selected_actions.append(name_to_action[action_name])

    # Randomly select actions from the big action space excluding those already in `a`
    remaining_actions = [action for action in big_actions if action not in a]


    obj_filtered_actions = []
    # 遍历 big_actions 并检查 name 是否包含 obj_ls 中的任意一个物体
    for action in big_actions:
        if all(obj in action.name for obj in obj_ls) and (action.name not in a):
            obj_filtered_actions.append(action)
    sorted_remaining_actions = sorted(remaining_actions, key=lambda x: x.name)

    # 这个地方 不排序就随机
    if action_space_size - len(a)-len(obj_filtered_actions)>0:
        random_actions = random.sample(sorted_remaining_actions, action_space_size - len(a)-len(obj_filtered_actions))
    else:
        random_actions = random.sample(obj_filtered_actions, action_space_size- len(a) )
    # Combine selected actions with randomly chosen ones
    final_action_list = selected_actions + random_actions +obj_filtered_actions

    return final_action_list