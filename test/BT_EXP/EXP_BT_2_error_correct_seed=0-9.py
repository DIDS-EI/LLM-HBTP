import copy
import time

from btgym import BehaviorTree
import btgym
from btgym.utils import ROOT_PATH
from btgym.algos.bt_autogen.main_interface import BTExpInterface
from btgym.envs.virtualhometext.exec_lib._base.VHTAction_small import VHTAction_small
from btgym.envs.virtualhometext.exec_lib._base.VHTAction import VHTAction
from btgym.algos.bt_autogen.tools import state_transition
from btgym.algos.llm_client.tools import goal_transfer_str, act_str_process
import random
import numpy as np
import pandas as pd


from btgym.utils.tools import collect_action_nodes
from btgym.utils.read_dataset import read_dataset
from tools import count_accuracy, identify_and_print_diffs, analyze_data_tabular,generate_custom_action_list
import pickle
all_start_time = time.time()


# env = btgym.make("VHT-Small")
env = btgym.make("VHT-PutMilkInFridge")
cur_cond_set = env.agents[0].condition_set = {"IsRightHandEmpty(self)", "IsLeftHandEmpty(self)", "IsStanding(self)"}
cur_cond_set |= {f'IsClose({arg})' for arg in VHTAction.CAN_OPEN}
cur_cond_set |= {f'IsSwitchedOff({arg})' for arg in VHTAction.HAS_SWITCH}
cur_cond_set |= {f'IsUnplugged({arg})' for arg in VHTAction.HAS_PLUG}
big_actions = collect_action_nodes(env.behavior_lib)

# 导入数据
data_path = f"{ROOT_PATH}/../test/BT_EXP/DATA_BT_100_ori_yz_revby_cys.txt" #DATA_BT_100_ori_yz_revby_cys
data = read_dataset(data_path)
len_data = len(data)
print(f"导入 {len_data} 条数据")
print(data[0])
analyze_data_tabular(data,[47,1,1,1])


error_rate_range_ls = [0, 0.5 ,1, 3, 5]
# error_rate_range_ls = [0, 0.5 ,1, 3, 5]
# error_rate_range_ls = [0,1,5,10]
# error_rate_range_ls = [15]
# error_rate_range_ls = [1] #5-80
# correct_rate_range_ls = [0,  1]
# correct_rate_range_ls = [0, 0.25, 0.5, 0.75, 1]
# correct_rate_range_ls = [1]
# correct_rate_range_ls = [1]
correct_rate_range_ls = np.arange(0, 1.1, 0.2)  # 注意：1.1是因为arange不包含终止值
# correct_rate_range_ls = [0]

for heuristic_choice in [1]:

    for seed in range(49,50):
        # seed = 0
        random.seed(seed)
        np.random.seed(seed)



        # Data storage dictionary
        data_storage_dic = {(e, c): {'id':[],'error_num': [], 'correct_num': [], \
                                     'expanded_num': [], 'planning_time_total': [], \
                                     'current_cost': [], \
                                     'error':[], 'timeout':[],\
                                     'act_steps': [], 'bt_status': []}
                            for e in error_rate_range_ls for c in correct_rate_range_ls}

        for ind,d in enumerate(data):
            # for data_ind in range(4,5):
            combined_string = ' & '.join(d['Goals'])
            goal_set = goal_transfer_str(combined_string)
            true_priority_act_set = set(act_str_process(d['Optimal Actions'], already_split=True))
            print("\ndata:", ind)
            print("goal:", goal_set)
            print("act:", true_priority_act_set)

            # 随机选择动作这里存在较大的随机性 ？？？！！！
            custom_action_list = generate_custom_action_list(big_actions, 80, true_priority_act_set)
            custom_action_name_ls = {act.name for act in custom_action_list}

            error_priority_act_set = custom_action_name_ls - true_priority_act_set

            cur_cond_set = env.agents[0].condition_set = {"IsRightHandEmpty(self)", "IsLeftHandEmpty(self)",
                                                          "IsStanding(self)"}
            cur_cond_set |= {f'IsClose({arg})' for arg in VHTAction.CAN_OPEN}
            cur_cond_set |= {f'IsSwitchedOff({arg})' for arg in VHTAction.HAS_SWITCH}
            cur_cond_set |= {f'IsUnplugged({arg})' for arg in VHTAction.HAS_PLUG}

            error_act_tmp_set=set()
            for error_rate in error_rate_range_ls:
                corr_act_tmp_set = set()
                for correct_rate in correct_rate_range_ls:

                    error_num = int(len(true_priority_act_set) * error_rate)
                    correct_num = int(len(true_priority_act_set) * correct_rate)
                    # correct_num=len(true_priority_act_set)-error_num
                    print("------Error:", error_num,"Correct:", correct_num,'---------')

                    error_act_tmp_set |= set(random.sample(sorted(list(error_priority_act_set-error_act_tmp_set)), \
                                                                error_num-len(error_act_tmp_set)))
                    corr_act_tmp_set |= set(random.sample(sorted(list(true_priority_act_set-corr_act_tmp_set)), \
                                                                correct_num-len(corr_act_tmp_set)))
                    # 排序是为了取消随机性
                    # 推荐优先级
                    priority_act_ls = set()
                    # priority_act_ls |= set(random.sample(sorted(list(error_priority_act_set)), error_num))
                    # priority_act_ls |= set(random.sample(sorted(list(true_priority_act_set)), correct_num))
                    priority_act_ls |= error_act_tmp_set # 采用逐步增加的方法
                    priority_act_ls |= corr_act_tmp_set
                    priority_act_ls = sorted(list(priority_act_ls))

                    # priority_act_ls=error_priority_act_set|true_priority_act_set

                    print("data:", ind,"priority_act_ls:", priority_act_ls)
                    # {'LeftPut(cupcake,floor)', 'Wipe(juice)', 'PlugIn(toaster)', 'LeftPut(towel,radio)', 'Walk(bathroomcabinet)', 'LeftPut(bellpepper,kitchencounter)'}
                    # priority_act_ls = {'RightGrab(pillow)', 'LeftPut(cuttingboard,kitchentable)', 'LeftPut(cuttingboard,sofa)', 'RightPut(boardgame,sofa)', 'LeftPut(cutleryknife,desk)', 'Walk(book)'}
                    # priority_act_ls = []

                    # cur_cond_set = set(env_dic[data[0]['Environment']])
                    # print("cur_cond_set:",cur_cond_set)

                    priority_obj_ls = []
                    # 这里写错了 baseline 应该是不推荐的
                    # algo = BTExpInterface(None, cur_cond_set, priority_act_ls, priority_obj_ls,\
                    #                       selected_algorithm="baseline",action_list=action_list)

                    # 这里应该是推荐错了怎么处理
                    algo = BTExpInterface(env.behavior_lib, cur_cond_set=copy.deepcopy(cur_cond_set), \
                                          priority_act_ls=priority_act_ls,  \
                                          selected_algorithm="opt", mode="user-defined", action_list=custom_action_list,\
                                          llm_reflect=False, time_limit=None,
                                          heuristic_choice=heuristic_choice)

                    start_time = time.time()
                    algo.process(goal_set)
                    end_time = time.time()

                    ptml_string, cost, expanded_num = algo.post_process()
                    planning_time_total = (end_time - start_time)

                    goal = goal_set[0]
                    state = copy.deepcopy(cur_cond_set)
                    error, state, act_num, current_cost, record_act_ls = algo.execute_bt(goal, state, verbose=False)

                    print("current_cost",current_cost,"expanded_num:", expanded_num, "planning_time_total", planning_time_total)
                    time_limit_exceeded = algo.algo.time_limit_exceeded
                    # algo.algo.clear()

                    # 数据记录
                    # Store the data
                    data_storage_dic[(error_rate, correct_rate)]['id'].append(ind)

                    data_storage_dic[(error_rate, correct_rate)]['error_num'].append(error_num)
                    data_storage_dic[(error_rate, correct_rate)]['correct_num'].append(correct_num)

                    data_storage_dic[(error_rate, correct_rate)]['expanded_num'].append(expanded_num)
                    data_storage_dic[(error_rate, correct_rate)]['planning_time_total'].append(planning_time_total)
                    data_storage_dic[(error_rate, correct_rate)]['current_cost'].append(current_cost)

                    data_storage_dic[(error_rate, correct_rate)]['error'].append(int(error))
                    data_storage_dic[(error_rate, correct_rate)]['timeout'].append(int(time_limit_exceeded))

                    data_storage_dic[(error_rate, correct_rate)]['act_steps'].append(planning_time_total)
                    data_storage_dic[(error_rate, correct_rate)]['bt_status'].append(current_cost)

        # Preparing the detailed data for CSV
        all_data_records = []
        summary_results = []
        for key, values in data_storage_dic.items():
            error_rate, correct_rate = key
            for i in range(len(values['expanded_num'])):
                record = {
                    'Data ID': values['id'][i],
                    'Error Rate': error_rate,
                    'Correct Rate': correct_rate,
                    'Error Num': values['error_num'][i],
                    'Correct Num': values['correct_num'][i],
                    'Expanded Num': values['expanded_num'][i],
                    'Planning Time Total': values['planning_time_total'][i],
                    'Current Cost': values['current_cost'][i],

                    'Error':values['error'][i],
                    'Timeout':values['timeout'][i],

                    'Act Steps': values['act_steps'][i],
                    'BT Status': values['bt_status'][i]
                }
                all_data_records.append(record)

            error_rate, correct_rate = key
            total_error_num = np.mean(values['error_num'])
            total_correct_num = np.mean(values['correct_num'])
            total_expanded_num = np.mean(values['expanded_num'])
            total_planning_time = np.mean(values['planning_time_total'])
            total_current_cost = np.mean(values['current_cost'])
            total_act_steps = np.mean(values['act_steps'])
            total_bt_status = np.mean(values['bt_status'])

            summary_results.append([
                error_rate, correct_rate, total_error_num, total_correct_num, total_expanded_num,
                total_planning_time, total_current_cost, total_act_steps, total_bt_status
            ])

        # Creating DataFrame for detailed and summary data
        df_details = pd.DataFrame(all_data_records, columns=[
            'Data ID',
            'Error Rate', 'Correct Rate', 'Error Num', 'Correct Num', 'Expanded Num',
            'Planning Time Total', 'Current Cost','Error','Timeout',
            'Act Steps', 'BT Status'
        ])

        df_summary = pd.DataFrame(summary_results, columns=[
            'Error Rate', 'Correct Rate', 'Total Error Num', 'Total Correct Num', 'Total Expanded Num',
            'Total Planning Time Total', 'Total Current Cost', 'Total Act Steps', 'Total BT Status'
        ])

        # Save detailed data and summary data to CSV
        # csv_file_path_details = 'EXP_2_output_detailed_bt_data_small_100_bigerror_0_1_5_10_heuristic_choice=1.csv'
        # csv_file_path_summary = 'EXP_2_output_summary_bt_data_small_100_bigerror_0_1_5_10_heuristic_choice=1.csv'
        csv_file_path_details = f'EXP_2_output_detailed_bt_data_small_100_bigerror_heuristic={heuristic_choice}_seed={seed}.csv'
        csv_file_path_summary = f'EXP_2_output_summary_bt_data_small_100_bigerror_heuristic={heuristic_choice}_seed={seed}.csv'


        df_details.to_csv(csv_file_path_details, index=False)
        df_summary.to_csv(csv_file_path_summary, index=False)

        print("Detailed data saved to", csv_file_path_details)
        print("Summary data saved to", csv_file_path_summary)

all_end_time = time.time()
total_time = (all_end_time - all_start_time)
print("Total Time:", total_time)

# 输出每条数据的
# 对称   0  0.3  0.5  0.7  1
# 数量
# 扩展节点数 时间
# 每条数据有5条这样的，然后保存为 csv

# 求出平均的一个结果，形成一张表
# 0    正确数 错误数  平均总数 expanded_num  time
# 0.3
# 0.5
# 0.7
# 1

# 分开算情况下
# 错误率占比0，正确从0-1
# 错误率占比0.3，正确从0-1
# 错误率占比0，5正确从0-1
# 得到结果
