import pandas as pd
import json
import  time
from btgym.utils import ROOT_PATH

def analyze_reflect_results(name,reflect_record_reflect):
    # 准备基础列
    base_columns = ['id', 'Instruction', 'Goals', 'Optimal Actions', 
                   'Vital Action Predicates', 'Vital Objects', 
                   'reflect_time', 'success', 'time_limit_exceeded',
                   'action_space', 'expanded_num','actions_path_len', 'planning_time_total']
    
    # 找出所有可能的reflect_time结果列
    reflect_result_columns = set()
    for record in reflect_record_reflect:
        reflect_result_columns.update([k for k in record.keys() if k.endswith('_result')])
    
    # 合并所有列
    all_columns = base_columns + sorted(list(reflect_result_columns))
    
    # 创建DataFrame
    df = pd.DataFrame(reflect_record_reflect)
    
    # 保存CSV
    # df.to_csv('reflect_analysis_results.csv', index=False)
    time_str = time.strftime('%Y%m%d_%H%M', time.localtime())
    df.to_csv(f'{ROOT_PATH}/../test/table2_vh/saved/reflect_analysis_results_{name}_{time_str}.csv', index=False)

    # 初始化每个reflect阶段的成功和总数计数器
    success_counts = {i: 0 for i in range(4)}  # 0-3
    total_counts = {i: 0 for i in range(4)}    # 0-3
    
    # 遍历每个案例
    for _, case in df.iterrows():
        final_reflect_time = case['reflect_time']
        final_success = case['success']
        
        # 更新每个reflect阶段的计数
        for rt in range(4):
            total_counts[rt] += 1
            
            if final_success:
                # 如果最终成功，则根据成功时的reflect_time判断各阶段是否成功
                if rt >= final_reflect_time:
                    success_counts[rt] += 1
            # 如果最终失败，则所有阶段都计为失败，无需特别处理
    
    # 计算各阶段的成功率
    success_rates = {rt: success_counts[rt]/total_counts[rt] if total_counts[rt] > 0 else 0 
                    for rt in range(4)}
    
    # 计算成功案例的平均指标
    success_metrics = df[df['success'] == True].agg({
        'action_space': 'mean',
        'expanded_num': 'mean',
        'planning_time_total': 'mean',
        'actions_path_len': 'mean'
    }).to_dict()
    
    return {
        'success_rates': success_rates,
        'success_metrics': success_metrics,
        'success_counts': success_counts,
        'total_counts': total_counts
    }

# 打印分析结果
def print_analysis(analysis_results):
    print("\n=== 不同reflect_time的成功率 ===")
    for rt, rate in analysis_results['success_rates'].items():
        success_count = analysis_results['success_counts'][rt]
        total_count = analysis_results['total_counts'][rt]
        print(f"reflect_time={rt}: {rate*100:.2f}% ({success_count}/{total_count})")
    
    print("\n=== 成功案例的平均指标 ===")
    metrics = analysis_results['success_metrics']
    print(f"平均action_space: {metrics['action_space']:.2f}")
    print(f"平均expanded_num: {metrics['expanded_num']:.2f}")
    print(f"平均planning_time_total: {metrics['planning_time_total']:.4f}") 
    print(f"平均actions_path_len: {metrics['actions_path_len']:.2f}")