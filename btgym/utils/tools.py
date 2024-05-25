
from btgym.algos.bt_autogen.Action import Action
from btgym.utils.read_dataset import read_dataset
from btgym.utils import ROOT_PATH
from btgym.envs.virtualhometext.exec_lib._base.VHTAction import VHTAction
import pickle


# 读入环境文件
def read_env_file(file_path):
    env_dict = {}
    current_key = None
    current_values = []

    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            line = line.strip()
            if line:
                if '#' in line:
                    parts = line.split('#', 1)
                    current_key = parts[0].strip()
                    current_values = []
                else:
                    current_values.extend(line.split(', '))
                    env_dict[int(current_key)] = set(current_values)
    return env_dict


# 导入不同的环境
import re
def extract_objects(actions):
    pattern = re.compile(r'\w+\(([^)]+)\)')
    objects = []
    for action in actions:
        match = pattern.search(action)
        if match:
            objects.append(match.group(1))
    return objects
def collect_action_nodes(behavior_lib):
    action_list = []

    for cls in behavior_lib["Action"].values():
        if cls.can_be_expanded:
            # print(f"可扩展动作：{cls.__name__}, 存在{len(cls.valid_args)}个有效论域组合")
            if cls.num_args == 0:
                action_list.append(Action(name=cls.get_ins_name(), **cls.get_info()))
            if cls.num_args == 1:
                for arg in cls.valid_args:
                    action_list.append(Action(name=cls.get_ins_name(arg), **cls.get_info(arg)))
            if cls.num_args > 1:
                for args in cls.valid_args:
                    action_list.append(Action(name=cls.get_ins_name(*args), **cls.get_info(*args)))

    print(f"共收集到{len(action_list)}个实例化动作:")
    # for a in self.action_list:
    #     if "Turn" in a.name:
    #         print(a.name)
    print("--------------------\n")

    return action_list


def refresh_VHT_samll_data():
    # 读入数据集合
    data_path = f"{ROOT_PATH}/../test/dataset/data0429.txt"
    data = read_dataset(data_path)
    data_num = len(data)
    print(f"导入 {data_num} 条数据")
    print(data[0])

    # 数据集中涉及的所有物体集合
    objs=set()
    for d in data:
        objs |= set(d['Key_Object'])

    categories = ['SURFACES', 'SITTABLE', 'CAN_OPEN', 'CONTAINERS', 'GRABBABLE', 'cleaning_tools', \
             'cutting_tools', 'HAS_SWITCH', 'HAS_PLUG', 'CUTABLE', 'EATABLE', 'WASHABLE', 'RECIPIENT', \
             'POURABLE', 'DRINKABLE']
    categories_objs_dic={}
    for ctg in categories:
        categories_objs_dic[ctg] = getattr(VHTAction, ctg)
        categories_objs_dic[ctg] &= objs


    ctg_objs_path = f"{ROOT_PATH}/../test/EXP/ctg_objs.pickle"
    # 打开一个文件用于写入，注意'b'表示二进制模式
    with open(ctg_objs_path, 'wb') as file:
        # 使用pickle.dump()函数将数据写入文件
        pickle.dump(categories_objs_dic, file)


def save_data_txt(output_path,data1):
    # Open the file for writing
    with open(output_path, "w", encoding="utf-8") as f:
        # Loop through each entry in data1 and write the required information
        for idx, entry in enumerate(data1, start=1):
            f.write(f"{idx}\n")
            f.write(f"Environment:{entry['Environment']}\n")
            f.write(f"Instruction: {entry['Instruction']}\n")
            # Use ' & ' to join goals, assuming this is the correct separator
            f.write(f"Goals: {' & '.join(entry['Goals'])}\n")
            # Join actions with a comma
            f.write(f"Actions: {', '.join(entry['Actions'])}\n")
            # Join key predicates with a comma
            f.write(f"Vital Action Predicates: {', '.join(entry['Vital Action Predicates'])}\n")
            # Ensure Key_Object is a list and join it with commas
            key_objects = entry['Vital Objects']
            if isinstance(key_objects, list):
                f.write(f"Vital Objects: {', '.join(key_objects)}\n\n")
            else:
                f.write(f"Vital Objects: {key_objects}\n\n")

    print(f"Data saved to {output_path}")



