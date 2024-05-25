import pickle
import json
# 替换'file_path.pkl'为你的.pkl文件路径
# file_path = 'graph_input.pkl'
# 打开文件并加载内容
# with open(file_path, 'rb') as file:
#     data = pickle.load(file)

prop_obj_dic={'SURFACES':set(),'SITTABLE':set(), # put、sit
              'CAN_OPEN':set(),'CONTAINERS':set(), # close、putin
              'GRABBABLE':set(),'HAS_SWITCH':set(),  # grab、switch
               # 'MOVABLE':[] # push pull move
              "HAS_PLUG":set(), #  PlugIn,PlugOut
              "CUTABLE" : set(), 'EATABLE':set(), # Cut
                'RECIPIENT':set(), 'POURABLE':set(), 'DRINKABLE':set(), # Pour
              }

from btgym.utils import ROOT_PATH

for env_index in range(50):

    graph_path = f'{ROOT_PATH}\\..\\test\\env_graphs_json\\{env_index}_m_graph.json'

    # Open the JSON file and parse the JSON content into a dictionary
    with open(graph_path, 'r') as file:
        data = json.load(file)
    # 确保加载的数据是字典类型
    if isinstance(data, dict):
        print("成功读取字典:",env_index)
        # print(data)
    else:
        print("文件内容不是字典",env_index)

    data = data['nodes']

    for obj in data:
        for prop in obj.get('properties', []):
            if prop in prop_obj_dic:
                prop_obj_dic[prop].add(obj['class_name'])
    # print(prop_obj_dic)


# Wipe,Rinse,Scrub,Wash,  PlugIn,PlugOut,Cut,Pour,Squeeze, Find

# House Cleaning
# wipe / put back放回去 wipe前提是拿着 清洁物品 然后才能清洁
# 清洁物品：rag/duster/paper_towel/brush

# House Arrangement
# Food Preparation
import json
def save_json_with_newlines(data, filename):
    with open(filename, 'w') as file:
        file.write('{\n')
        last_key = list(data.keys())[-1]
        for key, value in data.items():
            # Serialize the list value to a JSON string
            value_str = json.dumps(value)
            # Check if this is the last item to avoid a comma at the end
            if key == last_key:
                file.write(f'"{key}":{value_str}\n')
            else:
                file.write(f'"{key}":{value_str},\n')
        file.write('}')


# Convert sets to lists for JSON serialization
for key in prop_obj_dic:
    prop_obj_dic[key] = list(prop_obj_dic[key])
save_json_with_newlines(prop_obj_dic, 'categorized_objects.json')
# with open('categorized_objects.json', 'w') as file:
#     json.dump(prop_obj_dic, file, indent=4)
    # json.dump(prop_obj_dic, file, separators=(',', ':'), end='\n')
print("The data has been saved to 'categorized_objects.json'.")