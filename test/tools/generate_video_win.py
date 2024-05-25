# Generate video for a program. Make sure you have the executable open

from btgym.envs.virtualhome.simulation.unity_simulator.comm_unity import UnityCommunication
from btgym.envs.virtualhome.tools import add_object_to_scene

# script1 = ['<char0> [Walk] <tv> (1)','<char0> [switchon] <tv> (1)'] # Add here your script

import subprocess

# file_name =  'D:\Workspace\BaiduSyncdisk\CXL_Storage\Code\windows_exec.v2.2.4\VirtualHome.exe'
file_name = '/simulators/virtualhome/windows/VirtualHome.exe'

# 启动 exe 文件
process = subprocess.Popen(file_name, stdout=subprocess.PIPE, stderr=subprocess.PIPE)


# script = ['<char0> [Walk] <tv> (1)',
#           '<char0> [switchon] <tv> (1)',
#           '<char0> [Walk] <sofa> (1)',
#           '<char0> [Sit] <sofa> (1)',
#           '<char0> [Watch] <tv> (1)']

# 准备早餐
# script =  [
#     # '<char0> [Walk] <fridge> (1)',
#     # '<char0> [Open] <fridge> (1)',
#     # '<char0> [Walk] <milk> (400)',
#     # '<char0> [Grab] <milk> (400)',
#     # '<char0> [Close] <fridge> (1)',
#     # '<char0> [Put] <milk> (400) <desk> (1)'
#
#
#     # '<char0> [Walk] <kitchentable> (1)',
#     # '<char0> [Open] <fridge> (1)',
# '<char0> [Walk] <desk> (2)',
#     '<char0> [Walk] <mug> (257)',
#     '<char0> [Grab] <mug> (257)',
#     # '<char0> [Close] <fridge> (1)', #kitchencabinet=130
#     # '<char0> [Put] <mug> (405) <desk> (1)'
#
# '<char0> [Put] <mug> (257) <desk> (2)'
#
#     ]



# test
script = [
'<char0> [walk] <washingmachine> (1)',
    '<char0> [PlugIn] <washingmachine> (1)',
    '<char0> [switchon] <washingmachine> (1)',

]






print('Starting Unity...')
comm = UnityCommunication()


print('Starting scene...')
scene_name=0
simulator_launched = False
while not simulator_launched:
    try:
        comm.reset(6) # 0-49
        simulator_launched = True
    except:
        pass



# add some objects
# add_object_to_scene(comm, object_id, class_name, target_name, target_id=None, relat_pos=[0,0,0],\
#                         category=None,position=None, properties=None, rotation=[0.0, 0.0, 0.0, 1.0], scale=[1.0, 1.0, 1.0]):
# _, env_g = comm.environment_graph()
# properties = ['GRABBABLE', 'DRINKABLE', 'POURABLE', 'CAN_OPEN', 'MOVABLE']
# add_object_to_scene(comm=comm, object_id=410, class_name='Milk', target_name="fridge", relat_pos=[0, 0, 0], properties=properties)
#
#
# properties = ["GRABBABLE", "RECIPIENT", "POURABLE", "MOVABLE"]
# add_object_to_scene(comm=comm, object_id=405, class_name='Milk',target_name="kitchentable", relat_pos=[0, 0.05, 0], properties=properties)


# properties = ["GRABBABLE", "RECIPIENT", "CAN_OPEN", "MOVABLE"]
# add_object_to_scene(comm=comm,object_id=405, class_name='coffeepot', target_name="kitchentable", relat_pos=[0, 0, 0], properties=properties)
# add_object_to_scene(comm=comm,object_id=405, class_name='Apple', target_name="kitchencabinet", position=[0, 0, 0],\
#                     scale=[0.983430862, 0.983431935, 0.9834311], properties=properties) #Cereal

# coffeepot

_, env_g = comm.environment_graph()
# del_no = None
for no in env_g['nodes']:
    if no["class_name"]=="mug":
        print(no)
        break
# print("del_no:",del_no)
#
# for no in env_g['edges']:
#     if no["from_id"]==193 or no["to_id"]==193 :
#         env_g['edges'].remove(no)


# new_object = {
#     'id': 410,
#     'category': 'Food',
#     # You might want to make this a parameter if you plan to add non-food items.
#     'class_name': 'Cereal', #Coffee_pot poundcake
#     'prefab_name': f'SMGP_PRE_Cereal_1024',
#     # Assuming the prefab name follows a specific pattern; adjust as needed.
#     'obj_transform': {
#         # 'position': [-6.04132843, 0.982, 2.32381487],
#         'position': [0,0,0],
#         'rotation': [0,0,0,1],
#         'scale':  [0.252370358,0.2523704,0.252370328]
#     },
#     'bounding_box': {
#         # 'center': [-6.04132843, 1.12907231, 2.323814],
#         'center': [0,0,0],
#         'size': [0.237073123, 0.294145018, 0.08550128]  # You might need a way to set this based on the object.
#     },
#     'properties':  ["GRABBABLE", "EATABLE", "MOVABLE"],
#     'states': []  # Assuming default state; adjust as needed.
# }
#
# # Define the relation
# new_relation = {
# "from_id": 410,
# "to_id": 55, #kitchen
# "relation_type": "INSIDE"
# }
# new_relation2 = {
# "from_id": 410,
# "to_id": 127, #kitchentable
# "relation_type": "ON"
# }
#
# # Add the new object and relation to the environment graph
# env_g['nodes'].append(new_object)
# env_g['edges'].append(new_relation)
# env_g['edges'].append(new_relation2)

# Expand the scene with the new object
# success, message = comm.expand_scene(env_g,randomize=True)
# print(f"Expansion result: {success}, {message}")






# comm.add_character('Chars/Female1')
# comm.add_character('Chars/Male1',initial_room="kitchen")
comm.add_character('Chars/Male2')

print('Generating video...')
for script_instruction in script:
    # success, message = comm.render_script([script_instruction], recording=True,frame_rate=10,camera_mode=["FIRST_PERSON"],find_solution=True) #AUTO FIRST_PERSON
    # success, message = comm.render_script([script_instruction], recording=True,frame_rate=10,camera_mode=["AUTO"],find_solution=True) #AUTO FIRST_PERSON
    success, message = comm.render_script([script_instruction], recording=True,frame_rate=10,camera_mode=["PERSON_FROM_BACK"],find_solution=True) #AUTO FIRST_PERSON

    # 检查指令是否成功执行
    if success:
        print(f"'Successfully: {script_instruction}'.")
    else:
        print(f"'Failed {script_instruction},{message}'.")

    # _, env_g = comm.environment_graph()
    # for no in env_g['nodes']:
    #     if no["id"] == 260:
    #         print(no)
    #         print("pos:",no['obj_transform']['position'])

# comm.render_script(script, recording=True,frame_rate=10,camera_mode=["PERSON_FROM_BACK"],find_solution=True)

# comm.render_script(script1, recording=True, find_solution=True)
# # time.sleep(2)
# comm.render_script(script2, recording=True, find_solution=True)

print('Generated, find video in simulation/unity_simulator/output/')
