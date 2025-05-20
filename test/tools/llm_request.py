import time

from btgym.algos.llm_client.llms.gpt3 import LLMGPT3
from btgym import ExecBehaviorLibrary
import btgym


# behavior_tree = BehaviorTree("Default.btml")
# behavior_tree.print()
# behavior_tree.draw()

# lib_path = f'{btgym.ROOT_PATH}/exec_lib'
# exec_lib = ExecBehaviorLibrary(lib_path)
# print(exec_lib.Action)

# exec_bt = ExecBehaviorTree("Default.btml",exec_lib)

llm = LLMGPT3()
answer = llm.request(question="are you gpt 3.5? or gpt 4?")
print(answer)