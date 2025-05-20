from btgym import BehaviorTree
from btgym import ExecBehaviorLibrary
import btgym

# behavior_tree = BehaviorTree("Default.btml")
# behavior_tree.print()
# behavior_tree.draw()

lib_path = f'{btgym.ROOT_PATH}/exec_lib'
exec_lib = ExecBehaviorLibrary(lib_path)
print(exec_lib.Action)

exec_bt = ExecBehaviorTree("Default.btml",exec_lib)
