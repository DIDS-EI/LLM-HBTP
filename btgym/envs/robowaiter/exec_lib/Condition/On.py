from btgym.envs.robowaiter.exec_lib._base.VHTCondition import VHTCondition
from btgym.envs.robowaiter.exec_lib._base.VHTAction import VHTAction
class On(VHTCondition):
    can_be_expanded = True
    num_args = 1
    valid_args = [tuple(VHTAction.all_object),
            tuple(VHTAction.tables_for_placement)]

