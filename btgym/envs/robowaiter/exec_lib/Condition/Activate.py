from btgym.envs.robowaiter.exec_lib._base.VHTCondition import VHTCondition

class Activate(VHTCondition):
    can_be_expanded = True
    num_args = 1
    valid_args = {'AC','TubeLight','HallLight'}

