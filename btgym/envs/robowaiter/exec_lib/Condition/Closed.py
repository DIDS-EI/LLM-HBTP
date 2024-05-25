from btgym.envs.robowaiter.exec_lib._base.VHTCondition import VHTCondition

class Closed(VHTCondition):
    can_be_expanded = True
    num_args = 1
    valid_args = {'Curtain'}
