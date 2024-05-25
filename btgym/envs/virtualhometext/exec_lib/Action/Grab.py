from btgym.envs.virtualhometext.exec_lib._base.VHTAction import VHTAction
from btgym.envs.virtualhometext.exec_lib._base.VHTAction_small import VHTAction_small

class Grab(VHTAction):
    can_be_expanded = False
    num_args = 1
    valid_args = VHTAction.GRABBABLE
    valid_args_small = VHTAction_small.GRABBABLE

    def __init__(self, *args):
        super().__init__(*args)
