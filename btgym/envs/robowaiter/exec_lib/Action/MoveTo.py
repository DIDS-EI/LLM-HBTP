from btgym.envs.robowaiter.exec_lib._base.VHTAction import VHTAction

class MoveTo(VHTAction):
    can_be_expanded = True
    num_args = 1
    valid_args = VHTAction.all_object | VHTAction.tables_for_placement | VHTAction.tables_for_guiding

    def __init__(self, *args):
        super().__init__(*args)

    @classmethod
    def get_info(cls,*arg):
        info = {}
        info['pre'] = set()
        if arg[0] in VHTAction.all_object:
            info['pre'] |= {f'Exists({arg[0]})'}

        info["add"] = {f'RobotNear({arg[0]})'}
        info["del_set"] = {f'RobotNear({place})' for place in cls.valid_args if place != arg[0]}

        info['cost'] = 15
        return info

    def change_condition_set(self):
        self.agent.condition_set |= (self.info["add"])
        self.agent.condition_set -= self.info["del_set"]

        # self.agent.condition_set.add(f"IsSwitchedOn({self.args[0]})")