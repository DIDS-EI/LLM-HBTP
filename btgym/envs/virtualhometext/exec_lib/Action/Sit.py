import py_trees as ptree

from typing import Any
from btgym.envs.virtualhometext.exec_lib._base.VHTAction import VHTAction
from btgym.behavior_tree import Status
from btgym.envs.virtualhometext.exec_lib._base.VHTAction_small import VHTAction_small

class Sit(VHTAction):
    can_be_expanded = False
    num_args = 1
    valid_args=VHTAction.SITTABLE
    valid_args_small = VHTAction_small.SITTABLE

    def __init__(self, *args):
        super().__init__(*args)

    @classmethod
    def get_info(cls,*arg):
        info = {}
        info["pre"]={"IsStanding(self)",f"IsNear(self,{arg[0]})"}
        info["add"]={f"IsSittingOn(self,{arg[0]})",f"IsSitting(self)"}
        info["del_set"] = {f"IsStanding(self)"}
        info["cost"] = 15
        return info

    def change_condition_set(self):
        self.agent.condition_set |= (self.info["add"])
        self.agent.condition_set -= self.info["del_set"]