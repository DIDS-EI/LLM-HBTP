import btgym
from btgym.utils.tools import collect_action_nodes


# # ================== RW ===============
name = "RW"
from btgym.envs.RoboWaiter.exec_lib._base.RWAction import RWAction
env = btgym.make("RWEnv")
cur_cond_set = env.agents[0].condition_set = {'RobotNear(Bar)','Holding(Nothing)' }
cur_cond_set |= {f'Exists({arg})' for arg in RWAction.all_object-{'Coffee', 'Water', 'Dessert'}}
big_actions = collect_action_nodes(env.behavior_lib)
print(f" {len(RWAction.AllObject)} objects")

# ===================== VirtualHome ========================
# from btgym.envs.VirtualHome.exec_lib._base.VHAction import VHAction
#
# env = btgym.make("VH-PutMilkInFridge")
# cur_cond_set = env.agents[0].condition_set = {"IsRightHandEmpty(self)", "IsLeftHandEmpty(self)",
#                                               "IsStanding(self)"}
# cur_cond_set |= {f'IsClose({arg})' for arg in VHAction.CanOpenPlaces}
# cur_cond_set |= {f'IsSwitchedOff({arg})' for arg in VHAction.HasSwitchObjects}
# big_actions = collect_action_nodes(env.behavior_lib)
# print(f" {len(VHAction.AllObject)} objects")


# # ================== RHS ===============
# name = "RHS"
# from btgym.envs.RobotHow_Small.exec_lib._base.RHSAction import RHSAction
# env = btgym.make("VHT-Small")
# cur_cond_set = env.agents[0].condition_set = {"IsRightHandEmpty(self)", "IsLeftHandEmpty(self)", "IsStanding(self)"}
# cur_cond_set |= {f'IsClose({arg})' for arg in RHSAction.CAN_OPEN}
# cur_cond_set |= {f'IsUnplugged({arg})' for arg in RHSAction.HAS_PLUG}
# cur_cond_set |= {f'IsSwitchedOff({arg})' for arg in RHSAction.HAS_SWITCH}
# big_actions = collect_action_nodes(env.behavior_lib)
# print(f" {len(RHSAction.AllObject)} objects")


# ================== RHB ===============
# name = "RHB"
# from btgym.envs.RobotHow.exec_lib._base.RHAction import RHAction as RHB
# env = btgym.make("VHT-WatchTV")
# cur_cond_set = env.agents[0].condition_set = {"IsRightHandEmpty(self)", "IsLeftHandEmpty(self)", "IsStanding(self)"}
# cur_cond_set |= {f'IsClose({arg})' for arg in RHB.CAN_OPEN}
# cur_cond_set |= {f'IsSwitchedOff({arg})' for arg in RHB.HAS_SWITCH}
# cur_cond_set |= {f'IsUnplugged({arg})' for arg in RHB.HAS_PLUG}
# big_actions = collect_action_nodes(env.behavior_lib)
# print(f" {len(RHB.AllObject)} objects")