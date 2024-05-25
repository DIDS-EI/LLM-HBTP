from btgym.algos.bt_autogen.Action import Action, state_transition
from btgym.algos.bt_autogen.OptimalBTExpansionAlgorithm import OptBTExpAlgorithm
from btgym.algos.bt_autogen.OptimalBTExpansionAlgorithm_baseline import OptBTExpAlgorithm_BaseLine
from btgym.algos.bt_autogen.BTExpansionAlgorithm import BTalgorithm
from btgym.algos.bt_autogen.BTExpansionAlgorithmBFS import BTalgorithmBFS
from btgym.algos.bt_autogen.BTExpansionAlgorithmDFS import BTalgorithmDFS
from btgym.algos.bt_autogen.OptimalBTExpansionAlgorithmHeuristics import OptBTExpAlgorithmHeuristics
from btgym.algos.bt_autogen.examples import *
import re
import random
import numpy as np
seed=0
random.seed(seed)
np.random.seed(seed)

class BTExpInterface:
    def __init__(self, behavior_lib, cur_cond_set, priority_act_ls=[], key_predicates=[], key_objects=[], selected_algorithm="opt",
                 mode="big",
                 bt_algo_opt=True, llm_reflect=False, llm=None, messages=None, action_list=None,use_priority_act=True,time_limit=None,
                 heuristic_choice=-1):
        """
        Initialize the BTOptExpansion with a list of actions.
        :param action_list: A list of actions to be used in the behavior tree.
        """
        self.cur_cond_set = cur_cond_set
        self.bt_algo_opt = bt_algo_opt
        self.selected_algorithm = selected_algorithm
        self.time_limit=time_limit

        self.consider_priopity = False

        self.heuristic_choice = heuristic_choice


        if behavior_lib == None:
            self.actions = action_list
            self.big_actions=self.actions
        else:
            self.big_actions = collect_action_nodes(behavior_lib)


        if mode == "big":
            self.actions = self.big_actions
        elif mode=="user-defined":
            self.actions = action_list
            print(f"自定义小动作空间：收集到 {len(self.actions)} 个动作")
            # print("----------------------------------------------")
        elif mode=="small-objs":
            self.actions = self.collect_compact_object_actions(key_objects)
            print(f"选择小动作空间，只考虑物体：收集到 {len(self.actions)} 个动作")
            # print("----------------------------------------------")
        elif mode=="small-predicate-objs":
            self.actions = self.collect_compact_predicate_object_actions(key_predicates,key_objects)
            print(f"选择小动作空间，考虑谓词和物体：收集到 {len(self.actions)} 个动作")
            # print("----------------------------------------------")


        if use_priority_act:
            self.priority_act_ls = self.filter_actions(priority_act_ls)
            self.priority_obj_ls = key_objects
        else:
            self.priority_act_ls=[]
            self.priority_obj_ls =[]
        if self.priority_act_ls !=[]:
            self.consider_priopity=True


        self.actions = self.adjust_action_priority(self.actions, self.priority_act_ls, self.priority_obj_ls,
                                                       self.selected_algorithm)

        self.has_processed = False
        self.llm_reflect = llm_reflect
        self.llm = llm
        self.messages = messages

        self.min_cost = float("inf")

    def process(self, goal):
        """
        Process the input sets and return a string result.
        :param input_set: The set of goal states and the set of initial states.
        :return: A btml string representing the outcome of the behavior tree.
        """
        self.goal = goal
        # if self.bt_algo_opt:
        #     self.algo = OptBTExpAlgorithm(verbose=False)
        # else:
        #     self.algo = BTalgorithm(verbose=False)
        if self.selected_algorithm == "opt":

            self.algo = OptBTExpAlgorithm(verbose=False, \
                                          llm_reflect=self.llm_reflect, llm=self.llm, messages=self.messages, \
                                          priority_act_ls=self.priority_act_ls,time_limit=self.time_limit,
                                          consider_priopity = self.consider_priopity,
                                          heuristic_choice = self.heuristic_choice)

        elif self.selected_algorithm == "baseline":
            self.algo = OptBTExpAlgorithm_BaseLine(verbose=False, \
                                          llm_reflect=self.llm_reflect, llm=self.llm, messages=self.messages, \
                                          priority_act_ls=self.priority_act_ls)
        elif self.selected_algorithm == "opt-h":
            self.algo = OptBTExpAlgorithmHeuristics(verbose=False)
        elif self.selected_algorithm == "bfs":
            self.algo = BTalgorithmBFS(verbose=False,time_limit = self.time_limit)
            # self.algo = BTalgorithm(verbose=False)
        elif self.selected_algorithm == "dfs":
            self.algo = BTalgorithmDFS(verbose=False)
        else:
            print("Error in algorithm selection: This algorithm does not exist.")

        self.algo.clear()
        self.algo.run_algorithm(self.cur_cond_set, self.goal, self.actions)  # 调用算法得到行为树保存至 algo.bt

        # self.btml_string = self.algo.get_btml()
        self.has_processed = True
        # algo.print_solution() # print behavior tree

        # return self.btml_string
        return True

    def post_process(self,ptml_string=True):
        if ptml_string:
            self.btml_string = self.algo.get_btml()
        else:
            self.btml_string=""
        if self.selected_algorithm == "opt":
            self.min_cost = self.algo.min_cost
        else:
            self.min_cost = self.algo.get_cost()
        return self.btml_string, self.min_cost, len(self.algo.expanded)

    def filter_actions(self, priority_act_ls):
        # 创建一个集合，包含所有标准动作的名称
        standard_actions_set = {act.name for act in self.big_actions}
        # 过滤 priority_act_ls，只保留名称在标准集合中的动作
        filtered_priority_act_ls = [act for act in priority_act_ls if act in standard_actions_set]
        return filtered_priority_act_ls


    def adjust_action_priority(self, action_list, priority_act_ls, priority_obj_ls, selected_algorithm):

        recommended_acts = priority_act_ls
        recommended_objs = priority_obj_ls

        for act in action_list:
            act.priority = act.cost
            if act.name in recommended_acts:
                if self.heuristic_choice==0:
                    act.priority = 0
                elif self.heuristic_choice==1:
                    act.priority = act.priority * 1.0 / 10000

        action_list.sort(key=lambda x: (x.priority, x.real_cost, x.name ))


        return action_list


    def collect_compact_object_actions(self, key_objects):
        small_act=[]
        pattern = re.compile(r'\((.*?)\)')
        for act in self.big_actions:
            match = pattern.search(act.name)


            if match:
                action_objects = match.group(1).split(',')
                if all(obj in key_objects for obj in action_objects):
                    small_act.append(act)
        return small_act

    def collect_compact_predicate_object_actions(self, key_predicates, key_objects):
        small_act = []
        pattern = re.compile(r'\((.*?)\)')

        for act in self.big_actions:
            if any(predicate in act.name for predicate in key_predicates):
                match = pattern.search(act.name)
                if match:
                    action_objects = match.group(1).split(',')
                    if all(obj in key_objects for obj in action_objects):
                        small_act.append(act)

        return small_act


    def execute_bt(self,goal,state,verbose=True):
        from btgym.algos.bt_autogen.tools import state_transition
        steps = 0
        current_cost = 0
        current_tick_time = 0
        act_num=1
        record_act = []
        error=False

        val, obj, cost, tick_time = self.algo.bt.cost_tick(state, 0, 0)  # tick行为树，obj为所运行的行动
        if verbose:
            print(f"Action: {act_num}  {obj.__str__().ljust(35)}cost: {cost}")
        record_act.append(obj.__str__())
        current_tick_time += tick_time
        current_cost += cost
        while val != 'success' and val != 'failure':
            state = state_transition(state, obj)
            val, obj, cost, tick_time = self.algo.bt.cost_tick(state, 0, 0)
            act_num+=1
            if verbose:
                print(f"Action: {act_num}  {obj.__str__().ljust(35)}cost: {cost}")
            record_act.append(obj.__str__())
            current_cost += cost
            current_tick_time += tick_time
            if (val == 'failure'):
                if verbose:
                    print("bt fails at step", steps)
                error = True
                break
            steps += 1
            if (steps >= 500):
                break
        if goal <= state:
            if verbose:
                print("Finished!")
        else:
            error = True
        return error,state,act_num-1,current_cost,record_act[:-1]


def collect_action_nodes(behavior_lib):
    action_list = []
    can_expand_ored=0
    for cls in behavior_lib["Action"].values():
        if cls.can_be_expanded:
            can_expand_ored+=1
            print(f"可扩展动作：{cls.__name__}, 存在{len(cls.valid_args)}个有效论域组合")
            # print({cls.__name__})
            if cls.num_args == 0:
                action_list.append(Action(name=cls.get_ins_name(), **cls.get_info()))
            if cls.num_args == 1:
                for arg in cls.valid_args:
                    action_list.append(Action(name=cls.get_ins_name(arg), **cls.get_info(arg)))
            if cls.num_args > 1:
                for args in cls.valid_args:
                    action_list.append(Action(name=cls.get_ins_name(*args), **cls.get_info(*args)))

    print(f"共收集到 {len(action_list)} 个实例化动作")
    print(f"共收集到 {can_expand_ored} 个动作谓词")


    return action_list


def collect_conditions(node):
    from btgym.algos.bt_autogen.behaviour_tree import Leaf
    conditions = set()
    if isinstance(node, Leaf) and node.type == 'cond':
        conditions.update(node.content)
    elif hasattr(node, 'children'):
        for child in node.children:
            conditions.update(collect_conditions(child))
    return conditions




if __name__ == '__main__':

    # todo: Example Cafe
    # todo: Define goal, start, actions
    actions = [
        Action(name='PutDown(Table,Coffee)', pre={'Holding(Coffee)', 'At(Robot,Table)'},
               add={'At(Table,Coffee)', 'NotHolding'}, del_set={'Holding(Coffee)'}, cost=1),
        Action(name='PutDown(Table,VacuumCup)', pre={'Holding(VacuumCup)', 'At(Robot,Table)'},
               add={'At(Table,VacuumCup)', 'NotHolding'}, del_set={'Holding(VacuumCup)'}, cost=1),

        Action(name='PickUp(Coffee)', pre={'NotHolding', 'At(Robot,Coffee)'}, add={'Holding(Coffee)'},
               del_set={'NotHolding'}, cost=1),

        Action(name='MoveTo(Table)', pre={'Available(Table)'}, add={'At(Robot,Table)'},
               del_set={'At(Robot,FrontDesk)', 'At(Robot,Coffee)', 'At(Robot,CoffeeMachine)'}, cost=1),
        Action(name='MoveTo(Coffee)', pre={'Available(Coffee)'}, add={'At(Robot,Coffee)'},
               del_set={'At(Robot,FrontDesk)', 'At(Robot,Table)', 'At(Robot,CoffeeMachine)'}, cost=1),
        Action(name='MoveTo(CoffeeMachine)', pre={'Available(CoffeeMachine)'}, add={'At(Robot,CoffeeMachine)'},
               del_set={'At(Robot,FrontDesk)', 'At(Robot,Coffee)', 'At(Robot,Table)'}, cost=1),

        Action(name='OpCoffeeMachine', pre={'At(Robot,CoffeeMachine)', 'NotHolding'},
               add={'Available(Coffee)', 'At(Robot,Coffee)'}, del_set=set(), cost=1),
    ]
    algo = BTOptExpInterface(actions)

    goal = {'At(Table,Coffee)'}
    btml_string = algo.process(goal)
    print(btml_string)

    file_name = "sub_task"
    with open(f'./{file_name}.btml', 'w') as file:
        file.write(btml_string)

    # 判断初始状态能否到达目标状态
    start = {'At(Robot,Bar)', 'Holding(VacuumCup)', 'Available(Table)', 'Available(CoffeeMachine)',
             'Available(FrontDesk)'}
    # 方法一：算法返回所有可能的初始状态，在里面看看有没有对应的初始状态
    right_bt = algo.find_all_leaf_states_contain_start(start)
    if not right_bt:
        print("ERROR1: The current state cannot reach the goal state!")
    else:
        print("Right1: The current state can reach the goal state!")
    # 方法二：预先跑一边行为树，看能否到达目标状态
    right_bt2 = algo.run_bt_from_start(goal, start)
    if not right_bt2:
        print("ERROR2: The current state cannot reach the goal state!")
    else:
        print("Right2: The current state can reach the goal state!")
