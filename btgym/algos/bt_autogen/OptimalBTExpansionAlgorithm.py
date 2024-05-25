import copy
import time
import random
import heapq
import re
from btgym.algos.bt_autogen.behaviour_tree import Leaf, ControlBT
from btgym.algos.bt_autogen.Action import Action, state_transition
from collections import deque
import random
import numpy as np

seed = 0
random.seed(seed)
np.random.seed(seed)


class CondActPair:
    def __init__(self, cond_leaf, act_leaf):
        self.cond_leaf = cond_leaf
        self.act_leaf = act_leaf
        self.parent = None
        self.children = []
        self.isCostOneAdded = False
        self.path = 1
        self.pact_dic = {}

    def __lt__(self, other):
        return self.act_leaf.min_cost < other.act_leaf.min_cost



def set_to_tuple(s):
    """
    Convert a set of strings to a tuple with elements sorted.
    This ensures that the order of elements in the set does not affect the resulting tuple,
    making it suitable for use as a dictionary key.

    Parameters:
    - s: The set of strings to convert.

    Returns:
    - A tuple containing the sorted elements from the set.
    """
    return tuple(sorted(s))


# self状态:互斥状态映射
mutually_exclusive_states = {
    'IsLeftHandEmpty': 'IsLeftHolding',
    'IsLeftHolding': 'IsLeftHandEmpty',
    'IsRightHandEmpty': 'IsRightHolding',
    'IsRightHolding': 'IsRightHandEmpty',

    'IsSitting': 'IsStanding',
    'IsStanding': 'IsSitting',

}

# 物体状态: Mapping from state to anti-state
state_to_opposite = {
    'IsOpen': 'IsClose',
    'IsClose': 'IsOpen',
    'IsSwitchedOff': 'IsSwitchedOn',
    'IsSwitchedOn': 'IsSwitchedOff',
    'IsPlugged': 'IsUnplugged',
    'IsUnplugged': 'IsPlugged',
}


def extract_argument(state):
    match = re.search(r'\((.*?)\)', state)
    if match:
        return match.group(1)
    return None


def update_state(c, state_dic):
    for state, opposite in state_to_opposite.items():
        if state in c:
            obj = extract_argument(c)
            if obj in state_dic and opposite in state_dic[obj]:
                return False
            # 更新状态字典
            elif obj in state_dic:
                state_dic[obj].add(state)
            else:
                state_dic[obj] = set()
                state_dic[obj].add(state)
            break
    return True


def check_conflict(conds):
    obj_state_dic = {}
    self_state_dic = {}
    self_state_dic['self'] = set()
    is_near = False
    for c in conds:
        if "IsNear" in c and is_near:
            return True
        elif "IsNear" in c:
            is_near = True
            continue
        # Cannot be updated, the value already exists in the past
        if not update_state(c, obj_state_dic):
            return True
        # Check for mutually exclusive states without obj
        for state, opposite in mutually_exclusive_states.items():
            if state in c and opposite in self_state_dic['self']:
                return True
            elif state in c:
                self_state_dic['self'].add(state)
                break
    # 检查是否同时具有 'IsHoldingCleaningTool(self)', 'IsLeftHandEmpty(self)', 'IsRightHandEmpty(self)'
    required_states = {'IsHoldingCleaningTool(self)', 'IsLeftHandEmpty(self)', 'IsRightHandEmpty(self)'}
    if all(state in conds for state in required_states):
        return True
    required_states = {'IsHoldingKnife(self)', 'IsLeftHandEmpty(self)', 'IsRightHandEmpty(self)'}
    if all(state in conds for state in required_states):
        return True

    return False


class OptBTExpAlgorithm:
    def __init__(self, verbose=False, llm_reflect=False, llm=None, messages=None, priority_act_ls=None, time_limit=None, \
                 consider_priopity=False, heuristic_choice=-1):
        self.bt = None
        self.start = None
        self.goal = None
        self.actions = None
        self.min_cost = float('inf')

        self.nodes = []
        self.cycles = 0
        self.tree_size = 0

        self.expanded = []  # Conditions for storing expanded nodes
        self.traversed = []  # Conditions for storing nodes that have been put into the priority queue
        self.traversed_state_num = 0

        self.bt_without_merge = None
        self.subtree_count = 1

        self.verbose = verbose
        self.bt_merge = True
        self.output_just_best = True
        self.merge_time = 5

        self.act_bt = None

        self.llm_reflect = llm_reflect
        self.llm = llm
        self.messages = messages
        self.priority_act_ls = priority_act_ls

        self.act_cost_dic = {}
        self.time_limit_exceeded = False
        self.time_limit = time_limit

        self.consider_priopity = consider_priopity
        self.heuristic_choice = heuristic_choice

    def clear(self):
        self.bt = None
        self.start = None
        self.goal = None
        self.actions = None
        self.min_cost = float('inf')

        self.nodes = []
        self.cycles = 0
        self.tree_size = 0

        self.expanded = []  # Conditions for storing expanded nodes
        self.traversed = []  # Conditions for storing nodes that have been put into the priority queue
        self.traversed_state_num = 0

        self.bt_without_merge = None
        self.subtree_count = 1

        self.act_bt = None

    def post_processing(self, pair_node, g_cond_anc_pair, subtree, bt, child_to_parent, cond_to_condActSeq):
        '''
        Process the summary work after the algorithm ends.
        '''
        if self.output_just_best:
            # Only output the best
            output_stack = []
            tmp_pair = pair_node
            while tmp_pair != g_cond_anc_pair:
                tmp_seq_struct = cond_to_condActSeq[tmp_pair]
                output_stack.append(tmp_seq_struct)
                tmp_pair = child_to_parent[tmp_pair]

            while output_stack != []:
                tmp_seq_struct = output_stack.pop()
                # print(tmp_seq_struct)
                subtree.add_child([copy.deepcopy(tmp_seq_struct)])

        self.tree_size = self.bfs_cal_tree_size_subtree(bt)
        self.bt_without_merge = bt
        if self.bt_merge:
            bt = self.merge_adjacent_conditions_stack_time(bt, merge_time=self.merge_time)
        return bt

    def transfer_pair_node_to_bt(self, path_nodes, root_pair):
        bt = ControlBT(type='cond')

        goal_condition_node = root_pair.cond_leaf
        goal_action_node = root_pair.act_leaf
        bt.add_child([goal_condition_node])


        queue = deque([root_pair])
        while queue:
            current = queue.popleft()

            # 建树
            subtree = ControlBT(type='?')
            subtree.add_child([copy.deepcopy(current.cond_leaf)])

            for child in current.children:
                if child not in path_nodes:
                    continue
                queue.append(child)

                seq = ControlBT(type='>')
                seq.add_child([child.cond_leaf, child.act_leaf])
                subtree.add_child([seq])

            parent_of_c = current.cond_leaf.parent
            parent_of_c.children[0] = subtree
        return bt

    def transfer_pair_node_to_act_tree(self, path_nodes, root_pair):

        conditions = ', '.join(root_pair.cond_leaf.content)
        act_tree_string = f'GOAL {conditions}\n'

        def build_act_tree(node, indent, act_count):

            node_string = ''
            current_act = 1
            for child in node.children:
                if child in path_nodes:

                    prefix = '    ' * indent
                    act_label = f'ACT {act_count}.{current_act}: ' if act_count else f'ACT {current_act}: '

                    node_string += f'{prefix}{act_label}{child.act_leaf.content.name}  f={round(child.act_leaf.min_cost, 1)} g={round(child.act_leaf.trust_cost, 1)}\n'

                    node_string += build_act_tree(child, indent + 1,
                                                  f'{act_count}.{current_act}' if act_count else str(current_act))
                    current_act += 1

            return node_string

        act_tree_string += build_act_tree(root_pair, 1, '')
        return act_tree_string

    # 调用大模型进行反馈
    def call_large_model(self, goal_cond_act_pair):

        # top_five_leaves = heapq.nlargest(5, self.nodes)
        top_five_leaves = heapq.nsmallest(20, self.nodes)

        path_nodes = set()

        for leaf in top_five_leaves:
            current = leaf
            while current.parent != None:
                path_nodes.add(current)
                current = current.parent
            path_nodes.add(goal_cond_act_pair)  # 添加根节点


        # 如果不建立BT,之间里动作树
        self.act_tree_string = self.transfer_pair_node_to_act_tree(path_nodes=path_nodes, root_pair=goal_cond_act_pair)
        print(self.act_tree_string)
        # =================================================

        prompt = ""
        prompt += self.act_tree_string
        prompt += (
                "\nThis is the action tree currently found by the reverse expansion algorithm of the behavior tree. " + \
                "To reach the goal state, please recommend the key actions needed next to reach the goal state based on the existing action tree. " + \
                "This time, there is no need to output the goal state, only the key actions are needed. " + \
                'The format for presenting key actions should start with the word \'Actions:\'. ')

    def run_algorithm_selTree(self, start, goal, actions, merge_time=99999999):
        '''
        Run the planning algorithm to calculate a behavior tree from the initial state, goal state, and available actions
        '''

        start_time = time.time()

        self.start = start
        self.goal = goal
        self.actions = actions
        self.merge_time = merge_time
        self.traversed_state_num = 0
        min_cost = float('inf')

        child_to_parent = {}
        cond_to_condActSeq = {}

        self.nodes = []
        self.cycles = 0
        self.tree_size = 0

        self.expanded = []  # Conditions for storing expanded nodes

        self.traversed = []  # Conditions for storing nodes that have been put into the priority queue
        self.traversed_state_num = 0

        self.bt_without_merge = None
        self.subtree_count = 1

        if self.verbose:
            print("\nAlgorithm starts！")

        # 初始化

        for act in self.actions:
            self.act_cost_dic[act.name] = act.cost

        # Initialize the behavior tree with only the target conditions
        bt = ControlBT(type='cond')
        # goal_condition_node = Leaf(type='cond', content=goal, min_cost=0)
        # goal_action_node = Leaf(type='act', content=None, min_cost=0)

        goal_condition_node = Leaf(type='cond', content=goal, min_cost=0)
        goal_action_node = Leaf(type='act', content=None, min_cost=0)

        # Retain the expanded nodes in the subtree first
        subtree = ControlBT(type='?')
        subtree.add_child([copy.deepcopy(goal_condition_node)])
        bt.add_child([subtree])
        goal_cond_act_pair = CondActPair(cond_leaf=goal_condition_node, act_leaf=goal_action_node)

        # I(C,act)
        for act in self.priority_act_ls:
            if act not in goal_cond_act_pair.pact_dic:
                goal_cond_act_pair.pact_dic[act] = 1
            else:
                goal_cond_act_pair.pact_dic[act] += 1

        D_first_cost = 0
        D_first_num = 0
        for key, value in goal_cond_act_pair.pact_dic.items():
            D_first_cost += self.act_cost_dic[key] * value
            D_first_num += value
        goal_condition_node.trust_cost = 0
        goal_action_node.trust_cost = 0
        goal_condition_node.min_cost = D_first_cost
        goal_action_node.min_cost = D_first_cost

        # Using priority queues to store extended nodes
        heapq.heappush(self.nodes, goal_cond_act_pair)
        # self.expanded.append(goal)
        self.expanded.append(goal_condition_node)
        self.traversed_state_num += 1
        self.traversed = [goal]  # Set of expanded conditions

        if goal <= start:
            self.bt_without_merge = bt
            print("goal <= start, no need to generate bt.")
            return bt, 0, self.time_limit_exceeded

        epsh = 0
        while len(self.nodes) != 0:

            if self.llm_reflect:
                if len(self.expanded) >= 1:
                    self.call_large_model(goal_cond_act_pair=goal_cond_act_pair)

            self.cycles += 1
            current_pair = heapq.heappop(self.nodes)
            min_cost = current_pair.cond_leaf.min_cost

            if self.verbose:
                print("\nSelecting condition node for expansion:", current_pair.cond_leaf.content)

            c = current_pair.cond_leaf.content

            # # Mount the action node and extend the behavior tree if condition is not the goal and not an empty set
            if c != goal and c != set():
                sequence_structure = ControlBT(type='>')
                sequence_structure.add_child(  # 这里做 ACT TREE 时候，没有copy 被更新了父节点
                    [copy.deepcopy(current_pair.cond_leaf), copy.deepcopy(current_pair.act_leaf)])
                # self.expanded.append(c)
                self.expanded.append(current_pair.cond_leaf)

                if self.output_just_best:
                    cond_to_condActSeq[current_pair] = sequence_structure
                else:
                    subtree.add_child([copy.deepcopy(sequence_structure)])

                if c <= start:
                    bt = self.post_processing(current_pair, goal_cond_act_pair, subtree, bt, child_to_parent,
                                              cond_to_condActSeq)
                    return bt, min_cost, self.time_limit_exceeded

            elif c == set() and c <= start:
                sequence_structure = ControlBT(type='>')
                sequence_structure.add_child(
                    [copy.deepcopy(current_pair.cond_leaf), copy.deepcopy(current_pair.act_leaf)])
                # self.expanded.append(c)
                self.expanded.append(current_pair.cond_leaf)

                if self.output_just_best:
                    cond_to_condActSeq[current_pair] = sequence_structure
                else:
                    subtree.add_child([copy.deepcopy(sequence_structure)])
                bt = self.post_processing(current_pair, goal_cond_act_pair, subtree, bt, child_to_parent,
                                          cond_to_condActSeq)
                return bt, min_cost, self.time_limit_exceeded

            # timeout
            if self.time_limit != None and time.time() - start_time > self.time_limit:
                self.time_limit_exceeded = True
                bt = self.post_processing(current_pair, goal_cond_act_pair, subtree, bt, child_to_parent,
                                          cond_to_condActSeq)
                return bt, min_cost, self.time_limit_exceeded

                if self.verbose:
                    print("Expansion complete for action node={}, with new conditions={}, min_cost={}".format(
                        current_pair.act_leaf.content.name, current_pair.cond_leaf.content,
                        current_pair.cond_leaf.min_cost))

            if self.verbose:
                print("Traverse all actions and find actions that meet the conditions:")
                print("============")
            current_mincost = current_pair.cond_leaf.min_cost
            current_trust = current_pair.cond_leaf.trust_cost

            # ====================== Action Trasvers ============================ #
            # Traverse actions to find applicable ones
            traversed_current = []
            for act in actions:

                epsh += 0.00000000001
                if not c & ((act.pre | act.add) - act.del_set) <= set():
                    if (c - act.del_set) == c:

                        c_attr = (act.pre | c) - act.add

                        if check_conflict(c_attr):
                            continue

                        valid = True

                        if self.consider_priopity:
                            for expanded_condition in self.expanded:
                                if expanded_condition.content <= c_attr:
                                    valid = False
                                    break
                            pass
                        else:
                            for expanded_condition in self.expanded:
                                if expanded_condition.content <= c_attr:
                                    valid = False
                                    break

                        if valid:

                            c_attr_node = Leaf(type='cond', content=c_attr, min_cost=current_mincost + act.cost,
                                               parent_cost=current_mincost)
                            a_attr_node = Leaf(type='act', content=act, min_cost=current_mincost + act.cost,
                                               parent_cost=current_mincost)

                            new_pair = CondActPair(cond_leaf=c_attr_node, act_leaf=a_attr_node)
                            new_pair.path = current_pair.path + 1


                            h = 0

                            new_pair.pact_dic = copy.deepcopy(current_pair.pact_dic)
                            if act.name in new_pair.pact_dic and new_pair.pact_dic[act.name] > 0:
                                new_pair.pact_dic[act.name] -= 1
                                c_attr_node.min_cost = current_mincost + act.priority
                                a_attr_node.min_cost = current_mincost + act.priority

                            heapq.heappush(self.nodes, new_pair)

                            # 记录结点的父子关系
                            new_pair.parent = current_pair
                            current_pair.children.append(new_pair)

                            # 如果之前标记过/之前没标记但现在是1
                            if (
                                    current_pair.isCostOneAdded == False and act.cost == 1) or current_pair.isCostOneAdded == True:
                                new_pair.isCostOneAdded = True
                            # 之前标记过但现在是1 表示多加了1
                            if current_pair.isCostOneAdded == True and act.cost == 1:
                                new_pair.cond_leaf.min_cost -= 1
                                new_pair.act_leaf.min_cost -= 1

                            # Need to record: The upper level of c_attr is c
                            if self.output_just_best:
                                child_to_parent[new_pair] = current_pair

                            self.traversed_state_num += 1
                            # Put all action nodes that meet the conditions into the list
                            traversed_current.append(c_attr)

                            if self.verbose:
                                print("———— -- Action={} meets conditions, new condition={}".format(act.name, c_attr))

            self.traversed.extend(traversed_current)
            # ====================== End Action Trasvers ============================ #

        bt = self.post_processing(current_pair, goal_cond_act_pair, subtree, bt, child_to_parent,
                                  cond_to_condActSeq)

        self.tree_size = self.bfs_cal_tree_size_subtree(bt)
        self.bt_without_merge = bt

        if self.bt_merge:
            bt = self.merge_adjacent_conditions_stack_time(bt, merge_time=merge_time)

        if self.verbose:
            print("Error: Couldn't find successful bt!")
            print("Algorithm ends!\n")

        return bt, min_cost, self.time_limit_exceeded

    def run_algorithm(self, start, goal, actions, merge_time=999999):
        """
        Generates a behavior tree for achieving specified goal(s) from a start state using given actions.
        If multiple goals are provided, it creates individual trees per goal and merges them based on
        minimum cost. For a single goal, it generates one behavior tree.

        Parameters:
        - start: Initial state.
        - goal: Single goal state or a list of goal states.
        - actions: Available actions.
        - merge_time (optional): Controls tree merging process; default is 3.

        Returns:
        - True if successful. Specific behavior depends on implementation details.
        """
        self.bt = ControlBT(type='cond')
        subtree = ControlBT(type='?')

        subtree_with_costs_ls = []

        self.subtree_count = len(goal)

        if len(goal) > 1:
            for g in goal:
                bt_sel_tree, min_cost,time_limit_exceeded = self.run_algorithm_selTree(start, g, actions)
                subtree_with_costs_ls.append((bt_sel_tree, min_cost))

            sorted_trees = sorted(subtree_with_costs_ls, key=lambda x: x[1])
            for tree, cost in sorted_trees:
                subtree.add_child([tree.children[0]])
            self.bt.add_child([subtree])
            self.min_cost = sorted_trees[0][1]
        else:
            self.bt, min_cost, time_limit_exceeded = self.run_algorithm_selTree(start, goal[0], actions,
                                                                                merge_time=merge_time)
            self.min_cost = min_cost
            # print("min_cost:", mincost)
        return True

    def run_algorithm_test(self, start, goal, actions):
        self.bt, mincost = self.run_algorithm_selTree(start, goal, actions)
        return True

    def merge_adjacent_conditions_stack_time(self, bt_sel, merge_time=9999999):

        merge_time = min(merge_time, 500)


        bt = ControlBT(type='cond')
        sbtree = ControlBT(type='?')

        bt.add_child([sbtree])

        parnode = bt_sel.children[0]
        stack = []
        time_stack = []
        for child in parnode.children:
            if isinstance(child, ControlBT) and child.type == '>':
                if stack == []:
                    stack.append(child)
                    time_stack.append(0)
                    continue

                last_child = stack[-1]
                last_time = time_stack[-1]

                if last_time < merge_time and isinstance(last_child, ControlBT) and last_child.type == '>':
                    set1 = last_child.children[0].content
                    set2 = child.children[0].content
                    inter = set1 & set2



                    if inter != set():
                        c1 = set1 - set2
                        c2 = set2 - set1
                        inter_node = Leaf(type='cond', content=inter)
                        c1_node = Leaf(type='cond', content=c1)
                        c2_node = Leaf(type='cond', content=c2)
                        a1_node = last_child.children[1]
                        a2_node = child.children[1]


                        if (c1 == set() and isinstance(last_child.children[1], Leaf) and isinstance(child.children[1],
                                                                                                    Leaf) \
                                and isinstance(last_child.children[1].content, Action) and isinstance(
                                    child.children[1].content, Action)):
                            continue

                        if len(last_child.children) == 3 and \
                                isinstance(last_child.children[2], Leaf) and isinstance(child.children[1], Leaf) \
                                and isinstance(last_child.children[2].content, Action) and isinstance(
                            child.children[1].content, Action) \
                                and last_child.children[2].content.name == child.children[1].content.name \
                                and c1 == set() and c2 != set():
                            last_child.children[1].add_child([c2_node])
                            continue
                        elif len(last_child.children) == 3:
                            stack.append(child)
                            time_stack.append(0)
                            continue


                        if isinstance(last_child.children[1], Leaf) and isinstance(child.children[1], Leaf) \
                                and isinstance(last_child.children[1].content, Action) and isinstance(
                            child.children[1].content, Action) \
                                and last_child.children[1].content.name == child.children[1].content.name:

                            if c2 == set():
                                tmp_tree = ControlBT(type='>')
                                tmp_tree.add_child(
                                    [inter_node, a1_node])
                            else:
                                _sel = ControlBT(type='?')
                                _sel.add_child([c1_node, c2_node])
                                tmp_tree = ControlBT(type='>')
                                tmp_tree.add_child(
                                    [inter_node, _sel, a1_node])
                        else:
                            if c1 == set():
                                seq1 = last_child.children[1]
                            else:
                                seq1 = ControlBT(type='>')
                                seq1.add_child([c1_node, a1_node])

                            if c2 == set():
                                seq2 = child.children[1]
                            else:
                                seq2 = ControlBT(type='>')
                                seq2.add_child([c2_node, a2_node])
                            sel = ControlBT(type='?')
                            sel.add_child([seq1, seq2])
                            tmp_tree = ControlBT(type='>')
                            tmp_tree.add_child(
                                [inter_node, sel])

                        stack.pop()
                        time_stack.pop()
                        stack.append(tmp_tree)
                        time_stack.append(last_time + 1)

                    else:
                        stack.append(child)
                        time_stack.append(0)
                else:
                    stack.append(child)
                    time_stack.append(0)
            else:
                stack.append(child)
                time_stack.append(0)

        for tree in stack:
            sbtree.add_child([tree])
        bt_sel = bt
        return bt_sel

    def print_solution(self, bt=None, without_merge=False, act_bt_tree=False):
        print("========= BT ==========")
        nodes_ls = []
        if without_merge == True:
            nodes_ls.append(self.bt_without_merge)
        else:
            if act_bt_tree:
                bt = bt
                nodes_ls.append(bt)
            else:
                nodes_ls.append(self.bt)
        while len(nodes_ls) != 0:
            parnode = nodes_ls[0]
            print("Parrent:", parnode.type)
            for child in parnode.children:
                if isinstance(child, Leaf):
                    print("---- Leaf:", child.content)
                elif isinstance(child, ControlBT):
                    print("---- ControlBT:", child.type)
                    nodes_ls.append(child)
            print()
            nodes_ls.pop(0)
        print("========= BT ==========\n")


    def get_all_state_leafs(self):
        state_leafs = []

        nodes_ls = []
        nodes_ls.append(self.bt)
        while len(nodes_ls) != 0:
            parnode = nodes_ls[0]
            for child in parnode.children:
                if isinstance(child, Leaf):
                    if child.type == "cond":
                        state_leafs.append(child.content)
                elif isinstance(child, ControlBT):
                    nodes_ls.append(child)
            nodes_ls.pop(0)

        return state_leafs


    def dfs_btml(self, parnode, is_root=False, act_bt_tree=False):
        for child in parnode.children:
            if isinstance(child, Leaf):
                if child.type == 'cond':

                    if is_root and len(child.content) > 1:

                        self.btml_string += "sequence{\n"

                        if act_bt_tree == False:
                            self.btml_string += "cond "
                            c_set_str = '\n cond '.join(map(str, child.content)) + "\n"
                            self.btml_string += c_set_str
                            self.btml_string += '}\n'
                    else:
                        if act_bt_tree == False:
                            self.btml_string += "cond "
                            c_set_str = '\n cond '.join(map(str, child.content)) + "\n"
                            self.btml_string += c_set_str

                elif child.type == 'act':
                    if '(' not in child.content.name:
                        self.btml_string += 'act ' + child.content.name + "()\n"
                    else:
                        self.btml_string += 'act ' + child.content.name + "\n"
            elif isinstance(child, ControlBT):
                if child.type == '?':
                    self.btml_string += "selector{\n"
                    self.dfs_btml(parnode=child)
                elif child.type == '>':
                    self.btml_string += "sequence{\n"
                    self.dfs_btml(parnode=child)
                self.btml_string += '}\n'

    def dfs_btml_indent(self, parnode, level=0, is_root=False, act_bt_tree=False):
        indent = " " * (level * 4)  # 4 spaces per indent level
        for child in parnode.children:
            if isinstance(child, Leaf):

                if is_root and len(child.content) > 1:
                    # 把多个 cond 串起来
                    self.btml_string += " " * (level * 4) + "sequence\n"
                    if act_bt_tree == False:
                        for c in child.content:
                            self.btml_string += " " * ((level + 1) * 4) + "cond " + str(c) + "\n"

                elif child.type == 'cond':

                    if act_bt_tree == False:
                        for c in child.content:
                            self.btml_string += indent + "cond " + str(c) + "\n"
                elif child.type == 'act':

                    self.btml_string += indent + 'act ' + child.content.name + "\n"
            elif isinstance(child, ControlBT):
                if child.type == '?':
                    self.btml_string += indent + "selector\n"
                    self.dfs_btml_indent(child, level + 1, act_bt_tree=act_bt_tree)  # 增加缩进级别
                elif child.type == '>':
                    self.btml_string += indent + "sequence\n"
                    self.dfs_btml_indent(child, level + 1, act_bt_tree=act_bt_tree)  # 增加缩进级别

    def get_btml(self, use_braces=True, act_bt_tree=False):

        if use_braces:
            self.btml_string = "selector\n"
            if act_bt_tree == False:
                self.dfs_btml_indent(self.bt.children[0], 1, is_root=True)
            else:
                self.dfs_btml_indent(self.act_bt.children[0], 1, is_root=True, act_bt_tree=act_bt_tree)
            return self.btml_string
        else:
            self.btml_string = "selector{\n"
            if act_bt_tree == False:
                self.dfs_btml(self.bt.children[0], is_root=True)
            else:
                self.dfs_btml(self.act_bt.children[0], is_root=True, act_bt_tree=True)
            self.btml_string += '}\n'
        return self.btml_string

    def ACT_BT_dfs_btml_indent(self, parnode, level=0, is_root=False):
        indent = " " * (level * 4)  # 4 spaces per indent level
        for child in parnode.children:
            if isinstance(child, Leaf):

                if is_root and len(child.content) > 1:

                    self.ACT_BT_btml_string += " " * (level * 4) + "sequence\n"
                    for c in child.content:
                        self.ACT_BT_btml_string += " " * ((level + 1) * 4) + "cond " + str(c) + "\n"

                elif child.type == 'cond':
                    for c in child.content:
                        self.ACT_BT_btml_string += indent + "cond " + str(c) + "\n"
                elif child.type == 'act':

                    self.ACT_BT_btml_string += indent + 'act ' + child.content.name + "\n"
            elif isinstance(child, ControlBT):
                if child.type == '?':
                    self.ACT_BT_btml_string += indent + "selector\n"
                    self.ACT_BT_dfs_btml_indent(child, level + 1)
                elif child.type == '>':
                    self.ACT_BT_btml_string += indent + "sequence\n"
                    self.ACT_BT_dfs_btml_indent(child, level + 1)

    def ACT_BT_get_btml(self):
        self.ACT_BT_btml_string = "selector\n"
        self.ACT_BT_dfs_btml_indent(self.act_bt.children[0], 1, is_root=True)
        return self.ACT_BT_btml_string

    def save_btml_file(self, file_name):
        self.btml_string = "selector{\n"
        self.dfs_btml(self.bt.children[0])
        self.btml_string += '}\n'
        with open(f'./{file_name}.btml', 'w') as file:
            file.write(self.btml_string)
        return self.btml_string

    def bfs_cal_tree_size(self):
        from collections import deque
        queue = deque([self.bt.children[0]])

        count = 0
        while queue:
            current_node = queue.popleft()
            count += 1
            for child in current_node.children:
                if isinstance(child, Leaf):
                    count += 1
                else:
                    queue.append(child)
        return count

    def bfs_cal_tree_size_subtree(self, bt):
        from collections import deque
        queue = deque([bt.children[0]])

        count = 0
        while queue:
            current_node = queue.popleft()
            count += 1
            for child in current_node.children:
                if isinstance(child, Leaf):
                    count += 1
                else:
                    queue.append(child)
        return count

    def get_cost(self):

        state = self.start
        steps = 0
        current_cost = 0
        current_tick_time = 0
        val, obj, cost, tick_time = self.bt.cost_tick(state, 0, 0)

        current_tick_time += tick_time
        current_cost += cost
        while val != 'success' and val != 'failure':
            state = state_transition(state, obj)
            val, obj, cost, tick_time = self.bt.cost_tick(state, 0, 0)
            current_cost += cost
            current_tick_time += tick_time
            if (val == 'failure'):
                print("bt fails at step", steps)
                error = True
                break
            steps += 1
            if (steps >= 500):
                break
        if not self.goal <= state:
            print("wrong solution", steps)
            error = True
            return current_cost
        else:
            return current_cost
