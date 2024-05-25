import copy
import random
import heapq
import re
from btgym.algos.bt_autogen.behaviour_tree import Leaf, ControlBT
from btgym.algos.bt_autogen.Action import Action,state_transition


class CondActPair:
    def __init__(self, cond_leaf, act_leaf):
        self.cond_leaf = cond_leaf
        self.act_leaf = act_leaf

    def __lt__(self, other):
        # 定义优先级比较：按照 cost 的值来比较
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

# 互斥状态映射
mutually_exclusive_states = {
    'IsLeftHandEmpty': 'IsLeftHolding',
    'IsLeftHolding': 'IsLeftHandEmpty',
    'IsRightHandEmpty': 'IsRightHolding',
    'IsRightHolding': 'IsRightHandEmpty',

    'IsSitting': 'IsStanding',
    'IsStanding': 'IsSitting'
}

# Mapping from state to anti-state
state_to_opposite={
    'IsOpen': 'IsClose',
    'IsClose': 'IsOpen',
    'IsSwitchedOff': 'IsSwitchedOn',
    'IsSwitchedOn':'IsSwitchedOff'
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
            if obj in state_dic:
                # 如果对象已经有一个反状态，返回False表示冲突
                if state_dic[obj] == opposite:
                    return False
            # 更新状态字典
            state_dic[obj] = state
            break
    return True

def check_conflict(conds):
    obj_state_dic={}
    is_near=False
    for c in conds:
        if "IsNear" in c and is_near:
            return True
        elif "IsNear" in c:
            is_near=True
        # Cannot be updated, the value already exists in the past
        if not update_state(c, obj_state_dic):
            return True
        # Check for mutually exclusive states without obj
        for state, opposite in mutually_exclusive_states.items():
            if state in c and opposite in obj_state_dic.values():
                return True
            elif state in c:
                obj_state_dic['self'] = state
                break
    return False

class OptBTExpAlgorithmHeuristics:
    def __init__(self, verbose=False):
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

        self.verbose = False
        self.bt_merge = False
        self.output_just_best = False
        self.merge_time=999999

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

    def post_processing(self,pair_node,g_cond_anc_pair,subtree,bt,child_to_parent,cond_to_condActSeq):
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
                print(tmp_seq_struct)
                subtree.add_child([copy.deepcopy(tmp_seq_struct)])

        self.tree_size = self.bfs_cal_tree_size_subtree(bt)
        self.bt_without_merge = bt
        if self.bt_merge:
            bt = self.merge_adjacent_conditions_stack_time(bt, merge_time=self.merge_time)
        return bt




    def run_algorithm_selTree(self, start, goal, actions, merge_time=99999999):
        '''
        Run the planning algorithm to calculate a behavior tree from the initial state, goal state, and available actions
        '''

        self.start = start
        self.goal = goal
        self.actions = actions
        self.merge_time = merge_time
        self.traversed_state_num = 0
        min_cost = float ('inf')

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

        # Initialize the behavior tree with only the target conditions
        bt = ControlBT(type='cond')
        goal_condition_node = Leaf(type='cond', content=goal, min_cost=0)
        goal_action_node = Leaf(type='act', content=None, min_cost=0)

        # Retain the expanded nodes in the subtree first
        subtree = ControlBT(type='?')
        subtree.add_child([goal_condition_node])
        bt.add_child([subtree])
        goal_cond_act_pair = CondActPair(cond_leaf=goal_condition_node, act_leaf=goal_action_node)

        # Using priority queues to store extended nodes
        heapq.heappush(self.nodes, goal_cond_act_pair)
        self.expanded.append(goal)
        self.traversed_state_num += 1
        self.traversed = [goal]  # Set of expanded conditions

        if goal <= start:
            self.bt_without_merge = bt
            print("goal <= start, no need to generate bt.")
            return bt, 0

        while len(self.nodes) != 0:

            self.cycles += 1

            #  Find the condition for the shortest cost path
            # min_cost = float('inf')
            current_pair = heapq.heappop(self.nodes)
            min_cost = current_pair.cond_leaf.min_cost

            if self.verbose:
                print("\nSelecting condition node for expansion:", current_pair.cond_leaf.content)

            c = current_pair.cond_leaf.content

            # # Mount the action node and extend the behavior tree if condition is not the goal and not an empty set
            if c != goal and c != set():
                sequence_structure = ControlBT(type='>')
                sequence_structure.add_child(
                    [current_pair .cond_leaf, current_pair .act_leaf])
                self.expanded.append(c)

                if self.output_just_best:
                    cond_to_condActSeq[current_pair] = sequence_structure
                else:
                    subtree.add_child([copy.deepcopy(sequence_structure)])

                if c <= start:
                    bt = self.post_processing(current_pair , goal_cond_act_pair, subtree, bt,child_to_parent,cond_to_condActSeq)
                    return bt, min_cost

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

                if not c & ((act.pre | act.add) - act.del_set) <= set():
                    if (c - act.del_set) == c:
                        if self.verbose:
                            # Action satisfies conditions for expansion
                            print(f"———— 动作：{act.name}  满足条件可以扩展")
                        c_attr = (act.pre | c) - act.add

                        if check_conflict(c_attr):
                            if self.verbose:
                                print("———— Conflict: action={}, conditions={}".format(act.name, act))
                            continue

                        # 剪枝操作,现在的条件是以前扩展过的条件的超集
                        valid = True

                        for expanded_condition in self.expanded:
                            if expanded_condition <= c_attr:
                                valid = False
                                break

                        # tmp_heap = list(self.nodes)
                        # # tmp_heap = copy.deepcopy(self.nodes)
                        # while tmp_heap:  # 剪枝操作 self.expanded?
                        #     cond_anc_pair = heapq.heappop(tmp_heap)
                        #     j = cond_anc_pair.cond_leaf.content
                        #     if j <= c_attr:
                        #         if cond_anc_pair.cond_leaf.mincost < current_mincost + actions[i].cost:
                        #             valid = False
                        #             break

                        if valid:
                            # c_attr_node = Leaf(type='cond', content=c_attr, min_cost=current_mincost + act.cost)
                            # a_attr_node = Leaf(type='act', content=act, min_cost=current_mincost + act.cost)

                            c_attr_node = Leaf(type='cond', content=c_attr, trust_cost=current_trust+act.cost ,min_cost=current_trust + act.priority)
                            a_attr_node = Leaf(type='act', content=act, trust_cost=current_trust+act.cost ,min_cost=current_trust + act.priority)

                            new_pair = CondActPair(cond_leaf=c_attr_node, act_leaf=a_attr_node)
                            heapq.heappush(self.nodes, new_pair)

                            # Need to record: The upper level of c_attr is c
                            if self.output_just_best:
                                child_to_parent[new_pair] = current_pair

                            self.traversed_state_num += 1
                            # Put all action nodes that meet the conditions into the list
                            traversed_current.append(c_attr)

                            if self.verbose:
                                print("———— -- Action={} meets conditions, new condition={}".format(act.name, c_attr))

            # print(len(traversed_current))
            self.traversed.extend(traversed_current)
            # ====================== End Action Trasvers ============================ #

        self.tree_size = self.bfs_cal_tree_size_subtree(bt)
        self.bt_without_merge = bt

        if self.bt_merge:
            bt = self.merge_adjacent_conditions_stack_time(bt, merge_time=merge_time)

        if self.verbose:
            print("Error: Couldn't find successful bt!")
            print("Algorithm ends!\n")

        return bt, min_cost


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
                bt_sel_tree, min_cost = self.run_algorithm_selTree(start, g, actions)
                subtree_with_costs_ls.append((bt_sel_tree, min_cost))
            # 要排个序再一次add
            sorted_trees = sorted(subtree_with_costs_ls, key=lambda x: x[1])
            for tree, cost in sorted_trees:
                subtree.add_child([tree.children[0]])
            self.bt.add_child([subtree])
            self.min_cost = sorted_trees[0][1]
        else:
            self.bt, min_cost = self.run_algorithm_selTree(start, goal[0], actions, merge_time=merge_time)
            self.min_cost = min_cost
            # print("min_cost:", mincost)
        return True


    def run_algorithm_test(self, start, goal, actions):
        self.bt, mincost = self.run_algorithm_selTree(start, goal, actions)
        return True

    def merge_adjacent_conditions_stack_time(self, bt_sel, merge_time=9999999):

        merge_time = min(merge_time, 500)

        # 只针对第一层合并，之后要考虑层层递归合并
        bt = ControlBT(type='cond')
        sbtree = ControlBT(type='?')
        # gc_node = Leaf(type='cond', content=self.goal, mincost=0)  # 为了统一，都成对出现
        # sbtree.add_child([copy.deepcopy(gc_node)])  # 子树首先保留所扩展结
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
                # 检查合并的条件，前面一个的条件包含了后面的条件，把包含部分提取出来
                last_child = stack[-1]
                last_time = time_stack[-1]

                if last_time < merge_time and isinstance(last_child, ControlBT) and last_child.type == '>':
                    set1 = last_child.children[0].content
                    set2 = child.children[0].content
                    inter = set1 & set2

                    # print("merge time:", last_time,set1,set2)

                    if inter != set():
                        c1 = set1 - set2
                        c2 = set2 - set1
                        inter_node = Leaf(type='cond', content=inter)
                        c1_node = Leaf(type='cond', content=c1)
                        c2_node = Leaf(type='cond', content=c2)
                        a1_node = last_child.children[1]
                        a2_node = child.children[1]

                        # set1<=set2,此时set2对应的动作永远不会执行
                        if (c1 == set() and isinstance(last_child.children[1], Leaf) and isinstance(child.children[1],
                                                                                                    Leaf) \
                                and isinstance(last_child.children[1].content, Action) and isinstance(
                                    child.children[1].content, Action)):
                            continue

                        # 再写一个特殊情况处理，三个结点动作last 遇到 两个结点 且动作相同
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

                        # 判断动作相不相同
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


    def print_solution(self, without_merge=False):
        print("========= BT ==========")  # 树的bfs遍历
        nodes_ls = []
        if without_merge == True:
            nodes_ls.append(self.bt_without_merge)
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

    # 返回所有能到达目标状态的初始状态
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

    # 树的dfs
    def dfs_btml(self, parnode, is_root=False):
        for child in parnode.children:
            if isinstance(child, Leaf):
                if child.type == 'cond':

                    if is_root and len(child.content) > 1:
                        # 把多个 cond 串起来
                        self.btml_string += "sequence{\n"
                        self.btml_string += "cond "
                        c_set_str = '\n cond '.join(map(str, child.content)) + "\n"
                        self.btml_string += c_set_str
                        self.btml_string += '}\n'
                    else:
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

    def dfs_btml_indent(self, parnode, level=0, is_root=False):
        indent = " " * (level * 4)  # 4 spaces per indent level
        for child in parnode.children:
            if isinstance(child, Leaf):

                if is_root and len(child.content) > 1:
                    # 把多个 cond 串起来
                    self.btml_string += " " * (level * 4) + "sequence\n"
                    for c in child.content:
                        self.btml_string += " " * ((level + 1) * 4) + "cond " + str(c) + "\n"

                elif child.type == 'cond':
                    # 直接添加cond及其内容，不需要特别处理根节点下多个cond的情况
                    # self.btml_string += indent + "cond " + ', '.join(map(str, child.content)) + "\n"
                    # 对每个条件独立添加，确保它们各占一行
                    for c in child.content:
                        self.btml_string += indent + "cond " + str(c) + "\n"
                elif child.type == 'act':
                    # 直接添加act及其内容
                    self.btml_string += indent + 'act ' + child.content.name + "\n"
            elif isinstance(child, ControlBT):
                if child.type == '?':
                    self.btml_string += indent + "selector\n"
                    self.dfs_btml_indent(child, level + 1)  # 增加缩进级别
                elif child.type == '>':
                    self.btml_string += indent + "sequence\n"
                    self.dfs_btml_indent(child, level + 1)  # 增加缩进级别

    def get_btml(self, use_braces=True):

        if use_braces:
            self.btml_string = "selector\n"
            self.dfs_btml_indent(self.bt.children[0], 1, is_root=True)
            return self.btml_string
        else:
            self.btml_string = "selector{\n"
            self.dfs_btml(self.bt.children[0], is_root=True)
            self.btml_string += '}\n'
        return self.btml_string

    # def dfs_btml_many_act(self, parnode, is_root=False):
    #     for child in parnode.children:
    #         if isinstance(child, Leaf):
    #             if child.type == 'cond':
    #
    #                 # if is_root and len(child.content) > 1:
    #                 if is_root and len(child.content) > 1:
    #                     # 把多个 cond 串起来
    #                     self.btml_string += "sequence{\n"
    #                     self.btml_string += "cond "
    #                     c_set_str = '\n cond '.join(map(str, child.content)) + "\n"
    #                     self.btml_string += c_set_str
    #                     self.btml_string += '}\n'
    #                 # elif is_root:
    #                 else:
    #                     self.btml_string += "cond "
    #                     c_set_str = '\n cond '.join(map(str, child.content)) + "\n"
    #                     self.btml_string += c_set_str
    #                 # else:
    #                 #     c_set_positive = {item for item in child.content if 'Not' not in item}
    #                 #     self.btml_string += "cond "
    #                 #     c_set_str = '\n cond '.join(map(str, c_set_positive)) + "\n"
    #                 #     self.btml_string += c_set_str
    #
    #
    #             elif child.type == 'act':
    #
    #                 # child.content.name = re.sub(r'\d+', '', child.content.name)
    #                 child.content.name = re.sub(r'\)\d+', ')', child.content.name)
    #                 if '(' not in child.content.name:
    #                     self.btml_string += 'act ' + child.content.name + "()\n"
    #                 else:
    #                     self.btml_string += 'act ' + child.content.name + "\n"
    #         elif isinstance(child, ControlBT):
    #             if child.type == '?':
    #                 self.btml_string += "selector{\n"
    #                 if len(child.children)>2:
    #                     self.dfs_btml_many_act(parnode=child, is_root=True)
    #                 else:
    #                     self.dfs_btml_many_act(parnode=child)
    #             elif child.type == '>':
    #                 self.btml_string += "sequence{\n"
    #                 self.dfs_btml_many_act(parnode=child)
    #             self.btml_string += '}\n'
    #
    # def get_btml_many_act(self):
    #     self.btml_string = "selector{\n"
    #     self.dfs_btml_many_act(self.bt.children[0],is_root=True)
    #     self.btml_string += '}\n'
    #     return self.btml_string

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
        # 开始从初始状态运行行为树，测试
        state = self.start
        steps = 0
        current_cost = 0
        current_tick_time = 0
        val, obj, cost, tick_time = self.bt.cost_tick(state, 0, 0)  # tick行为树，obj为所运行的行动

        current_tick_time += tick_time
        current_cost += cost
        while val != 'success' and val != 'failure':  # 运行直到行为树成功或失败
            state = state_transition(state, obj)
            val, obj, cost, tick_time = self.bt.cost_tick(state, 0, 0)
            current_cost += cost
            current_tick_time += tick_time
            if (val == 'failure'):
                print("bt fails at step", steps)
                error = True
                break
            steps += 1
            if (steps >= 500):  # 至多运行500步
                break
        if not self.goal <= state:  # 错误解，目标条件不在执行后状态满足
            print ("wrong solution",steps)
            error = True
            return current_cost
        else:  # 正确解，满足目标条件
            return current_cost