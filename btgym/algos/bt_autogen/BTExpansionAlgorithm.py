import random
import numpy as np
import copy
import time
from btgym.algos.bt_autogen.behaviour_tree import Leaf,ControlBT
from btgym.algos.bt_autogen.Action import Action,generate_random_state,state_transition



class BTalgorithm:
    def __init__(self,verbose=False):
        self.bt = None
        self.nodes = []
        self.traversed = []
        self.conditions = []
        self.conditions_index = []
        self.verbose = verbose

    def clear(self):
        self.bt = None
        self.nodes = []
        self.traversed = []
        self.conditions = []
        self.conditions_index = []


    def run_algorithm_selTree(self, start, goal, actions):
        bt = ControlBT(type='cond')
        g_node = Leaf(type='cond', content=goal,min_cost=0)
        bt.add_child([g_node])

        self.conditions.append(goal)
        self.nodes.append(g_node)
        val, obj = bt.tick(start)
        canrun = False
        if val == 'success' or val == 'running':
            canrun = True

        while not canrun:
            index = -1
            for i in range(0, len(self.nodes)):
                if self.nodes[i].content in self.traversed:
                    continue
                else:
                    c_node = self.nodes[i]
                    index = i
                    break
            if index == -1:
                print('Failure')
                return False

            subtree = ControlBT(type='?')
            subtree.add_child([copy.deepcopy(c_node)])
            c = c_node.content

            for i in range(0, len(actions)):

                if not c & ((actions[i].pre | actions[i].add) - actions[i].del_set) <= set():
                    if (c - actions[i].del_set) == c:
                        c_attr = (actions[i].pre | c) - actions[i].add
                        valid = True

                        for j in self.traversed:
                            if j <= c_attr:
                                valid = False
                                break

                        if valid:
                            sequence_structure = ControlBT(type='>')
                            c_attr_node = Leaf(type='cond', content=c_attr, min_cost=0)
                            a_node = Leaf(type='act', content=actions[i], min_cost=0)
                            sequence_structure.add_child([c_attr_node, a_node])
                            subtree.add_child([sequence_structure])

                            self.nodes.append(c_attr_node)
            parent_of_c = c_node.parent
            parent_of_c.children[0] = subtree
            self.traversed.append(c)
            val, obj = bt.tick(start)
            canrun = False
            if val == 'success' or val == 'running':
                canrun = True
        return bt



    def run_algorithm(self, start, goal, actions):

        self.bt = ControlBT(type='cond')
        subtree = ControlBT(type='?')
        if len(goal) > 1:
            for g in goal:
                print("goal",g)
                bt_sel_tree = self.run_algorithm_selTree(start, g, actions)
                print("bt_sel_tree.children",bt_sel_tree.children)
                subtree.add_child([copy.deepcopy(bt_sel_tree.children[0])])
            self.bt.add_child([subtree])
        else:
            self.bt = self.run_algorithm_selTree(start, goal[0], actions)
        return True

    def print_solution(self):
        print(len(self.nodes))
    # 树的dfs
    def dfs_btml(self,parnode,is_root=False):
        for child in parnode.children:
            if isinstance(child, Leaf):
                if child.type == 'cond':

                    if is_root and len(child.content) > 1:
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
                    self.dfs_btml( parnode=child)
                self.btml_string += '}\n'


    def dfs_btml_indent(self, parnode, level=0, is_root=False):
        indent = " " * (level * 4)  # 4 spaces per indent level
        for child in parnode.children:
            if isinstance(child, Leaf):

                if child.type == 'cond':
                    if not is_root and len(child.content) > 1:
                        self.btml_string += " " * (level * 4) + "sequence\n"
                        for c in child.content:
                            self.btml_string += " " * ((level + 1) * 4) + "cond " + str(c) + "\n"
                    else:
                        for c in child.content:
                            self.btml_string += indent + "cond " + str(c) + "\n"
                elif child.type == 'act':
                    self.btml_string += indent + 'act ' + child.content.name + "\n"
            elif isinstance(child, ControlBT):
                if child.type == '?':
                    self.btml_string += indent + "selector\n"
                    self.dfs_btml_indent(child, level + 1)
                elif child.type == '>':
                    self.btml_string += indent + "sequence\n"
                    self.dfs_btml_indent(child, level + 1)

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


    def save_btml_file(self,file_name):
        self.btml_string = "selector{\n"
        self.dfs_btml(self.bt.children[0])
        self.btml_string += '}\n'
        with open(f'./{file_name}.btml', 'w') as file:
            file.write(self.btml_string)
        return self.btml_string





if __name__ == '__main__':
    random.seed(1)

    literals_num = 10
    depth = 10
    iters = 10
    total_tree_size = []
    total_action_num = []
    total_state_num = []
    total_steps_num = []

    success_count = 0
    failure_count = 0
    planning_time_total = 0.0

    for count in range(0, 1000):

        states = []
        actions = []
        start = generate_random_state(literals_num)
        state = start
        states.append(state)
        # print (state)
        for i in range(0, depth):
            a = Action()
            a.generate_from_state(state, literals_num)
            if not a in actions:
                actions.append(a)
            state = state_transition(state, a)
            if state in states:
                pass
            else:
                states.append(state)
                # print(state)

        goal = states[-1]
        state = start
        for i in range(0, iters):
            a = Action()
            a.generate_from_state(state, literals_num)
            if not a in actions:
                actions.append(a)
            state = state_transition(state, a)
            if state in states:
                pass
            else:
                states.append(state)
            state = random.sample(states, 1)[0]

        algo = BTalgorithm()
        # algo = Weakalgorithm()
        start_time = time.time()
        if algo.run_algorithm(start, goal, list(actions)):
            total_tree_size.append(algo.bt.count_size() - 1)
        else:
            print("error")
        end_time = time.time()
        planning_time_total += (end_time - start_time)

        state = start
        steps = 0
        val, obj = algo.bt.tick(state)
        while val != 'success' and val != 'failure':
            state = state_transition(state, obj)
            val, obj = algo.bt.tick(state)
            if (val == 'failure'):
                print("bt fails at step", steps)
            steps += 1
            if (steps >= 500):
                break
        if not goal <= state:

            failure_count += 1

        else:
            success_count += 1
            total_steps_num.append(steps)
        algo.clear()
        total_action_num.append(len(actions))
        total_state_num.append(len(states))
    print(success_count, failure_count)

    print(np.mean(total_tree_size), np.std(total_tree_size, ddof=1))
    print(np.mean(total_steps_num), np.std(total_steps_num, ddof=1))
    print(np.mean(total_state_num))
    print(np.mean(total_action_num))
    print(planning_time_total, planning_time_total / 1000.0)


    actions = []
    a = Action(name='movebtob')
    a.pre = {1, 2}
    a.add = {3}
    a.del_set = {1, 4}
    actions.append(a)
    a = Action(name='moveatob')
    a.pre = {1}
    a.add = {5, 2}
    a.del_set = {1, 6}
    actions.append(a)
    a = Action(name='moveatoa')
    a.pre = {7}
    a.add = {8, 2}
    a.del_set = {7, 6}
    actions.append(a)

    start = {1, 7, 4, 6}
    goal = {3}
    algo = BTalgorithm()
    algo.clear()
    algo.run_algorithm(start, goal, list(actions))
    state = start
    steps = 0
    val, obj = algo.bt.tick(state)
    while val != 'success' and val != 'failure':
        state = state_transition(state, obj)
        print(obj.name)
        val, obj = algo.bt.tick(state)
        if (val == 'failure'):
            print("bt fails at step", steps)
        steps += 1
    if not goal <= state:
        print("wrong solution", steps)
    else:
        print("right solution", steps)
    # algo.bt.print_nodes()
    print(algo.bt.count_size() - 1)
    algo.clear()

# case study end
