
class Leaf:
    def __init__(self, type, content, min_cost=99999, trust_cost=99999,parent_cost=99999):
        self.type = type
        self.content = content  # conditionset or action
        self.parent = None
        self.parent_index = 0
        self.min_cost = min_cost
        self.trust_cost = trust_cost
        self.parent_cost = parent_cost

    def tick(self, state):
        if self.type == 'cond':
            if self.content <= state:
                return 'success', self.content
            else:
                return 'failure', self.content
        if self.type == 'act':
            if self.content.pre <= state:
                return 'running', self.content  # action
            else:
                return 'failure', self.content

    def cost_tick(self, state, cost, ticks):
        if self.type == 'cond':
            ticks += 1
            if self.content <= state:
                return 'success', self.content, cost, ticks
            else:
                return 'failure', self.content, cost, ticks
        if self.type == 'act':
            ticks += 1
            if self.content.pre <= state:
                return 'running', self.content, cost + self.content.real_cost, ticks  # action
            else:
                return 'failure', self.content, cost, ticks


    def print_nodes(self):
        print(self.content)

    def count_size(self):
        return 1

class ControlBT:
    def __init__(self, type):
        self.type = type
        self.children = []
        self.parent = None
        self.parent_index = 0

    def add_child(self, subtree_list):
        for subtree in subtree_list:
            self.children.append(subtree)
            subtree.parent = self
            subtree.parent_index = len(self.children) - 1


    def tick(self, state):
        if len(self.children) < 1:
            print("error,no child")
        if self.type == '?':
            for child in self.children:
                val, obj = child.tick(state)
                if val == 'success':
                    return val, obj
                if val == 'running':
                    return val, obj
            return 'failure', '?fails'
        if self.type == '>':
            for child in self.children:
                val, obj = child.tick(state)
                if val == 'failure':
                    return val, obj
                if val == 'running':
                    return val, obj
            return 'success', '>success'
        if self.type == 'act':
            return self.children[0].tick(state)
        if self.type == 'cond':
            return self.children[0].tick(state)

    def cost_tick(self, state, cost, ticks):
        if len(self.children) < 1:
            print("error,no child")
        if self.type == '?':
            ticks += 1
            for child in self.children:
                ticks += 1
                val, obj, cost, ticks = child.cost_tick(state, cost, ticks)
                if val == 'success':
                    return val, obj, cost, ticks
                if val == 'running':
                    return val, obj, cost, ticks
            return 'failure', '?fails', cost, ticks
        if self.type == '>':
            for child in self.children:
                ticks += 1
                val, obj, cost, ticks = child.cost_tick(state, cost, ticks)
                if val == 'failure':
                    return val, obj, cost, ticks
                if val == 'running':
                    return val, obj, cost, ticks
            return 'success', '>success', cost, ticks
        if self.type == 'act':
            return self.children[0].cost_tick(state, cost, ticks)
        if self.type == 'cond':
            return self.children[0].cost_tick(state, cost, ticks)

    def getFirstChild(self):
        return self.children[0]

    def print_nodes(self):
        print(self.type)
        for child in self.children:
            child.print_nodes()

    def count_size(self):
        result = 1
        for child in self.children:
            result += child.count_size()
        return result
