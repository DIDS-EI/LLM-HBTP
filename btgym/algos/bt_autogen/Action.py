
import copy
import random


# Define action categories, which include prerequisites, adding and deleting impacts
class Action:
    def __init__(self,name='anonymous action',pre=set(),add=set(),del_set=set(),cost=10,vaild_num=0,vild_args=set()):
        self.pre=copy.deepcopy(pre)
        self.add=copy.deepcopy(add)
        self.del_set=copy.deepcopy(del_set)
        self.name=name
        self.real_cost=cost
        self.cost=cost
        self.priority = cost
        self.vaild_num=vaild_num
        self.vild_args = vild_args

    def __str__(self):
        return self.name

    def generate_from_state_local(self,state,literals_num_set,all_obj_set=set(),obj_num=0,obj=None):

        pre_num = random.randint(0, len(state))
        self.pre = set(random.sample(state, pre_num))

        add_set = literals_num_set - self.pre
        add_num = random.randint(0, len(add_set))
        self.add = set(random.sample(add_set, add_num))

        del_set = literals_num_set - self.add
        del_num = random.randint(0, len(del_set))
        self.del_set = set(random.sample(del_set, del_num))

        if all_obj_set!=set():
            self.vaild_num = random.randint(1, obj_num-1)
            self.vild_args = (set(random.sample(all_obj_set, self.vaild_num)))
            if obj!=None:
                self.vild_args.add(obj)
                self.vaild_num = len(self.vild_args)

    def update(self,name,pre,del_set,add):
        self.name = name
        self.pre = pre
        self.del_set = del_set
        self.add = add
        return self


    def print_action(self):
        print (self.pre)
        print(self.add)
        print(self.del_set)

def generate_random_state(num):
    result = set()
    for i in range(0,num):
        if random.random()>0.5:
            result.add(i)
    return result

def state_transition(state,action):
    if not action.pre <= state:
        print ('error: action not applicable')
        return state
    new_state=(state | action.add) - action.del_set
    return new_state