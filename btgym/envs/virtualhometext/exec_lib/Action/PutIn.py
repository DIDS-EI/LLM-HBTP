from btgym.envs.virtualhometext.exec_lib._base.VHTAction import VHTAction
from btgym.envs.virtualhometext.exec_lib._base.VHTAction_small import VHTAction_small
import itertools

class PutIn(VHTAction):
    can_be_expanded = False
    num_args = 2
    valid_args = list(itertools.product(VHTAction.GRABBABLE, VHTAction.CONTAINERS))

    set_1_food = VHTAction.GRABBABLE & (VHTAction.EATABLE|VHTAction.DRINKABLE|{"apple","bananas",'chicken','cutlets','breadslice','chips','chocolatesyrup',
                 'milk','wine',"cereal","plate","lime","salmon", "peach","pear","plum"})
    set_2_cloth =  VHTAction.GRABBABLE & {"clothespile","clothesshirt","clothespants"}
    # 食物只能放冰箱或者加热的地方，其它东西都不能放在这里面，只有衣服能洗或者放在衣服堆里

    # 食物不能放的地方 "washingmachine","dishwasher","printer","folder","closet","clothespile"
    # 其它物品不能放的地方 "fridge","microwave","stove","closet","clothespile","washingmachine","dishwasher","printer","folder"
    # 衣服不能放的地方 fridge","microwave","stove","dishwasher","printer","folder"
    # 只有碗能放的地方 "dishwasher"
    # 只有纸能放的地方 "printer","folder"
    valid_args = set(list(itertools.product(set_1_food, \
                                        VHTAction.CONTAINERS - {"washingmachine","dishwasher","printer","folder","closet","clothespile"})) \
                 + list(itertools.product(VHTAction.GRABBABLE-set_1_food-set_2_cloth,\
                    VHTAction.CONTAINERS-{"fridge","microwave","stove","fryingpan","closet","clothespile","washingmachine","dishwasher","printer","folder"})) \
                + list(itertools.product(set_2_cloth,VHTAction.CONTAINERS-{"fridge","microwave","stove","fryingpan","dishwasher","printer","folder"})) \
            + list(itertools.product(VHTAction.GRABBABLE & {"dishbowl","plate"}, {"dishwasher"})) \
            + list(itertools.product(VHTAction.GRABBABLE & {"paper"}, {"printer","folder"})) \
            + list(itertools.product(VHTAction.GRABBABLE & {"papertowel"}, {"garbagecan"})))
    # Convert set back to a list
    valid_args = list(valid_args)


    set_1_food_small = VHTAction_small.GRABBABLE & (VHTAction_small.EATABLE|VHTAction_small.DRINKABLE|{"bananas",'chicken','cutlets','breadslice','chips','chocolatesyrup',
                 'milk','wine',"cereal"})
    set_2_cloth_small =  VHTAction_small.GRABBABLE & {"clothespile","clothesshirt","clothespants"}

    valid_args_small = set(list(itertools.product(set_1_food_small, \
                                        VHTAction_small.CONTAINERS - {"washingmachine","dishwasher","printer","folder","closet","clothespile"})) \
                 + list(itertools.product(VHTAction_small.GRABBABLE-set_1_food_small-set_2_cloth_small,\
                    VHTAction_small.CONTAINERS-{"fridge","microwave","stove","fryingpan","closet","clothespile","washingmachine","dishwasher","printer","folder"})) \
                + list(itertools.product(set_2_cloth_small,VHTAction_small.CONTAINERS-{"fridge","microwave","stove","fryingpan","dishwasher","printer","folder"})) \
            + list(itertools.product(VHTAction_small.GRABBABLE & {"dishbowl"}, {"dishwasher"})) \
            + list(itertools.product(VHTAction_small.GRABBABLE & {"paper"}, {"printer","folder"})))
    valid_args_small = list(valid_args_small)
    def __init__(self, *args):
        super().__init__(*args)
