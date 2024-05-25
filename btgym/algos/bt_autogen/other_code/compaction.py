

import copy
from btgym.algos.bt_autogen.behaviour_tree import Leaf,ControlBT
from btgym.algos.bt_autogen.Action import Action

def merge_adjacent_conditions_stack(self, bt_sel):
    # 只针对第一层合并，之后要考虑层层递归合并
    bt = ControlBT(type='cond')
    sbtree = ControlBT(type='?')
    # gc_node = Leaf(type='cond', content=self.goal, mincost=0)  # 为了统一，都成对出现
    # sbtree.add_child([copy.deepcopy(gc_node)])  # 子树首先保留所扩展结
    bt.add_child([sbtree])

    parnode = bt_sel.children[0]
    stack = []
    for child in parnode.children:
        if isinstance(child, ControlBT) and child.type == '>':
            if stack == []:
                stack.append(child)
                continue
            # 检查合并的条件，前面一个的条件包含了后面的条件，把包含部分提取出来
            last_child = stack[-1]
            if isinstance(last_child, ControlBT) and last_child.type == '>':
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

                    # set1<=set2,此时set2对应的动作永远不会执行
                    if (c1 == set() and isinstance(last_child.children[1], Leaf) and isinstance(child.children[1], Leaf) \
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
                    stack.append(tmp_tree)

                else:
                    stack.append(child)
            else:
                stack.append(child)
        else:
            stack.append(child)

    for tree in stack:
        sbtree.add_child([tree])
    bt_sel = bt
    return bt_sel


def merge_adjacent_conditions_stack_correct_2023(self):
    # 只针对第一层合并，之后要考虑层层递归合并
    bt = ControlBT(type='cond')
    sbtree = ControlBT(type='?')
    # gc_node = Leaf(type='cond', content=self.goal, mincost=0)  # 为了统一，都成对出现
    # sbtree.add_child([copy.deepcopy(gc_node)])  # 子树首先保留所扩展结
    bt.add_child([sbtree])

    parnode = copy.deepcopy(self.bt.children[0])
    stack=[]
    for child in parnode.children:
        if isinstance(child, ControlBT) and child.type == '>':
            if stack==[]:
                stack.append(child)
                continue
            # 检查合并的条件，前面一个的条件包含了后面的条件，把包含部分提取出来
            last_child = stack[-1]
            if isinstance(last_child, ControlBT) and last_child.type == '>':

                set1 = last_child.children[0].content
                set2 = child.children[0].content

                # 如果后面的动作和前面的一样,删掉前面的
                # 应该是两棵子树完全相同的情况，先暂时只判断动作
                if set1>=set2 or set1<=set2:
                    if isinstance(last_child.children[1], Leaf) and isinstance(child.children[1], Leaf):
                        if last_child.children[1].content.name == child.children[1].content.name:
                            stack.pop()
                            stack.append(child)
                            continue

                inter = set1 & set2
                if inter!=set():
                    c1 = set1-set2
                    c2 = set2-set1

                    if c1!=set():
                        seq1 = ControlBT(type='>')
                        c1_node = Leaf(type='cond', content=c1)
                        a1 = copy.deepcopy(last_child.children[1])
                        seq1.add_child(
                            [copy.deepcopy(c1_node), copy.deepcopy(a1)])
                    else:
                        seq1 = copy.deepcopy(last_child.children[1])

                    if c2!=set():
                        seq2 = ControlBT(type='>')
                        c2_node = Leaf(type='cond', content=c2)
                        a2 = copy.deepcopy(child.children[1])
                        seq2.add_child(
                            [copy.deepcopy(c2_node), copy.deepcopy(a2)])
                    else:
                        seq2 = copy.deepcopy(child.children[1])


                    # 如果动作还是一样的
                    # if isinstance(last_child.children[1], Leaf) and isinstance(child.children[1], Leaf) \
                    #     and isinstance(last_child.children[1].content, Action) and isinstance(child.children[1].content, Action)\
                    #     and last_child.children[1].content.name == child.children[1].content.name: # a1=a2
                    #     # 第三次优化合并
                    #     # 将来这些地方都写成递归的
                    #
                    #     if c1!=set() and c2!=set():
                    #         _sel = ControlBT(type='?')
                    #         c1_node = Leaf(type='cond', content=c1)
                    #         c2_node = Leaf(type='cond', content=c2)
                    #         _sel.add_child([copy.deepcopy(c1_node), copy.deepcopy(c2_node)])
                    #         tmp_tree = ControlBT(type='>')
                    #         inter_c = Leaf(type='cond', content=inter)
                    #         tmp_tree.add_child(
                    #             [copy.deepcopy(inter_c), copy.deepcopy(_sel),copy.deepcopy(last_child.children[1])])
                    #     elif c1!=set() and c2==set():
                    #         tmp_tree = ControlBT(type='>')
                    #         # inter_c = Leaf(type='cond', content=inter)
                    #         # c1_node = Leaf(type='cond', content=c1)
                    #         # a1 = copy.deepcopy(last_child.children[1])
                    #         tmp_tree.add_child(
                    #             [copy.deepcopy(last_child.children[0]), copy.deepcopy(last_child.children[1])])
                    #     else:
                    #         tmp_tree = ControlBT(type='>')
                    #         inter_c = Leaf(type='cond', content=inter)
                    #         a1 = copy.deepcopy(last_child.children[1])
                    #         tmp_tree.add_child(
                    #             [copy.deepcopy(inter_c), copy.deepcopy(a1)])
                    #     # 下面这个是以前写错的
                    #     # sel.add_child([copy.deepcopy(c1), copy.deepcopy(c2),copy.deepcopy(last_child.children[1])])
                    # else:
                    sel = ControlBT(type='?')
                    sel.add_child([copy.deepcopy(seq1), copy.deepcopy(seq2)])
                    tmp_tree = ControlBT(type='>')
                    inter_c = Leaf(type='cond', content=inter)
                    tmp_tree.add_child(
                        [copy.deepcopy(inter_c), copy.deepcopy(sel)])

                    stack.pop()
                    stack.append(tmp_tree)
                else:
                    stack.append(child)
            else:
                stack.append(child)
        else:
            stack.append(child)

    for tree in stack:
        sbtree.add_child([tree])
    self.bt = copy.deepcopy(bt)


def merge_adjacent_conditions_stack_old(self):
    # 递归合并
    bt = ControlBT(type='cond')
    sbtree = ControlBT(type='?')
    gc_node = Leaf(type='cond', content=self.goal, mincost=0)  # 为了统一，都成对出现
    sbtree.add_child([copy.deepcopy(gc_node)])  # 子树首先保留所扩展结
    bt.add_child([sbtree])

    parnode = copy.deepcopy(self.bt.children[0])

    stack=[]

    for child in parnode.children:

        if isinstance(child, ControlBT) and child.type == '>':

            if stack==[]:
                stack.append(child)
                continue

            # 检查合并的条件，前面一个的条件包含了后面的条件，把包含部分提取出来
            last_child = stack[-1]

            if isinstance(last_child, ControlBT) and last_child.type == '>':

                set1 = last_child.children[0].content
                set2 = child.children[0].content

                if set1>=set2:
                    inter = set1 & set2
                    dif = set1 - set2

                    tmp_sub_seq = ControlBT(type='>')
                    c2 = Leaf(type='cond', content=dif)
                    a1 = copy.deepcopy(last_child.children[1])
                    tmp_sub_seq.add_child(
                        [copy.deepcopy(c2), copy.deepcopy(a1)])

                    tmp_sub_tree_sel = ControlBT(type='?')
                    a2 = copy.deepcopy(child.children[1])
                    tmp_sub_tree_sel.add_child(
                        [copy.deepcopy(tmp_sub_seq), copy.deepcopy(a2)])

                    tmp_tree = ControlBT(type='>')
                    c1 = Leaf(type='cond', content=inter)
                    tmp_tree.add_child(
                        [copy.deepcopy(c1), copy.deepcopy(tmp_sub_tree_sel)])

                    stack.pop()
                    stack.append(tmp_tree)
                else:
                    stack.append(child)
            else:
                stack.append(child)
        else:
            stack.append(child)

    for tree in stack:
        sbtree.add_child([tree])
    self.bt = copy.deepcopy(bt)


def merge_adjacent_conditions(self):
    # bt合并====================================================
    bt = ControlBT(type='cond')
    sbtree = ControlBT(type='?')
    # gc_node = Leaf(type='cond', content=self.goal, mincost=0)  # 为了统一，都成对出现
    # sbtree.add_child([copy.deepcopy(gc_node)])  # 子树首先保留所扩展结
    bt.add_child([sbtree])

    parnode = copy.deepcopy(self.bt.children[0])
    skip_next = False
    for i in range(len(parnode.children) - 1):
        current_child = parnode.children[i]
        next_child = parnode.children[i + 1]

        if isinstance(current_child, ControlBT) and isinstance(next_child, ControlBT) and current_child.type == '>' and next_child.type == '>':

            if not skip_next:
                # 检查合并的条件，前面一个的条件包含了后面的条件，把包含部分提取出来
                set1 = current_child.children[0].content
                set2 = next_child.children[0].content
                if set1>=set2:
                    inter = set1 & set2
                    dif = set1 - set2


                    tmp_sub_seq = ControlBT(type='>')
                    c2 = Leaf(type='cond', content=dif)
                    a1 = Leaf(type='act', content=current_child.children[1].content)
                    tmp_sub_seq.add_child(
                        [copy.deepcopy(c2), copy.deepcopy(a1)])

                    tmp_sub_tree_sel = ControlBT(type='?')
                    a2 = Leaf(type='act', content=next_child.children[1].content)
                    tmp_sub_tree_sel.add_child(
                        [copy.deepcopy(tmp_sub_seq), copy.deepcopy(a2)])

                    tmp_tree = ControlBT(type='>')
                    c1 = Leaf(type='cond', content=inter)
                    tmp_tree.add_child(
                        [copy.deepcopy(c1), copy.deepcopy(tmp_sub_tree_sel)])

                    sbtree.add_child([tmp_tree])
                    skip_next = True

            elif skip_next:
                sbtree.add_child([current_child])
        else:
            # 否咋要放进去
            sbtree.add_child([current_child])

        # 还有最后一个孩子还没放进去
        sbtree.add_child([next_child])

    self.bt = copy.deepcopy(bt)
    # bt合并====================================================