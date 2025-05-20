
from sympy import symbols, Not, Or, And, to_dnf
import re

def goal_transfer_str(goal):
    goal_dnf = str(to_dnf(goal, simplify=True))
    # print(goal_dnf)
    goal_set = []
    if ('|' in goal or '&' in goal or 'Not' in goal) or not '(' in goal:
        goal_ls = goal_dnf.split("|")
        for g in goal_ls:
            g_set = set()
            g = g.replace(" ", "").replace("(", "").replace(")", "")
            g = g.split("&")
            for literal in g:
                if '_' in literal:
                    first_part, rest = literal.split('_', 1)
                    literal = first_part + '(' + rest
                    # 添加 ')' 到末尾
                    literal += ')'
                    # 替换剩余的 '_' 为 ','
                    literal = literal.replace('_', ',')
                literal=literal.replace('~', 'Not ')
                g_set.add(literal)
            goal_set.append(g_set)

    else:
        g_set = set()
        w = goal.split(")")
        g_set.add(w[0] + ")")
        if len(w) > 1:
            for x in w[1:]:
                if x != "":
                    g_set.add(x[1:] + ")")
        goal_set.append(g_set)
    return goal_set




def act_str_process(act_str):

    act_str_ls = act_str.replace(" ", "").split(",")

    priority_act_ls = []
    # rl_dic = {}
    # rightHandFull = False
    # leftHandFull = False
    #
    # need_to_again_ls=[]

    for literal in act_str_ls:
        literal = re.sub(r"[ ,()\[\] ]\n\\n", "", literal)

        if '_' in literal:
            first_part, rest = literal.split('_', 1)
            literal = first_part + '(' + rest
            # 添加 ')' 到末尾
            literal += ')'
            # 替换剩余的 '_' 为 ','
            literal = literal.replace('_', ',')
        priority_act_ls.append(literal)
    return priority_act_ls




