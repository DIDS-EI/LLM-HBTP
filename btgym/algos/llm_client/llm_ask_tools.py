import re
from btgym.algos.llm_client.tools import goal_transfer_str, act_str_process
from btgym.utils import ROOT_PATH
# 导入向量数据库检索的相关函数
from btgym.algos.llm_client.vector_database_env_goal import search_nearest_examples
from ordered_set import OrderedSet


def parse_llm_output(answer,goals=True):
    goal_set = set()
    priority_act_ls, key_predicate, key_objects = [], [], []

    try:
        if goals:
            goal_str = answer.split("Optimal Actions:")[0].replace("Goals:", "").strip()
            goal_set = goal_transfer_str(goal_str)

        act_str = answer.split("Optimal Actions:")[1].split("Vital Action Predicates:")[0].strip()
        predicate_str = answer.split("Vital Action Predicates:")[1].split("Vital Objects:")[0].strip()
        objects_str = answer.split("Vital Objects:")[1].strip()
        priority_act_ls = act_str_process(act_str)

        # Remove all spaces, Split by comma to create a list
        key_predicate = predicate_str.replace(" ", "").split(",")
        key_objects = objects_str.replace(" ", "").split(",")

        priority_act_ls = list(OrderedSet(priority_act_ls))
        key_predicate = list(OrderedSet(key_predicate))
        key_objects = list(OrderedSet(key_objects))
    except Exception as e:
        goal_set, priority_act_ls, key_predicate, key_objects = None,None,None,None
        print(f"Failed to parse LLM output: {e}")
    return goal_set, priority_act_ls, key_predicate, key_objects

def convert_conditions(conditions_set):
    # Initialize an empty list to store the formatted strings
    formatted_conditions = []

    # Loop over each condition in the set
    for condition in conditions_set:
        # Remove the parentheses and split the condition into parts based on the first opening parenthesis
        base, args = condition.split("(")
        # Remove the closing parenthesis and replace commas with underscores in the arguments
        args = args.strip(")").replace(",", "_")
        # Concatenate the base and the arguments with an underscore and add to the list
        formatted_conditions.append(f"{base.strip()}_{args}")

    formatted_conditions_str = " & ".join(formatted_conditions)
    return formatted_conditions_str


def extract_llm_from_instr_goal(llm, default_prompt_file, goals, cur_cond_set=None, verbose=False, messages=[]):
    with open(default_prompt_file, 'r', encoding="utf-8") as f:
        prompt = f.read().strip()

    parsed_output =None
    parsed_fail=-1
    RED = "\033[31m"
    RESET = "\033[0m"

    while parsed_output==None:

        parsed_fail += 1 # 第一次是第0次。 0-1-2-3
        print(f"--- LLM: Goal={goals}  Parsed Fail={parsed_fail} --- ")
        if parsed_fail > 3:
            print(f"{RED}----LLM: Goal={goals}  Parsed Fail={parsed_fail} >3 break -----{RESET}")
            break
        goals_str =' & '.join(goals)

        question = f"{prompt}\nGoals: {goals_str}"
        if verbose:
            print("============ Question ================\n",question)
        messages.append({"role": "user", "content": question})
        answer = llm.request(message=messages)
        messages.append({"role": "assistant", "content": answer})


        parsed_output = parse_llm_output(answer, goals=False)


    goal_set,priority_act_ls, key_predicates, key_objects = parsed_output

    cyan = "\033[36m"
    reset = "\033[0m"
    print(f"{cyan}============ Answer ================{reset}")
    print(f"{cyan}priority_act_ls: {', '.join(priority_act_ls)}{reset}")
    print(f"{cyan}key_predicates: {', '.join(key_predicates)}{reset}")
    print(f"{cyan}key_objects: {', '.join(key_objects)}{reset}")

    if priority_act_ls==None:
        print(f"\033[91mFailed to parse LLM output for goals: {goals_str}\033[0m")
    return priority_act_ls, key_predicates, key_objects, messages, parsed_fail





def llm_reflect(llm, messages, reflect_prompt):
    messages.append({"role": "user", "content": reflect_prompt})
    answer = llm.request(message=messages)
    messages.append({"role": "assistant", "content": answer})

    print("============ Answer ================\n",answer)

    goal_set, priority_act_ls, key_predicates, key_objects = parse_llm_output(answer)

    print("goal",goal_set)
    print("act:",priority_act_ls)
    print("key_predicate",key_predicates)
    print("Vital Objects:",key_objects)

    return goal_set, priority_act_ls, key_predicates, key_objects, messages


def convert_conditions(conditions_set):
    # Initialize an empty list to store the formatted strings
    formatted_conditions = []

    # Loop over each condition in the set
    for condition in conditions_set:
        # Remove the parentheses and split the condition into parts based on the first opening parenthesis
        base, args = condition.split("(")
        # Remove the closing parenthesis and replace commas with underscores in the arguments
        args = args.strip(")").replace(",", "_")
        # Concatenate the base and the arguments with an underscore and add to the list
        formatted_conditions.append(f"{base.strip()}_{args}")

    formatted_conditions_str = " & ".join(formatted_conditions)
    return formatted_conditions_str


def extract_llm_from_reflect(llm,messages):

    answer = llm.request(message=messages)
    messages.append({"role": "assistant", "content": answer})
    _, priority_act_ls, key_predicates, key_objects = parse_llm_output(answer,goals=False) # 返回的都是list

    # cyan = "\033[36m"
    # reset = "\033[0m"
    # print(f"{cyan}--- Reflect Just LLM ---{reset}")
    # print(f"{cyan}priority_act_ls: {', '.join(priority_act_ls)}{reset}")
    # print(f"{cyan}key_predicates: {', '.join(key_predicates)}{reset}")
    # print(f"{cyan}key_objects: {', '.join(key_objects)}{reset}")


    cyan = "\033[36m"
    reset = "\033[0m"
    print(f"{cyan}--- Reflect ---{reset}")
    print(f"{cyan}priority_act_ls: {', '.join(priority_act_ls)}{reset}")
    print(f"{cyan}key_predicates: {', '.join(key_predicates)}{reset}")
    print(f"{cyan}key_objects: {', '.join(key_objects)}{reset}")

    return priority_act_ls, key_predicates, key_objects, messages