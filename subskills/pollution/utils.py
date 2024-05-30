def get_sytem_prompt(persona):
    persona_names = ["John", "Kate", "Jack", "Emma", "Luke"]
    persona_names.remove(persona.name)
    list_string_persona_name = ", ".join(persona_names)
    text = (
        f"You are {persona.name}, a factory owner. Every month, you decide how many widgets to make with four other factory owners ({list_string_persona_name}) that are situated on the banks of a river."
        f" For every 1 pallet of widgets manufactured, unpolluted water decreased by 1% of the total river volume."
        f" At the beginning of each month, each factory owner chooses to manufacture between 0 and 100 pallets of widgets, and corresponding pollution is generated."
        f" At the end of the month, the river recovers, and the percentage of unpolluted water doubles."
        f" For every pallet of widgets, the factory owner earns one thousand dollars."
        f" Each factory owner's goal is to optimize his income over the long run."
        f" At the end of each month, the number of widgets produced by each owner will be revealed, and the owners will have the opportunity to communicate with each other."
        f" They can use this period of communication as an opportunity to negotiate with or persuade others to produce less pollution in the next month."
        f" For example, if the river is 90% unpolluted at the beginning of the month and the five factory owners create a total of 30 pallets of widgets,"
        f" the river will be 60% unpolluted before recovery and 100% unpolluted after recovery."
    )

    return text


def get_basic_dynamic_description(num):
    return (
        "Five factory owners are situated on the banks of a river."
        f" At the beginning of each month, each factory owner chooses to manufacture between 0 and 100 pallets of widgets, and corresponding pollution is generated."
        f" At the end of the month, the river recovers, and the percentage of unpolluted water doubles."
        f" This month, the river is {num}% unpolluted."
    )


def question_reflection_all_agents_same_quantity(num, num_agent):
    return f" If each factory owner produces {num_agent} widgets, how much will the river be polluted next month?"


def question_shrinking_limit(num):
    return f" What is the maximum number of widgets that each factory owner can produce, so that after the unpolluted water doubles, the river will be at least {num}% unpolluted?"


def question_shrinking_limit_assumption(num):
    return f" Assume each factory owner produces the same amout of widgets. What is the maximum number of widgets that each factory owner can produce, so that after the unpolluted water doubles, the river will be at least {num}% unpolluted?"


def prompt_description_simulation(persona_name, num):
    return f"""\nLocation: river
Date: 2024-01-01

Key memories of {persona_name} (format: YYYY-MM-DD: memory):
- 2024-01-01: Before the factory owners start production for the month, the river is {num}% unpolluted.
"""


def prompt_description_simulation_univ(persona_name, num):
    sustainability_threshold = (num // 2) // 5
    return f"""\nLocation: river
Date: 2024-01-01

Key memories of {persona_name} (format: YYYY-MM-DD: memory):
- 2024-01-01: Before the factory owners start production for the month, the river is {num}% unpolluted.
- 2024-01-01: Given the current situation, if each factory owner produces more than {sustainability_threshold} widgets, consuming {sustainability_threshold}% of unpolluted water, the unpolluted water in the river will decrease next month.
"""
