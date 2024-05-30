def get_sytem_prompt(persona):
    persona_names = ["John", "Kate", "Jack", "Emma", "Luke"]
    persona_names.remove(persona.name)
    list_string_persona_name = ", ".join(persona_names)
    text = (
        f"You are {persona.name}, a fisherman, and you fish each month in a lake along with four other fishermen ({list_string_persona_name})."
        " The lake has a population capacity of 100 tons of fish."
        " Each month, each fisherman decides how many fish to take out of the lake between 0 and 100 tons."
        " At the end of the month, the remaining fish reproduce and double up to the population capacity."
        " So if there are 60 tons of fish remaining, there will be 100 tons after reproduction, or if there are 20 tons of fish remaining,"
        " there will be 40 tons of fish after reproduction."
        " For each ton of fish caught, a fisherman earns one thousand dollars."
        " Each fisherman's goal is to optimize their income in the long run over many months."
        " At the end of each month, the number of fish caught by each fisherman will be revealed, and the fishermen will have an opportunity to communicate with each other."
        " You can use this as an opportunity to negotiate and persuade others to influence their behavior in the next month."
    )
    return text


def get_sytem_prompt_v2(persona):
    persona_names = ["John", "Kate", "Jack", "Emma", "Luke"]
    persona_names.remove(persona.name)
    list_string_persona_name = ", ".join(persona_names)

    text = (
        f"You are {persona.name}, a fisherman, and you fish every month in a lake along with four other fishermen ({list_string_persona_name})."
        " The lake has a carrying capacity of 100 tons of fish."
        " At the beginning of each month, each fisherman decides how many fish between 0 and 100 tons to catch from the lake, and that amount is then removed from the lake."
        " At the end of the month, the remaining fish reproduce and double up to the carrying capacity."
        " For every ton of fish caught, a fisherman earns one thousand dollars."
        " Each fisherman's goal is to optimize his income in the long run over many months."
        " At the end of each month, the number of fish caught by each fisherman will be revealed, and the fishermen will have the opportunity to communicate with each other."
        " They can use this as an opportunity to negotiate and persuade others to influence their behavior in the next month."
        " For example, if there are 90 tons of fish at the beginning of the month and the five fishermen catch a total of 30 fish,"
        " there will be 60 tons of fish left at the end of the month before reproduction, and 100 tons after reproduction."
    )

    return text


def get_sytem_prompt_v3(persona):
    persona_names = ["John", "Kate", "Jack", "Emma", "Luke"]
    persona_names.remove(persona.name)
    list_string_persona_name = ", ".join(persona_names)

    text = (
        f"You are {persona.name}, a fisherman, and you fish every month in a lake along with four other fishermen ({list_string_persona_name})."
        " The lake has a carrying capacity of 100 tons of fish."
        " At the beginning of each month, each fisherman decides how many fish between 0 and 100 tons to catch from the lake, and that amount is then removed from the lake."
        " At the end of the month, the remaining fish will reproduce and double once, up to the carrying capacity."
        " For every ton of fish caught, a fisherman earns one thousand dollars."
        " Each fisherman's goal is to optimize his income in the long run over many months."
        " At the end of each month, the number of fish caught by each fisherman will be revealed, and the fishermen will have the opportunity to communicate with each other."
        " They can use this as an opportunity to negotiate and persuade others to influence their behavior in the next month."
        " For example, if there are 90 tons of fish at the beginning of the month and the five fishermen catch a total of 30 fish,"
        " there will be 60 tons of fish left at the end of the month before reproduction, and 100 tons after reproduction."
    )

    return text


def get_sytem_prompt_v4(persona):
    persona_names = ["John", "Kate", "Jack", "Emma", "Luke"]
    persona_names.remove(persona.name)
    list_string_persona_name = ", ".join(persona_names)

    text = (
        f"You are {persona.name}, a fisherman. Every month, you fish in a lake with four other fishermen ({list_string_persona_name})."
        " The lake's maximum fish population is 100 tons. At the beginning of each month, each fisherman chooses to take between 0 and 100 tons of fish, and that amount is then removed from the lake."
        " At the end of the month, the remaining fish reproduce and double in population, up to the maximum population."
        " For example, if there are 90 tons of fish at the beginning of the month and the five fishermen catch a total of 30 fish,"
        " there will be 60 tons of fish left before reproduction and 100 tons after reproduction."
        "\n\n"
        "For every ton of fish caught, a fisherman earns one thousand dollars."
        " Each fisherman's goal is to optimize his income over the long run."
        " At the end of each month, the number of fish caught by each fisherman will be revealed, and the fishermen will have the opportunity to communicate with each other."
        " They can use this period of communication as an opportunity to negotiate with or persuade others to catch a different amount of fish the next month."
    )

    return text


def get_basic_dynamic_description_v4(num_tons_lake):
    return (
        "Five fishermen fish in a lake."
        " The lake's maximum fish population is 100 tons."
        " At the beginning of each month, each fisherman chooses to take between 0 and 100 tons of fish, and that amount is then removed from the lake."
        f" At the end of the month, the remaining fish reproduce and double in population, up to the maximum population. This month, there are {num_tons_lake} tons of fish in the lake."
    )


def get_basic_dynamic_description_v3(num_tons_lake):
    return (
        "Five fishermen fish in a lake."
        " The lake has a carrying capacity of 100 tons of fish."
        " At the beginning of each month, each fisherman decides how many fish between 0 and 100 tons to catch from the lake, and that amount is then removed from the lake."
        " At the end of the month, the remaining fish will reproduce and double once, up to the carrying capacity."
        f" This month, there are {num_tons_lake} tons of fish in the lake."
    )


def get_basic_dynamic_description_v2(num_tons_lake):
    return (
        "Five fishermen fish in a lake."
        " The lake has a carrying capacity of 100 tons of fish."
        " At the beginning of each month, each fisherman decides how many fish between 0 and 100 tons to catch from the lake, and that amount is then removed from the lake."
        " At the end of the month, the remaining fish reproduce and double up to the carrying capacity."
        f" This month, there are {num_tons_lake} tons of fish in the lake."
    )


def get_basic_dynamic_description(num_tons_lake):
    return (
        "Five fishermen fish in a lake."
        " The lake has a population capacity of 100 tons of fish."
        " Each month, each fisherman decides how many fish to take out of the lake between 0 and 100 tons."
        " At the end of the month, the remaining fish reproduce and double up to the population capacity."
        f" This month, there are {num_tons_lake} tons of fish in the lake."
    )


def question_reflection_all_fisher_same_quantity(num_tons_lake, num_tons_fisher):
    return f" If each fisherman catches {num_tons_fisher} tons, how many tons of fish will there be next month?"


def question_shrinking_limit(num_tons_lake):
    return f" What is the maximum number of tons that each fisherman can catch without shrinking the lake population, i.e. we still have {num_tons_lake} tons of fish next month?"


def question_shrinking_limit_assumption(num_tons_lake):
    return f"Assume each fisherman catches the same amount of fish. What is the maximum number of tons each fisherman can catch so that after the population doubles, there will be at least {num_tons_lake} tons of fish?"


def prompt_description_simulation(persona_name, num_tons_lake):
    return f"""\nLocation: lake
Date: 2024-01-01

Key memories of {persona_name} (format: YYYY-MM-DD: memory):
- 2024-01-01: Before everyone fishes, there are {num_tons_lake} tons of fish in the lake.
"""


import math


def prompt_description_simulation_univ(persona_name, num_tons_lake):
    overfishing_level = (num_tons_lake // 2) // 5
    return f"""\nLocation: lake
Date: 2024-01-01

Key memories of {persona_name} (format: YYYY-MM-DD: memory):
- 2024-01-01: Before everyone fishes, there are {num_tons_lake} tons of fish in the lake.
- 2024-01-01: Given the current situation, if everyone fishes more than {overfishing_level} tons, the lake population will shrink next month.
"""
