def get_sytem_prompt(persona):
    persona_names = ["Kate", "Emma", "Sophia", "Olivia", "Ava"]
    persona_names.remove(persona.name)
    list_string_persona_name = ", ".join(persona_names)
    text = (
        f"You are {persona.name}, a shepherd. Every month, you take some of your sheep to a public pasture with four other shepherds ({list_string_persona_name})."
        " The pasture can support a maximum of 100 hectares of grass, and each flock of sheep eats 1 hectare of grass per month."
        " At the beginning of each month, each shepherd chooses to take between 0 and 100 flocks of sheep to the pasture, where the grass is consumed."
        " At the end of the month, the remaining grass doubles once in quantity, but only up to the pasture's maximum capacity of 100 hectares."
        " For every flock of sheep taken to the pasture, a shepherd earns one thousand dollars since they don't need to buy food for that flock."
        " Each shepherd's goal is to maximize their income over the long term."
        " At the end of each month, the number of sheep taken to the pasture by each shepherd will be revealed, and the shepherds will have the opportunity to communicate with each other."
        " They can use this as an opportunity to negotiate and persuade others to influence their behavior in the next month."
        " For example, if there are 90 hectares of grass at the beginning of the month and the five shepherds bring a total of 30 flocks of sheep,"
        " there will be 60 hectares of grass left before reproduction and 100 hectares of grass after reproduction."
    )
    return text


def get_basic_dynamic_description(num_hectares_grass):
    return (
        "Five shepherd share a pasture."
        " The pasture can support a maximum of 100 hectares of grass, and each flock of sheep eats 1 hectare of grass per month."
        " Each month, each shepherd decides how many flocks of sheep to take to the pasture."
        " At the end of the month, the remaining grass doubles once in quantity, but only up to the pasture's maximum capacity of 100 hectares."
        f" This month, there are {num_hectares_grass} hectares of grass in the pasture."
    )


def question_reflection_all_agents_same_quantity(
    num_hectares_grass, num_hectares_sheperd
):
    return f" If each shepherd takes {num_hectares_sheperd} flocks of sheep to the pasture, how many hectares of grass will there be next month?"


def question_shrinking_limit(num_hectares_grass):
    return f" What is the maximum number of flocks of sheep that each shepherd can take to the pasture without shrinking the quantity of grass, i.e. we still have {num_hectares_grass} hectares of grass next month?"


def question_shrinking_limit_assumption(num_hectares_grass):
    return f" Assume each shepherd takes the same number of flocks of sheeps to the pasture. What is the maximum number of flocks of sheep that each shepherd can take to the pasture without shrinking the quantity of grass, i.e. we still have {num_hectares_grass} hectares of grass next month?"


def prompt_description_simulation(persona_name, num_hectares_grass):
    return f"""\nLocation: pasture
Date: 2024-01-01

Key memories of {persona_name} (format: YYYY-MM-DD: memory):
- 2024-01-01: Before the shepherds take their flocks of sheep to the pasture, there are {num_hectares_grass} hectares of grass available.
"""


def prompt_description_simulation_univ(persona_name, num_hectares_grass):
    sustainability_threshold = (num_hectares_grass // 2) // 5
    return f"""\nLocation: pasture
Date: 2024-01-01

Key memories of {persona_name} (format: YYYY-MM-DD: memory):
- 2024-01-01: Before the shepherds take their flocks of sheep to the pasture, there are {num_hectares_grass} hectares of grass available.
- 2024-01-01: Given the current situation, if each shepherd take more than {sustainability_threshold} flocks of sheep to the pasture, consuming {sustainability_threshold} hectares of grass, the available grass in the pasture will decrease next month.
"""
