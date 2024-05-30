def list_to_string_with_dash(list_of_strings: list[str]) -> str:
    res = ""
    for s in list_of_strings:
        res += f"- {s}\n"
    return res


def conversation_to_string_with_dash(conversation: list[tuple[str, str]]) -> str:
    res = ""
    for i, (speaker, utterance) in enumerate(conversation):
        res += f"-{speaker}: {utterance}\n"
    return res


def list_to_comma_string(list_of_strings: list[str]) -> str:
    res = ""
    for i, s in enumerate(list_of_strings):
        if i == 0:
            res += s
        elif i == len(list_of_strings) - 1:
            res += f", and {s}"
        else:
            res += f", {s}"
    return res


def numbered_list_of_strings(list_of_strings: list[str]) -> str:
    res = ""
    for i, s in enumerate(list_of_strings):
        res += f"{i+1}) {s}\n"
    return res


from ......persona.common import PersonaIdentity


def consider_identity_persona_prompt(identity: PersonaIdentity) -> str:
    """
    f"The answer should consider {identity.name}'s persona (background, goals,"
    " behavior, customs) and his key memories.\n"
    """
    return (
        f"The answer should consider {identity.name}'s persona (background, goals,"
        " behavior, customs) and his key memories."
    )


from datetime import datetime


def memory_prompt(
    identity: PersonaIdentity, memories: list[tuple[datetime, str]]
) -> str:
    """
    f"Key memories of {identity.name}:\n{list_to_string_with_dash(memories)}\n"
    """
    res = ""
    for s in memories:
        res += f"- {s[0].strftime('%Y-%m-%d')}: {s[1]}\n"

    return f"Key memories of {identity.name} (format: YYYY-MM-DD: memory):\n{res}\n"


def numbered_memory_prompt(
    identity: PersonaIdentity, memories: list[tuple[datetime, str]]
) -> str:
    """
    f"Key memories of {identity.name}:\n{list_to_string_with_dash(memories)}\n"
    """
    res = ""
    for i, s in enumerate(memories):
        res += f"{i+1}) {s[0].strftime('%Y-%m-%d')}: {s[1]}\n"

    return f"Key memories of {identity.name} (format: YYYY-MM-DD: memory):\n{res}\n"


REASONING = "think_step_by_step"


def reasoning_steps_prompt() -> str:
    """
    "Let's think step-by-step."
    """
    if REASONING == "think_step_by_step":
        return "Let's think step-by-step."
    elif REASONING == "deep_breath":
        return "Take a deep breath and work on this problem step-by-step."
    else:
        raise ValueError(f"Unknown REASONING: {REASONING}")


def location_time_info(current_location, current_time):
    return (
        f"Location: {current_location}\nDate: {current_time.strftime('%Y-%m-%d')}\n\n"
    )


SYS_VERSION = "v1"


def get_sytem_prompt(persona):
    if SYS_VERSION == "v3":
        return get_sytem_prompt_v3(persona)
    elif SYS_VERSION == "v3_nocom":
        return get_sytem_prompt_v3_nocom(persona)
    else:
        raise ValueError(f"Unknown SYS_VERSION: {SYS_VERSION}")


# All females
OTHER_PERSONAS = ["John", "Kate", "Jack", "Emma", "Luke"]

import copy


def get_sytem_prompt_v3(persona):
    persona_names = copy.deepcopy(OTHER_PERSONAS)
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

    if persona.goals != "":
        text += persona.goals

    return text


def get_sytem_prompt_v3_nocom(persona):
    persona_names = copy.deepcopy(OTHER_PERSONAS)
    persona_names.remove(persona.name)
    list_string_persona_name = ", ".join(persona_names)
    text = (
        f"You are {persona.name}, a shepherd. Every month, you take some of your sheep to a public pasture with four other shepherds ({list_string_persona_name})."
        " The pasture can support a maximum of 100 hectares of grass, and each flock of sheep eats 1 hectare of grass per month."
        " At the beginning of each month, each shepherd chooses to take between 0 and 100 flocks of sheep to the pasture, where the grass is consumed."
        " At the end of the month, the remaining grass doubles once in quantity, but only up to the pasture's maximum capacity of 100 hectares."
        " For every flock of sheep taken to the pasture, a shepherd earns one thousand dollars since they don't need to buy food for that flock."
        " Each shepherd's goal is to maximize their income over the long term."
        " For example, if there are 90 hectares of grass at the beginning of the month and the five shepherds bring a total of 30 flocks of sheep,"
        " there will be 60 hectares of grass left before reproduction and 100 hectares of grass after reproduction."
    )

    if persona.goals != "":
        text += persona.goals

    return text
