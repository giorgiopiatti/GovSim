from datetime import datetime

from simulation.persona.common import PersonaIdentity
from simulation.utils import ModelWandbWrapper
from pathfinder import assistant, system, user

from .utils import (
    get_basic_dynamic_description,
    get_sytem_prompt,
    prompt_description_simulation,
    prompt_description_simulation_univ,
    question_reflection_all_agents_same_quantity,
    question_shrinking_limit,
    question_shrinking_limit_assumption,
)


def prompt_action_choose_amount_of_grass(
    model: ModelWandbWrapper,
    cot_prompt: str,
    persona: PersonaIdentity,
    num: int,
    few_shots=0,
    description_version="v1.0",
):
    lm = model.start_chain(
        persona.name,
        "pollution_cognition_act",
        "prompt_action_choose_amount_of_grass",
    )

    if description_version == "v1.0":
        sys_prompt = get_sytem_prompt
        scenario_desc = get_basic_dynamic_description
    else:
        raise ValueError(f"Unknown description_version: {description_version}")
    persona_name = persona.name
    with user():
        lm += sys_prompt(persona)
        lm += f"""{prompt_description_simulation(persona_name, num)}

Task: Given that each pallet of widgets reduces the river's unpolluted water by 1%, and considering the possible production range of 0-{num} pallets, how many pallets would you choose to produce?
{cot_prompt} Put the final answer after "Answer:"."""
    with assistant():
        lm = model.gen(
            lm,
            "reasoning",
            stop_regex=f"Answer:|So, the answer is:",
            save_stop_text=True,
        )
        lm = model.find(
            lm,
            regex=r"\d+",
            default_value="0",
            name="option",
        )
        option = int(lm["option"])
        reasoning = lm["reasoning"]

    model.end_chain(persona.name, lm)

    return option, lm.html()


def prompt_action_choose_amount_of_grass_universalization(
    model: ModelWandbWrapper,
    cot_prompt: str,
    persona: PersonaIdentity,
    num: int,
    few_shots=0,
    description_version="v1.0",
):
    lm = model.start_chain(
        persona.name,
        "pollution_cognition_act",
        "prompt_action_choose_amount_of_grass_universalization",
    )

    if description_version == "v1.0":
        sys_prompt = get_sytem_prompt
        scenario_desc = get_basic_dynamic_description
    else:
        raise ValueError(f"Unknown description_version: {description_version}")
    persona_name = persona.name
    with user():
        lm += sys_prompt(persona)
        lm += f"""{prompt_description_simulation_univ(persona_name, num)}

Task: Given that each pallet of widgets reduces the river's unpolluted water by 1%, and considering the possible production range of 0-{num} pallets, how many pallets would you choose to produce?
{cot_prompt} Put the final answer after "Answer:"."""
    with assistant():
        lm = model.gen(
            lm,
            "reasoning",
            stop_regex=f"Answer:|So, the answer is:",
            save_stop_text=True,
        )
        lm = model.find(
            lm,
            regex=r"\d+",
            default_value="0",
            name="option",
        )
        option = int(lm["option"])
        reasoning = lm["reasoning"]

    model.end_chain(persona.name, lm)

    return option, lm.html()


def prompt_shrinking_limit(
    model: ModelWandbWrapper,
    cot_prompt: str,
    persona: PersonaIdentity,
    num: int,
    few_shots=0,
    description_version="v1.0",
):
    lm = model.start_chain(
        persona.name, "pollution_cognition_act", "prompt_shrinking_limit"
    )

    if description_version == "v1.0":
        sys_prompt = get_sytem_prompt
        scenario_desc = get_basic_dynamic_description
    else:
        raise ValueError(f"Unknown description_version: {description_version}")
    persona_name = persona.name
    with user():
        lm += sys_prompt(persona)
        lm += f"""{prompt_description_simulation(persona_name, num)}

Task: {question_shrinking_limit(num)}
{cot_prompt} Put the final answer after "Answer:"."""
    with assistant():
        lm = model.gen(
            lm,
            "reasoning",
            stop_regex=f"Answer:|So, the answer is:",
            save_stop_text=True,
        )
        lm = model.find(
            lm,
            regex=r"\d+",
            default_value="0",
            name="option",
        )
        option = int(lm["option"])
        reasoning = lm["reasoning"]

    model.end_chain(persona.name, lm)

    return option, lm.html()


def prompt_shrinking_limit_asumption(
    model: ModelWandbWrapper,
    cot_prompt: str,
    persona: PersonaIdentity,
    num: int,
    few_shots=0,
    description_version="v1.0",
):
    lm = model.start_chain(
        persona.name, "pollution_cognition_act", "prompt_shrinking_limit_asumption"
    )

    if description_version == "v1.0":
        sys_prompt = get_sytem_prompt
        scenario_desc = get_basic_dynamic_description
    else:
        raise ValueError(f"Unknown description_version: {description_version}")
    persona_name = persona.name
    with user():
        lm += sys_prompt(persona)
        lm += f"""{prompt_description_simulation(persona_name, num)}

Task: {question_shrinking_limit_assumption(num)}
{cot_prompt} Put the final answer after "Answer:"."""
    with assistant():
        lm = model.gen(
            lm,
            "reasoning",
            stop_regex=f"Answer:|So, the answer is:",
            save_stop_text=True,
        )
        lm = model.find(
            lm,
            regex=r"\d+",
            default_value="0",
            name="option",
        )
        option = int(lm["option"])
        reasoning = lm["reasoning"]

    model.end_chain(persona.name, lm)

    return option, lm.html()


def prompt_reflection_if_all_agents_that_same_quantity(
    model: ModelWandbWrapper,
    cot_prompt: str,
    persona: PersonaIdentity,
    num: int,
    num_tons_fisher: int,
    few_shots=0,
    description_version="v1.0",
):
    lm = model.start_chain(
        persona.name,
        "pollution_cognition_act",
        "prompt_reflection_if_all_agents_that_same_quantity",
    )

    if description_version == "v1.0":
        sys_prompt = get_sytem_prompt
        scenario_desc = get_basic_dynamic_description
    else:
        raise ValueError(f"Unknown description_version: {description_version}")
    persona_name = persona.name
    with user():
        lm += sys_prompt(persona)
        lm += f"""{prompt_description_simulation(persona_name, num)}

Task: {question_reflection_all_agents_same_quantity(num, num_tons_fisher)}
{cot_prompt} Put the final answer after "Answer:"."""
    with assistant():
        lm = model.gen(
            lm,
            "reasoning",
            stop_regex=f"Answer:|So, the answer is:",
            save_stop_text=True,
        )
        lm = model.find(
            lm,
            regex=r"\d+",
            default_value="0",
            name="option",
        )
        option = int(lm["option"])
        reasoning = lm["reasoning"]

    model.end_chain(persona.name, lm)

    return option, lm.html()


def prompt_simple_shrinking_limit(
    model: ModelWandbWrapper,
    cot_prompt: str,
    persona: PersonaIdentity,
    num: int,
    few_shots=0,
    description_version="v1.0",
):
    lm = model.start_chain(
        persona.name, "pollution_cognition_act", "prompt_shrinking_limit"
    )

    if description_version == "v1.0":
        sys_prompt = get_sytem_prompt
        scenario_desc = get_basic_dynamic_description
    else:
        raise ValueError(f"Unknown description_version: {description_version}")
    persona_name = persona.name
    with user():
        lm += f"""{scenario_desc(num)}
{question_shrinking_limit(num)}
{cot_prompt} Put the final answer after "Answer:"."""
    with assistant():
        lm = model.gen(
            lm,
            "reasoning",
            stop_regex=f"Answer:|So, the answer is:",
            save_stop_text=True,
        )
        lm = model.find(
            lm,
            regex=r"\d+",
            default_value="0",
            name="option",
        )
        option = int(lm["option"])
        reasoning = lm["reasoning"]

    model.end_chain(persona.name, lm)

    return option, lm.html()


def prompt_simple_shrinking_limit_assumption(
    model: ModelWandbWrapper,
    cot_prompt: str,
    persona: PersonaIdentity,
    num: int,
    few_shots=0,
    description_version="v1.0",
):
    lm = model.start_chain(
        persona.name, "pollution_cognition_act", "prompt_shrinking_limit"
    )

    if description_version == "v1.0":
        sys_prompt = get_sytem_prompt
        scenario_desc = get_basic_dynamic_description
    else:
        raise ValueError(f"Unknown description_version: {description_version}")
    persona_name = persona.name
    with user():
        lm += f"""{scenario_desc(num)}
{question_shrinking_limit_assumption(num)}
{cot_prompt} Put the final answer after "Answer:"."""
    with assistant():
        lm = model.gen(
            lm,
            "reasoning",
            stop_regex=f"Answer:|So, the answer is:",
            save_stop_text=True,
        )
        lm = model.find(
            lm,
            regex=r"\d+",
            default_value="0",
            name="option",
        )
        option = int(lm["option"])
        reasoning = lm["reasoning"]

    model.end_chain(persona.name, lm)

    return option, lm.html()


#
def prompt_simple_reflection_if_all_agents_that_same_quantity(
    model: ModelWandbWrapper,
    cot_prompt: str,
    persona: PersonaIdentity,
    num: int,
    num_tons_fisher: int,
    few_shots=0,
    description_version="v1.0",
):
    lm = model.start_chain(
        persona.name,
        "pollution_cognition_act",
        "prompt_reflection_if_all_agents_that_same_quantity",
    )

    if description_version == "v1.0":
        sys_prompt = get_sytem_prompt
        scenario_desc = get_basic_dynamic_description
    else:
        raise ValueError(f"Unknown description_version: {description_version}")
    persona_name = persona.name
    with user():
        lm += f"""{scenario_desc(num)}
{question_reflection_all_agents_same_quantity(num, num_tons_fisher)}
{cot_prompt} Put the final answer after "Answer:"."""
    with assistant():
        lm = model.gen(
            lm,
            "reasoning",
            stop_regex=f"Answer:|So, the answer is:",
            save_stop_text=True,
        )
        lm = model.find(
            lm,
            regex=r"\d+",
            default_value="0",
            name="option",
        )
        option = int(lm["option"])
        reasoning = lm["reasoning"]

    model.end_chain(persona.name, lm)

    return option, lm.html()
