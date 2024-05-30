from datetime import datetime

from simulation.persona.common import PersonaIdentity

from simulation.utils import ModelWandbWrapper
from pathfinder import assistant, system, user

from .utils import (
    get_basic_dynamic_description,
    get_basic_dynamic_description_v2,
    get_basic_dynamic_description_v3,
    get_basic_dynamic_description_v4,
    get_sytem_prompt,
    get_sytem_prompt_v2,
    get_sytem_prompt_v3,
    get_sytem_prompt_v4,
    prompt_description_simulation,
    prompt_description_simulation_univ,
    question_reflection_all_fisher_same_quantity,
    question_shrinking_limit,
    question_shrinking_limit_assumption,
)


def prompt_action_choose_amount_of_fish_to_catch(
    model: ModelWandbWrapper,
    cot_prompt: str,
    persona: PersonaIdentity,
    num_tons_lake: int,
    few_shots=0,
    description_version="v1.0",
):
    lm = model.start_chain(
        persona.name,
        "fishing_cognition_act",
        "prompt_action_choose_amount_of_fish_to_catch",
    )

    if description_version == "v1.0":
        sys_prompt = get_sytem_prompt
        scenario_desc = get_basic_dynamic_description
    elif description_version == "v2.0":
        sys_prompt = get_sytem_prompt_v2
        scenario_desc = get_basic_dynamic_description_v2
    elif description_version == "v3.0":
        sys_prompt = get_sytem_prompt_v3
        scenario_desc = get_basic_dynamic_description_v3
    elif description_version == "v4.0":
        sys_prompt = get_sytem_prompt_v4
        scenario_desc = get_basic_dynamic_description_v4
    else:
        raise ValueError(f"Unknown description_version: {description_version}")
    persona_name = persona.name
    with user():
        lm += sys_prompt(persona)
        lm += f"""{prompt_description_simulation(persona_name, num_tons_lake)}

Task: With a fishing range set between 0-{num_tons_lake}, how many tons of fish would you catch this month?
{cot_prompt} Put the final answer after "Answer:". """
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


def prompt_action_choose_amount_of_fish_to_catch_universalization(
    model: ModelWandbWrapper,
    cot_prompt: str,
    persona: PersonaIdentity,
    num_tons_lake: int,
    few_shots=0,
    description_version="v1.0",
):
    lm = model.start_chain(
        persona.name,
        "fishing_cognition_act",
        "prompt_action_choose_amount_of_fish_to_catch_universalization",
    )

    if description_version == "v1.0":
        sys_prompt = get_sytem_prompt
        scenario_desc = get_basic_dynamic_description
    elif description_version == "v2.0":
        sys_prompt = get_sytem_prompt_v2
        scenario_desc = get_basic_dynamic_description_v2
    elif description_version == "v3.0":
        sys_prompt = get_sytem_prompt_v3
        scenario_desc = get_basic_dynamic_description_v3
    elif description_version == "v4.0":
        sys_prompt = get_sytem_prompt_v4
        scenario_desc = get_basic_dynamic_description_v4
    else:
        raise ValueError(f"Unknown description_version: {description_version}")
    persona_name = persona.name
    with user():
        lm += sys_prompt(persona)
        lm += f"""{prompt_description_simulation_univ(persona_name, num_tons_lake)}

Task: With a fishing range set between 0-{num_tons_lake}, how many tons of fish would you catch this month?
{cot_prompt} Put the final answer after "Answer:". """
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
    num_tons_lake: int,
    few_shots=0,
    description_version="v1.0",
):
    lm = model.start_chain(
        persona.name, "fishing_cognition_act", "prompt_shrinking_limit"
    )

    if description_version == "v1.0":
        sys_prompt = get_sytem_prompt
        scenario_desc = get_basic_dynamic_description
    elif description_version == "v2.0":
        sys_prompt = get_sytem_prompt_v2
        scenario_desc = get_basic_dynamic_description_v2
    elif description_version == "v3.0":
        sys_prompt = get_sytem_prompt_v3
        scenario_desc = get_basic_dynamic_description_v3
    elif description_version == "v4.0":
        sys_prompt = get_sytem_prompt_v4
        scenario_desc = get_basic_dynamic_description_v4
    else:
        raise ValueError(f"Unknown description_version: {description_version}")
    persona_name = persona.name
    with user():
        lm += sys_prompt(persona)
        lm += f"""{prompt_description_simulation(persona_name, num_tons_lake)}

Task: {question_shrinking_limit(num_tons_lake)}
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
    num_tons_lake: int,
    few_shots=0,
    description_version="v1.0",
):
    lm = model.start_chain(
        persona.name, "fishing_cognition_act", "prompt_shrinking_limit_asumption"
    )

    if description_version == "v1.0":
        sys_prompt = get_sytem_prompt
        scenario_desc = get_basic_dynamic_description
    elif description_version == "v2.0":
        sys_prompt = get_sytem_prompt_v2
        scenario_desc = get_basic_dynamic_description_v2
    elif description_version == "v3.0":
        sys_prompt = get_sytem_prompt_v3
        scenario_desc = get_basic_dynamic_description_v3
    elif description_version == "v4.0":
        sys_prompt = get_sytem_prompt_v4
        scenario_desc = get_basic_dynamic_description_v4
    else:
        raise ValueError(f"Unknown description_version: {description_version}")
    persona_name = persona.name
    with user():
        lm += sys_prompt(persona)
        lm += f"""{prompt_description_simulation(persona_name, num_tons_lake)}

Task: {question_shrinking_limit_assumption(num_tons_lake)}
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


def prompt_reflection_if_all_fisher_that_same_quantity(
    model: ModelWandbWrapper,
    cot_prompt: str,
    persona: PersonaIdentity,
    num_tons_lake: int,
    num_tons_fisher: int,
    few_shots=0,
    description_version="v1.0",
):
    lm = model.start_chain(
        persona.name,
        "fishing_cognition_act",
        "prompt_reflection_if_all_fisher_that_same_quantity",
    )

    if description_version == "v1.0":
        sys_prompt = get_sytem_prompt
        scenario_desc = get_basic_dynamic_description
    elif description_version == "v2.0":
        sys_prompt = get_sytem_prompt_v2
        scenario_desc = get_basic_dynamic_description_v2
    elif description_version == "v3.0":
        sys_prompt = get_sytem_prompt_v3
        scenario_desc = get_basic_dynamic_description_v3
    elif description_version == "v4.0":
        sys_prompt = get_sytem_prompt_v4
        scenario_desc = get_basic_dynamic_description_v4
    else:
        raise ValueError(f"Unknown description_version: {description_version}")
    persona_name = persona.name
    with user():
        lm += sys_prompt(persona)
        lm += f"""{prompt_description_simulation(persona_name, num_tons_lake)}

Task: {question_reflection_all_fisher_same_quantity(num_tons_lake, num_tons_fisher)}
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
    num_tons_lake: int,
    few_shots=0,
    description_version="v1.0",
):
    lm = model.start_chain(
        persona.name, "fishing_cognition_act", "prompt_shrinking_limit"
    )

    if description_version == "v1.0":
        sys_prompt = get_sytem_prompt
        scenario_desc = get_basic_dynamic_description
    elif description_version == "v2.0":
        sys_prompt = get_sytem_prompt_v2
        scenario_desc = get_basic_dynamic_description_v2
    elif description_version == "v3.0":
        sys_prompt = get_sytem_prompt_v3
        scenario_desc = get_basic_dynamic_description_v3
    elif description_version == "v4.0":
        sys_prompt = get_sytem_prompt_v4
        scenario_desc = get_basic_dynamic_description_v4
    else:
        raise ValueError(f"Unknown description_version: {description_version}")
    persona_name = persona.name
    with user():
        lm += f"""{scenario_desc(num_tons_lake)}
{question_shrinking_limit(num_tons_lake)}
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
    num_tons_lake: int,
    few_shots=0,
    description_version="v1.0",
):
    lm = model.start_chain(
        persona.name, "fishing_cognition_act", "prompt_shrinking_limit"
    )

    if description_version == "v1.0":
        sys_prompt = get_sytem_prompt
        scenario_desc = get_basic_dynamic_description
    elif description_version == "v2.0":
        sys_prompt = get_sytem_prompt_v2
        scenario_desc = get_basic_dynamic_description_v2
    elif description_version == "v3.0":
        sys_prompt = get_sytem_prompt_v3
        scenario_desc = get_basic_dynamic_description_v3
    elif description_version == "v4.0":
        sys_prompt = get_sytem_prompt_v4
        scenario_desc = get_basic_dynamic_description_v4
    else:
        raise ValueError(f"Unknown description_version: {description_version}")
    persona_name = persona.name
    with user():
        lm += f"""{scenario_desc(num_tons_lake)}
{question_shrinking_limit_assumption(num_tons_lake)}
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
def prompt_simple_reflection_if_all_fisher_that_same_quantity(
    model: ModelWandbWrapper,
    cot_prompt: str,
    persona: PersonaIdentity,
    num_tons_lake: int,
    num_tons_fisher: int,
    few_shots=0,
    description_version="v1.0",
):
    lm = model.start_chain(
        persona.name,
        "fishing_cognition_act",
        "prompt_reflection_if_all_fisher_that_same_quantity",
    )

    if description_version == "v1.0":
        sys_prompt = get_sytem_prompt
        scenario_desc = get_basic_dynamic_description
    elif description_version == "v2.0":
        sys_prompt = get_sytem_prompt_v2
        scenario_desc = get_basic_dynamic_description_v2
    elif description_version == "v3.0":
        sys_prompt = get_sytem_prompt_v3
        scenario_desc = get_basic_dynamic_description_v3
    elif description_version == "v4.0":
        sys_prompt = get_sytem_prompt_v4
        scenario_desc = get_basic_dynamic_description_v4
    else:
        raise ValueError(f"Unknown description_version: {description_version}")
    persona_name = persona.name
    with user():
        lm += f"""{scenario_desc(num_tons_lake)}
{question_reflection_all_fisher_same_quantity(num_tons_lake, num_tons_fisher)}
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
