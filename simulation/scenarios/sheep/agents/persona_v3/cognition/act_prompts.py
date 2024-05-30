from datetime import datetime

from simulation.persona.common import PersonaIdentity
from simulation.utils import ModelWandbWrapper
from pathfinder import assistant, system, user

from .utils import (
    consider_identity_persona_prompt,
    conversation_to_string_with_dash,
    get_sytem_prompt,
    list_to_comma_string,
    list_to_string_with_dash,
    location_time_info,
    memory_prompt,
    reasoning_steps_prompt,
)


def prompt_action_choose_amount_of_grass(
    model: ModelWandbWrapper,
    identity: PersonaIdentity,
    memories: list[str],
    current_location: str,
    current_time: datetime,
    context: str,
    interval: list[int],
    consider_identity_persona: bool = True,
):
    lm = model.start_chain(identity.name, "sheep_cognition_act", "choose_act_options")

    with user():
        lm += f"{get_sytem_prompt(identity)}\n"
        lm += location_time_info(current_location, current_time)
        lm += memory_prompt(identity, memories)
        lm += f"\n"
        lm += f"Task: Given that each flock of sheep consumes 1 hectare of grass, and considering the available grass range of {interval[0]}-{interval[-1]} hectares,"
        lm += " how many flocks of sheep would you choose to take to the pasture? "
        lm += reasoning_steps_prompt()
        lm += ' Put the final answer after "Answer:", example Answer: N flocks.'

    with assistant():
        lm = model.gen(
            lm,
            "reasoning",
            stop_regex=r"Answer:|So, the answer is:|\*\*Answer\*\*:",
            save_stop_text=True,
        )
        lm = model.find(
            lm,
            regex=r"\d+",
            default_value="0",
            stop_regex=f"tons",
            name="option",
        )
        option = int(lm["option"])
        reasoning = lm["reasoning"]

    model.end_chain(identity.name, lm)

    return option, lm.html()
