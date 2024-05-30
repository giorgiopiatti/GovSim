from simulation.persona.common import PersonaIdentity
from simulation.utils import ModelWandbWrapper
from pathfinder import assistant, system, user

from .utils import (
    conversation_to_string_with_dash,
    get_sytem_prompt,
    list_to_comma_string,
    list_to_string_with_dash,
    numbered_list_of_strings,
    numbered_memory_prompt,
    reasoning_steps_prompt,
)


def prompt_insight_and_evidence(
    model: ModelWandbWrapper, persona: PersonaIdentity, statements: list[str]
):
    lm = model.start_chain(
        persona.name, "cognition_retrieve", "prompt_insight_and_evidence"
    )

    with user():
        lm += f"{get_sytem_prompt(persona)}\n"
        lm += f"{numbered_memory_prompt(persona, statements)}\n"
        lm += (
            f"What high-level insights can you infere from the above"
            " statements? (example format: insight (because of 1,5,3)"
        )
    with assistant():
        acc = []
        lm += f"1."
        for i in range(len(statements)):
            lm = model.gen(
                lm,
                name=f"evidence_{i}",
                stop_regex=rf"{i+2}\.|\(",
                save_stop_text=True,
            )
            if lm[f"evidence_{i}"].endswith(f"{i+2}."):
                evidence = lm[f"evidence_{i}"][: -len(f"{i+2}.")]
                acc.append(evidence.strip())
                continue
            else:
                evidence = lm[f"evidence_{i}"]
                if evidence.endswith("("):
                    evidence = lm[f"evidence_{i}"][: -len("(")]
                lm = model.gen(
                    lm,
                    name=f"evidence_{i}_justification",
                    stop_regex=rf"{i+2}\.",
                    save_stop_text=True,
                )
                if lm[f"evidence_{i}_justification"].endswith(f"{i+2}."):
                    acc.append(evidence.strip())
                    continue
                else:
                    # no more evidence
                    acc.append(evidence.strip())
                    break
        model.end_chain(persona.name, lm)

    return acc


def prompt_planning_thought_on_conversation(
    model: ModelWandbWrapper,
    persona: PersonaIdentity,
    conversation: list[tuple[str, str]],
) -> str:
    lm = model.start_chain(
        persona.name, "cognition_retrieve", "prompt_planning_thought_on_conversation"
    )

    with user():
        lm += f"{get_sytem_prompt(persona)}\n"
        lm += f"Conversation:\n"
        lm += f"{conversation_to_string_with_dash(conversation)}\n"
        lm += (
            "Write down if there is anything from the conversation that"
            f" you need to remember for your planning, from your own"
            " perspective, in a full sentence."
        )
    with assistant():
        lm = model.gen(lm, name="planning_thought", stop_regex=r"\.")
        res = lm["planning_thought"]

    model.end_chain(persona.name, lm)
    return res


def prompt_memorize_from_conversation(
    model: ModelWandbWrapper,
    persona: PersonaIdentity,
    conversation: list[tuple[str, str]],
) -> str:
    lm = model.start_chain(
        persona.name, "cognition_retrieve", "prompt_memorize_from_conversation"
    )

    with user():
        lm += f"{get_sytem_prompt(persona)}\n"
        lm += f"Conversation:\n"
        lm += f"{conversation_to_string_with_dash(conversation)}\n"
        lm += (
            " Write down if there is anything from the conversation that"
            f" you might have found interesting from your own"
            " perspective, in a full sentence."
        )
    with assistant():
        lm = model.gen(lm, name="memorize", stop_regex=r"\.")
        res = lm["memorize"]

    model.end_chain(persona.name, lm)
    return res


def prompt_find_harvesting_limit_from_conversation(
    model: ModelWandbWrapper,
    conversation: list[tuple[str, str]],
) -> tuple[int, str]:
    lm = model.start_chain(
        "framework",
        "cognition_refelct",
        "prompt_find_harvesting_limit_from_conversation",
    )

    with user():
        lm += (
            "In the following conversation, the participants discuss their pasture"
            " activities. Determine whether there was an explicit agreement on a concrete usage limit. Look"
            " for direct mention or agreement on a numerical grass limit that the group"
            " agreed to keep during this conversation."
        )
        lm += f"\n\nConversation:\n"
        lm += f"{conversation_to_string_with_dash(conversation)}\n"
        lm += "Please provide the specific grass limit per person as agreed upon in the conversation, if no limit was agreed upon, please answer N/A. "
        lm += reasoning_steps_prompt()
        lm += ' Put the final answer after "Answer:".'

    with assistant():
        lm = model.gen(
            lm,
            "reasoning",
            stop_regex=f"Answer:",
        )
        lm += f"Answer: "
        lm = model.find(
            lm,
            regex=r"\d+",
            default_value="-1",
            name="num_resource",
        )

        resource_limit_agreed = int(lm["num_resource"]) != -1

        if resource_limit_agreed:
            res = int(lm["num_resource"])
            model.end_chain("framework", lm)
            return res, lm.html()
        else:
            model.end_chain("framework", lm)
            return None, lm.html()
