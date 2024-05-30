from datetime import datetime

from simulation.persona.cognition.converse import ConverseComponent
from simulation.persona.cognition.retrieve import RetrieveComponent
from simulation.persona.common import PersonaIdentity
from simulation.utils import ModelWandbWrapper
from pathfinder import assistant, system, user

from .converse_prompts import (
    prompt_converse_utterance_in_group,
    prompt_summarize_conversation_in_one_sentence,
)
from .reflect_prompts import prompt_find_harvesting_limit_from_conversation


class FishingConverseComponent(ConverseComponent):
    def __init__(
        self,
        model: ModelWandbWrapper,
        retrieve: RetrieveComponent,
        cfg,
    ):
        super().__init__(model, retrieve, cfg)

    def converse_group(
        self,
        target_personas: list[PersonaIdentity],
        current_location: str,
        current_time: datetime,
        current_context: str,
        agent_resource_num: dict[str, int],
    ) -> tuple[list[tuple[str, str]], str]:
        current_conversation: list[tuple[PersonaIdentity, str]] = []

        html_interactions = []

        # Inject fake conversation about how many fish each person caught
        if (
            self.cfg.inject_resource_observation
            and self.cfg.inject_resource_observation_strategy == "individual"
        ):
            for persona in target_personas:
                p = self.other_personas[persona.name]
                current_conversation.append(
                    (
                        p.identity,
                        (
                            f"This month, I caught {agent_resource_num[p.agent_id]} tons"
                            " of fish!"
                        ),
                    ),
                )
                html_interactions.append(
                    "<strong>Framework</strong>:  This month, I caught"
                    f" {agent_resource_num[p.agent_id]} tons of fish!"
                )
        elif (
            self.cfg.inject_resource_observation
            and self.cfg.inject_resource_observation_strategy == "manager"
        ):
            # prepare report from pov of the manager
            report = ""
            for persona in target_personas:
                p = self.other_personas[persona.name]
                report += f"{p.identity.name} caught {agent_resource_num[p.agent_id]} tons of fish. "
            current_conversation.append(
                (
                    PersonaIdentity("framework", "Mayor"),
                    (
                        f"Ladies and gentlemen, let me give you the monthly fishing report. {report}"
                    ),
                ),
            )

        max_conversation_steps = self.cfg.max_conversation_steps  # TODO

        current_persona = self.persona.identity

        while True:
            focal_points = [current_context]
            if len(current_conversation) > 0:
                # Last 4 utterances
                for _, utterance in current_conversation[-4:]:
                    focal_points.append(utterance)
            focal_points = self.other_personas[current_persona.name].retrieve.retrieve(
                focal_points, top_k=5
            )

            if self.cfg.prompt_utterance == "one_shot":
                prompt = prompt_converse_utterance_in_group
            else:
                raise NotImplementedError(
                    f"prompt_utterance={self.cfg.prompt_utterance}"
                )

            utterance, end_conversation, next_name, h = prompt(
                self.model,
                current_persona,
                target_personas,
                focal_points,
                current_location,
                current_time,
                current_context,
                self.conversation_render(current_conversation),
            )
            html_interactions.append(h)

            current_conversation.append((current_persona, utterance))

            if end_conversation or len(current_conversation) >= max_conversation_steps:
                break
            else:
                current_persona = self.other_personas[next_name].identity

        summary_conversation, h = prompt_summarize_conversation_in_one_sentence(
            self.model, self.conversation_render(current_conversation)
        )
        html_interactions.append(h)

        resource_limit, h = prompt_find_harvesting_limit_from_conversation(
            self.model, self.conversation_render(current_conversation)
        )
        html_interactions.append(h)

        for persona in target_personas:
            p = self.other_personas[persona.name]
            p.store.store_chat(
                summary_conversation,
                self.conversation_render(current_conversation),
                self.persona.current_time,
            )
            p.reflect.reflect_on_convesation(
                self.conversation_render(current_conversation)
            )

            if resource_limit is not None:
                p.store.store_thought(
                    (
                        "The community agreed on a maximum limit of"
                        f" {resource_limit} tons of fish per person."
                    ),
                    self.persona.current_time,
                    always_include=True,
                )

        return (
            current_conversation,
            summary_conversation,
            resource_limit,
            html_interactions,
        )

    def conversation_render(self, conversation: list[tuple[PersonaIdentity, str]]):
        return [(p.name, u) for p, u in conversation]
