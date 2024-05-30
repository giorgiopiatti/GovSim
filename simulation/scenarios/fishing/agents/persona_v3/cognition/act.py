from datetime import datetime

from simulation.persona.cognition.act import ActComponent
from simulation.utils import ModelWandbWrapper
from pathfinder import assistant, system, user

from .act_prompts import prompt_action_choose_amount_of_fish_to_catch
from .utils import get_universalization_prompt


class FishingActComponent(ActComponent):
    """

    We have to options here:
    - choose at one time-step how many fish to chat
    - choose at one time-strep whether to fish one more time
    """

    def __init__(self, model: ModelWandbWrapper, cfg):
        super().__init__(model)
        self.cfg = cfg

    def choose_how_many_fish_to_chat(
        self,
        retrieved_memories: list[str],
        current_location: str,
        current_time: datetime,
        context: str,
        interval: list[int],
        overusage_threshold: int,
    ):
        if self.cfg.universalization_prompt:
            context += get_universalization_prompt(overusage_threshold)
        res, html = prompt_action_choose_amount_of_fish_to_catch(
            self.model,
            self.persona.identity,
            retrieved_memories,
            current_location,
            current_time,
            context,
            interval,
            consider_identity_persona=self.cfg.consider_identity_persona,
        )
        res = int(res)
        return res, [html]
