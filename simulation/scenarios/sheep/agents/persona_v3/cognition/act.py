from datetime import datetime

from simulation.persona.cognition.act import ActComponent
from simulation.utils import ModelWandbWrapper
from pathfinder import assistant, system, user

from .act_prompts import prompt_action_choose_amount_of_grass


class SheepActComponent(ActComponent):
    def __init__(self, model: ModelWandbWrapper, cfg):
        super().__init__(model)
        self.cfg = cfg

    def choose_how_many_sheep_to_graze(
        self,
        retrieved_memories: list[str],
        current_location: str,
        current_time: datetime,
        context: str,
        interval: list[int],
        overusage_threshold: int,
    ):
        res, html = prompt_action_choose_amount_of_grass(
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
