import typing

from pathfinder import assistant, system, user
from simulation.utils import ModelWandbWrapper

from .component import Component


class ActComponent(Component):
    def __init__(
        self, model: ModelWandbWrapper, model_framework: ModelWandbWrapper, cfg=None
    ):
        super().__init__(model, model_framework, cfg)

    def act(self, obs: typing.Dict[str, typing.Any], retireved_memory: list[str]):
        raise NotImplementedError
