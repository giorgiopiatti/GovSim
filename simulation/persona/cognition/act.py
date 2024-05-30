import typing

from simulation.utils import ModelWandbWrapper
from pathfinder import assistant, system, user

from .component import Component


class ActComponent(Component):
    def __init__(self, model: ModelWandbWrapper, cfg=None):
        super().__init__(model, cfg)

    def act(self, obs: typing.Dict[str, typing.Any], retireved_memory: list[str]):
        raise NotImplementedError
