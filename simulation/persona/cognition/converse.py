from simulation.utils import ModelWandbWrapper
from pathfinder import assistant, system, user

from ..common import PersonaIdentity
from .component import Component
from .retrieve import RetrieveComponent
from .store import StoreComponent


class ConverseComponent(Component):
    """
    Everything related to the conversation between personas.

    Utterance: a single thing said by a persona, then the other persona may respond.
    Conversation: a sequence of utterances between personas.

    """

    def __init__(self, model: ModelWandbWrapper, retrieve: RetrieveComponent, cfg=None):
        super().__init__(model, cfg)
        self.retrieve = retrieve
