import datetime

from simulation.utils import ModelWandbWrapper

from ..common import (
    ChatObservation,
    PersonaEvent,
    PersonaIdentity,
    PersonaOberservation,
)
from ..memory import Scratch
from .component import Component


class PerceiveComponent(Component):
    def __init__(
        self,
        model: ModelWandbWrapper,
    ):
        super().__init__(model)

    def init_persona_ref(self, persona):
        self.persona = persona

    def perceive(self, obs: PersonaOberservation):
        self._add_events(obs.events)

    def _add_events(self, events: list[PersonaEvent]):
        for event in events:
            self.persona.store.store_event(event)

    def _add_chats(self, chat: ChatObservation):
        self.persona.store.store_chat(
            chat.summary, chat.conversation, self.persona.current_time
        )
