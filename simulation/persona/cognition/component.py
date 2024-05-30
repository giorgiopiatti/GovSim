from typing import TYPE_CHECKING

from simulation.utils import ModelWandbWrapper

if TYPE_CHECKING:
    from ..persona import PersonaAgent


class Component:
    persona: "PersonaAgent"

    def __init__(self, model: ModelWandbWrapper, cfg=None) -> None:
        self.model = model
        self.cfg = cfg
        self.other_personas: dict[str, "PersonaAgent"] = {}

    def init_persona_ref(self, persona: "PersonaAgent"):
        self.persona = persona

    def add_reference_to_other_persona(self, persona: "PersonaAgent"):
        self.other_personas[persona.identity.name] = persona
