import os
from datetime import datetime
from typing import Any

from simulation.persona.common import (
    ChatObservation,
    PersonaAction,
    PersonaIdentity,
    PersonaOberservation,
)
from simulation.utils import ModelWandbWrapper

from .cognition import (
    ActComponent,
    ConverseComponent,
    PerceiveComponent,
    PlanComponent,
    ReflectComponent,
    RetrieveComponent,
    StoreComponent,
)
from .embedding_model import EmbeddingModel
from .memory import AssociativeMemory, Scratch


class PersonaAgent:
    agent_id: int
    identity: PersonaIdentity

    current_time: datetime

    scratch: Scratch

    def __init__(
        self,
        cfg,
        model: ModelWandbWrapper,
        embedding_model: EmbeddingModel,
        base_path: str,
        memory_cls: type[AssociativeMemory] = AssociativeMemory,
        perceive_cls: type[PerceiveComponent] = PerceiveComponent,
        retrieve_cls: type[RetrieveComponent] = RetrieveComponent,
        store_cls: type[StoreComponent] = StoreComponent,
        reflect_cls: type[ReflectComponent] = ReflectComponent,
        plan_cls: type[PlanComponent] = PlanComponent,
        act_cls: type[ActComponent] = ActComponent,
        converse_cls: type[ConverseComponent] = ConverseComponent,
    ) -> None:
        self.cfg = cfg
        self.base_path = base_path
        os.makedirs(base_path, exist_ok=True)

        self.memory = memory_cls(base_path)
        self.perceive = perceive_cls(model)
        self.retrieve = retrieve_cls(model, self.memory, embedding_model)
        self.store = store_cls(model, self.memory, embedding_model, self.cfg.store)
        self.reflect = reflect_cls(model)
        self.plan = plan_cls(model)
        self.act = act_cls(
            model,
            self.cfg.act,
        )
        self.converse = converse_cls(model, self.retrieve, self.cfg.converse)

        self.perceive.init_persona_ref(self)
        self.retrieve.init_persona_ref(self)
        self.store.init_persona_ref(self)
        self.reflect.init_persona_ref(self)
        self.plan.init_persona_ref(self)
        self.act.init_persona_ref(self)
        self.converse.init_persona_ref(self)

        self.other_personas: dict[str, PersonaAgent] = {}
        self.other_personas_from_id: dict[str, PersonaAgent] = {}

    def init_persona(self, agent_id: int, identity: PersonaIdentity, social_graph):
        self.agent_id = agent_id
        self.identity = identity

        self.scratch = Scratch(f"{self.base_path}")

    def add_reference_to_other_persona(self, persona: "PersonaAgent"):
        self.other_personas[persona.identity.name] = persona
        self.other_personas_from_id[persona.agent_id] = persona
        self.perceive.add_reference_to_other_persona(persona)
        self.retrieve.add_reference_to_other_persona(persona)
        self.store.add_reference_to_other_persona(persona)
        self.reflect.add_reference_to_other_persona(persona)
        self.plan.add_reference_to_other_persona(persona)
        self.act.add_reference_to_other_persona(persona)
        self.converse.add_reference_to_other_persona(persona)

    def loop(self, obs: PersonaOberservation) -> PersonaAction:
        raise NotImplementedError("needs to be implemented in subclass")
