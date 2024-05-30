from simulation.persona.cognition import StoreComponent
from simulation.persona.common import ChatObservation, PersonaIdentity
from simulation.persona.embedding_model import EmbeddingModel
from simulation.persona.memory.associative_memory import AssociativeMemory
from simulation.utils import ModelWandbWrapper

from .store_prompts import (
    prompt_importance_action,
    prompt_importance_chat,
    prompt_importance_event,
    prompt_importance_thought,
    prompt_text_to_triple,
)


class PollutionStoreComponent(StoreComponent):

    def __init__(
        self,
        model: ModelWandbWrapper,
        associative_memory: AssociativeMemory,
        embedding_model: EmbeddingModel,
        cfg,
    ) -> None:
        super().__init__(model, associative_memory, embedding_model, cfg)
        self.prompt_importance_thought = prompt_importance_thought
        self.prompt_importance_chat = prompt_importance_chat
        self.prompt_importance_event = prompt_importance_event
        self.prompt_importance_action = prompt_importance_action
