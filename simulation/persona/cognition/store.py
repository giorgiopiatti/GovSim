from datetime import datetime, timedelta

import numpy as np

from simulation.utils import ModelWandbWrapper

from ..common import PersonaEvent, PersonaIdentity
from ..embedding_model import EmbeddingModel
from ..memory.associative_memory import AssociativeMemory, Node, NodeType
from .component import Component


class StoreComponent(Component):
    prompt_importance_thought: callable
    prompt_importance_chat: callable
    prompt_importance_event: callable
    prompt_importance_action: callable

    def __init__(
        self,
        model: ModelWandbWrapper,
        associative_memory: AssociativeMemory,
        embedding_model: EmbeddingModel,
        cfg,
    ) -> None:
        super().__init__(model, cfg)
        self.associative_memory = associative_memory
        self.embedding_model = embedding_model

    def _compute_importance(self, node: Node) -> float:
        if node.type == NodeType.THOUGHT:
            score = self.prompt_importance_thought(
                self.model, self.persona.identity, node
            )
        elif node.type == NodeType.CHAT:
            score = self.prompt_importance_chat(self.model, self.persona.identity, node)
        elif node.type == NodeType.EVENT:
            score = self.prompt_importance_event(
                self.model, self.persona.identity, node
            )
        elif node.type == NodeType.ACTION:
            score = self.prompt_importance_action(
                self.model, self.persona.identity, node
            )
        else:
            raise ValueError(f"Unknown node type: {node.type}")
        node.importance_score = score

    def store_event(self, event: PersonaEvent):
        # s, p, o = prompt_text_to_triple(self.model, event.description)
        s, p, o = (None, None, None)
        node = self.associative_memory.add_event(
            s, p, o, event.description, event.created, event.expiration
        )
        if event.always_include:
            node.importance_score = 10
            node.always_include = True
        else:
            self._compute_importance(node)
        embedding = self.embedding_model.embed(event.description)
        self.associative_memory.set_node_embedding(node.id, embedding)

    def store_chat(
        self,
        summary: str,
        conversation: list[tuple[str, str]],
        created: datetime,
        expiration_delta: timedelta = None,
    ):
        if expiration_delta is None:
            expiration_delta = timedelta(days=self.cfg.expiration_delta.days)
        expiration = created + expiration_delta
        # s, p, o = prompt_text_to_triple(self.model, summary)
        s, p, o = (None, None, None)
        node = self.associative_memory.add_chat(
            s, p, o, summary, conversation, created, expiration
        )
        self._compute_importance(node)
        embedding = self.embedding_model.embed(summary)
        self.associative_memory.set_node_embedding(node.id, embedding)

    def store_action(
        self,
        description: str,
        created: datetime,
        expiration_delta: timedelta = None,
    ):
        if expiration_delta is None:
            expiration_delta = timedelta(days=self.cfg.expiration_delta.days)
        expiration = created + expiration_delta
        # s, p, o = prompt_text_to_triple(self.model, description)
        s, p, o = (None, None, None)
        node = self.associative_memory.add_action(
            s, p, o, description, created, expiration
        )
        self._compute_importance(node)
        embedding = self.embedding_model.embed(description)
        self.associative_memory.set_node_embedding(node.id, embedding)

    def store_thought(
        self,
        description: str,
        created: datetime,
        expiration_delta: timedelta = None,
        always_include: bool = False,
    ):
        if expiration_delta is None:
            expiration_delta = timedelta(days=self.cfg.expiration_delta.days)
        expiration = created + expiration_delta
        # s, p, o = prompt_text_to_triple(self.model, description)
        s, p, o = (None, None, None)
        node = self.associative_memory.add_thought(
            s, p, o, description, created, expiration
        )
        if always_include:
            node.importance_score = 10
            node.always_include = True
        else:
            self._compute_importance(node)
        embedding = self.embedding_model.embed(description)
        self.associative_memory.set_node_embedding(node.id, embedding)
