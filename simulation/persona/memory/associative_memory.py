import json
import os
import typing
from datetime import datetime
from enum import Enum

import numpy as np


class NodeType(Enum):
    CHAT = 1
    THOUGHT = 2
    EVENT = 3
    ACTION = 4

    def toJSON(self):
        return self.name


class Node:
    id: int
    type: NodeType

    subject: str
    predicate: str
    object: str

    description: str

    importance_score: float

    created: datetime
    expiration: datetime

    always_include: bool

    def __init__(
        self,
        id: int,
        type: NodeType,
        subject: str,
        predicate: str,
        object: str,
        description: str,
        created: datetime,
        expiration: datetime,
        always_include: bool = False,
    ) -> None:
        self.id = id
        self.type = type
        self.subject = subject
        self.predicate = predicate
        self.object = object
        self.description = description
        self.created = created
        self.expiration = expiration
        self.always_include = always_include

    def __str__(self) -> str:
        return f"{self.subject} {self.predicate} {self.object}"

    def toJSON(self):
        return {
            "id": self.id,
            "type": self.type.toJSON(),
            "subject": self.subject,
            "predicate": self.predicate,
            "object": self.object,
            "description": self.description,
            "importance_score": self.importance_score,
            "created": self.created.strftime("%Y-%m-%d %H:%M:%S"),
            "expiration": self.expiration.strftime("%Y-%m-%d %H:%M:%S"),
            "always_include": "true" if self.always_include else "false",
        }


class Thought(Node):
    def __init__(
        self,
        id: int,
        subject: str,
        predicate: str,
        object: str,
        description: str,
        created: datetime,
        expiration: datetime,
        always_include: bool = False,
    ) -> None:
        super().__init__(
            id,
            NodeType.THOUGHT,
            subject,
            predicate,
            object,
            description,
            created,
            expiration,
            always_include,
        )


class Chat(Node):
    conversation: list[tuple[str, str]]

    def __init__(
        self,
        id: int,
        subject: str,
        predicate: str,
        object: str,
        description: str,
        created: datetime,
        expiration: datetime,
        always_include: bool = False,
    ) -> None:
        self.conversation = []
        super().__init__(
            id,
            NodeType.CHAT,
            subject,
            predicate,
            object,
            description,
            created,
            expiration,
            always_include,
        )

    def toJSON(self):
        return {
            "id": self.id,
            "type": self.type.toJSON(),
            "subject": self.subject,
            "predicate": self.predicate,
            "object": self.object,
            "description": self.description,
            "importance_score": self.importance_score,
            "conversation": self.conversation,
            "created": self.created.strftime("%Y-%m-%d %H:%M:%S"),
            "expiration": self.expiration.strftime("%Y-%m-%d %H:%M:%S"),
            "always_include": "true" if self.always_include else "false",
        }


class Event(Node):
    def __init__(
        self,
        id: int,
        subject: str,
        predicate: str,
        object: str,
        description: str,
        created: datetime,
        expiration: datetime,
        always_include: bool = False,
    ) -> None:
        super().__init__(
            id,
            NodeType.EVENT,
            subject,
            predicate,
            object,
            description,
            created,
            expiration,
            always_include,
        )


class Action(Node):
    def __init__(
        self,
        id: int,
        subject: str,
        predicate: str,
        object: str,
        description: str,
        created: datetime,
        expiration: datetime,
        always_include: bool = False,
    ) -> None:
        super().__init__(
            id,
            NodeType.ACTION,
            subject,
            predicate,
            object,
            description,
            created,
            expiration,
            always_include,
        )


import os


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


class AssociativeMemory:
    def __init__(self, base_path, do_load=False) -> None:
        self.id_to_node: typing.Dict[int, Node] = dict()

        self.thought_id_to_node: typing.Dict[int, Thought] = dict()
        self.chat_id_to_node: typing.Dict[int, Node] = dict()
        self.event_id_to_node: typing.Dict[int, Node] = dict()
        self.action_id_to_node: typing.Dict[int, Node] = dict()

        self.nodes_without_chat_by_time: list[Node] = []

        self.embeddings: dict[int, list[float]] = dict()

        self.base_path = base_path
        if (
            os.path.exists(f"{base_path}/embeddings.json")
            and os.path.exists(f"{base_path}/nodes.json")
            and do_load
        ):
            self._load(base_path)

    def _load(self, base_path):
        raise NotImplementedError("Not sure if it works")
        self.embeddings = json.load(open(f"{base_path}/embeddings.json"))
        saved_nodes = json.load(open(f"{base_path}/nodes.json"))
        for node in saved_nodes:
            self.id_to_node[node["id"]] = Node(
                node["id"],
                NodeType[node["type"]],
                node["subject"],
                node["predicate"],
                node["object"],
                node["description"],
                node["embedding_key"],
                node["created"],
                node["expiration"],
            )

    def save(self):
        json.dump(
            [node.toJSON() for node in self.id_to_node.values()],
            open(f"{self.base_path}/nodes.json", "w"),
        )
        json.dump(
            self.embeddings,
            open(f"{self.base_path}/embeddings.json", "w"),
            cls=NumpyEncoder,
        )

    def _add(
        self, subject, predicate, obj, description, type, created, expiration
    ) -> Node:
        id = len(self.id_to_node) + 1

        if type == NodeType.CHAT:
            node = Chat(id, subject, predicate, obj, description, created, expiration)
            self.chat_id_to_node[id] = node
        elif type == NodeType.THOUGHT:
            node = Thought(
                id, subject, predicate, obj, description, created, expiration
            )
            self.thought_id_to_node[id] = node
        elif type == NodeType.EVENT:
            node = Event(id, subject, predicate, obj, description, created, expiration)
            self.event_id_to_node[id] = node
        elif type == NodeType.ACTION:
            node = Action(id, subject, predicate, obj, description, created, expiration)
            self.action_id_to_node[id] = node

        if type != NodeType.CHAT:
            self.nodes_without_chat_by_time.append(node)

        self.id_to_node[id] = node

        return node

    def add_chat(
        self, subject, predicate, obj, description, conversation, created, expiration
    ) -> Chat:
        node = self._add(
            subject, predicate, obj, description, NodeType.CHAT, created, expiration
        )
        node.conversation = conversation
        return node

    def add_thought(
        self, subject, predicate, obj, description, created, expiration
    ) -> Thought:
        return self._add(
            subject, predicate, obj, description, NodeType.THOUGHT, created, expiration
        )

    def add_event(
        self, subject, predicate, obj, description, created, expiration
    ) -> Event:
        return self._add(
            subject, predicate, obj, description, NodeType.EVENT, created, expiration
        )

    def add_action(
        self, subject, predicate, obj, description, created, expiration
    ) -> Action:
        return self._add(
            subject, predicate, obj, description, NodeType.ACTION, created, expiration
        )

    def get_nodes_for_retrieval(self, current_time: datetime) -> list[Node]:
        """
        Get all nodes except chat
        """
        nodes = []
        for node in self.nodes_without_chat_by_time:
            if node.expiration > current_time:
                nodes.append(node)
        return nodes

    def get_node_embedding(self, node_id: int):
        return self.embeddings[node_id]

    def set_node_embedding(self, node_id: int, embedding: list[float]):
        self.embeddings[node_id] = embedding
