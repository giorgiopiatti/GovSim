from datetime import datetime

import numpy as np

from simulation.utils import ModelWandbWrapper

from ..common import PersonaIdentity
from ..embedding_model import EmbeddingModel
from ..memory.associative_memory import AssociativeMemory, Node, NodeType, Thought
from .component import Component


class RetrieveComponent(Component):
    """
    Retrieve works as follows.
    First we need to have 3 scores for each item in the memory.
    - Recency: based on the time since the item was added to the memory (the more recent the more relevant) Weight: 0.5
    - Importance: based on the importance of the item (the more important the more relevant), via LLM Weight: 3
    - Relevance: based on the relevance of the item to the current context (the more relevant the more relevant), via cosine similarity of embedding Weight: 2

    Recency:
    assisgn a score base on 0.99**i where i is the index of the item in the memory (the more recent the more relevant)

    Importance:
    use LLM to generate a score for each item in the memory. It is computed when saved in memory


    Relevance:
    use cosine similarity of embedding to generate a score for each item in the memory
    TODO: choose embedding model
    """

    def __init__(
        self,
        model: ModelWandbWrapper,
        associative_memory: AssociativeMemory,
        embedding_model: EmbeddingModel,
    ):
        super().__init__(model)
        self.associative_memory = associative_memory

        self.embedding_model = embedding_model

        self.weights = {
            "recency": 0.5,
            "importance": 3,
            "relevance": 2,
        }
        self.recency_decay_param = 0.99

    def _recency_retrieval(self, nodes: list[Node]) -> dict[str, float]:
        """
        Calculate the recency retrieval scores for a list of nodes.

        Args:
            nodes (list[Node]): The list of nodes to calculate recency retrieval scores for.

        Returns:
            dict[str, float]: A dictionary mapping node IDs to their recency retrieval scores.
        """
        result = dict()
        for i, node in enumerate(nodes):
            result[node.id] = self.recency_decay_param**i
        return result

    def _importance_retrieval(self, nodes: list[Node]) -> dict[str, float]:
        """
        Retrieve the importance scores for a list of nodes and normalize them.

        Args:
            nodes (list[Node]): The list of nodes to retrieve importance scores for.

        Returns:
            dict[str, float]: A dictionary mapping node IDs to their normalized importance scores.
        """
        result = dict()
        for node in nodes:
            result[node.id] = node.importance_score

        # normalize
        min_score = 1
        max_score = 10
        for node_id in result.keys():
            result[node_id] = (result[node_id] - min_score) / (max_score - min_score)

        return result

    def _relevance_retrieval(
        self, nodes: list[Node], focal_point: str
    ) -> dict[str, float]:
        """
        Retrieves the relevance scores of nodes based on their similarity to the focal point.

        Args:
            nodes (list[Node]): The list of nodes to retrieve relevance scores for.
            focal_point (str): The focal point used for comparison.

        Returns:
            dict[str, float]: A dictionary mapping node IDs to their relevance scores.
        """

        result = dict()
        focal_point_embedding = self.embedding_model.embed_retrieve(focal_point)

        def cosine_similarity(a, b):
            return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

        for node in nodes:
            node_embedding = self.associative_memory.get_node_embedding(node.id)
            result[node.id] = cosine_similarity(focal_point_embedding, node_embedding)

        return result

    def _retrieve_dict(
        self, focal_points: list[str], top_k: int
    ) -> dict[str, list[Node]]:
        """
        Retrieve nodes from the associative memory based on given focal points.

        Args:
            focal_points (list[str]): List of focal points to retrieve nodes for.
            top_k (int): Number of top nodes to retrieve for each focal point.

        Returns:
            dict[str, list[Node]]: Dictionary mapping each focal point to a list of top-k nodes.

        """
        nodes = self.associative_memory.get_nodes_for_retrieval(
            self.persona.current_time
        )

        recency_scores = self._recency_retrieval(nodes)
        importance_scores = self._importance_retrieval(nodes)

        acc_nodes = dict()

        for focal_point in focal_points:
            relevance_scores = self._relevance_retrieval(nodes, focal_point)

            # combine scores
            combined_scores = dict()

            for node_id in recency_scores.keys():
                combined_scores[node_id] = (
                    recency_scores[node_id] * self.weights["recency"]
                    + importance_scores[node_id] * self.weights["importance"]
                    + relevance_scores[node_id] * self.weights["relevance"]
                )

            # Put max score to node with always_include flag
            max_value = max(combined_scores.values()) if combined_scores else 10
            for node in nodes:
                if node.always_include:
                    combined_scores[node.id] = max_value + 1

            # sort by combined scores
            sorted_nodes = sorted(
                nodes, key=lambda node: combined_scores[node.id], reverse=True
            )

            # pick top k
            top_k_nodes = sorted_nodes[:top_k]
            acc_nodes[focal_point] = top_k_nodes
        return acc_nodes

    def retrieve(
        self, focal_points: list[str], top_k: int
    ) -> list[tuple[datetime, str]]:
        res = self._retrieve_dict(focal_points, top_k)
        res = res.values()
        res = [node for nodes in res for node in nodes]

        # make sure we don't return the same node twice
        res = set(res)
        res = list(res)
        res_sort = [(node.created, node.description) for node in res]
        # sort by time, most recent last
        res_sort = sorted(res_sort, key=lambda x: x[0])
        return res_sort
