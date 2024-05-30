from omegaconf import DictConfig

from ...common import ConcurrentEnv, PerturbationEnv


def tons_in_lake(num):
    return f"Before everyone fishes, there are {num} tons of fish in the lake."


def tons_caught(agent_name, wanted, caught):
    return (
        f"{agent_name} wanted to catch {wanted} tons of fish, and caught {caught} tons."
    )


def univ(sustainability_threshold):
    return f"Given the current situation, if everyone fishes more than {sustainability_threshold} tons, the lake population will shrink next month."


def tons_caught_home(agent_name, caught):
    return f"This month, {agent_name} caught {caught} tonnes of fish."


class FishingConcurrentEnv(ConcurrentEnv):
    def __init__(
        self, cfg: DictConfig, experiment_storage: str, map_id_to_name: dict[str, str]
    ) -> None:
        super().__init__(cfg, experiment_storage, map_id_to_name)
        self.POOL_LOCATION = "lake"

    def _prompt_pool_amount_of_resource(self):
        if self.cfg.harvesting_order == "concurrent":
            num = self.internal_global_state["resource_in_pool"]
        else:
            raise ValueError(f"Unknown fishing order: {self.cgf.harvesting_order}")
        return tons_in_lake(num)

    def _prompt_pool_amount_of_resource_after_harvesting(self, agent):
        wanted = self.internal_global_state["wanted_resource"][agent]
        caught = self.internal_global_state["last_collected_resource"][agent]
        agent_name = self.agent_id_to_name[agent]
        return tons_caught(agent_name, wanted, caught)

    def _prompt_universalization(self, sustainability_threshold):
        return univ(sustainability_threshold)


class FishingPerturbationEnv(PerturbationEnv):
    def __init__(
        self, cfg: DictConfig, experiment_storage: str, map_id_to_name: dict[str, str]
    ) -> None:
        super().__init__(cfg, experiment_storage, map_id_to_name)
        self.POOL_LOCATION = "lake"

    def _prompt_pool_amount_of_resource(self):
        if self.cfg.harvesting_order == "concurrent":
            num = self.internal_global_state["resource_in_pool"]
        else:
            raise ValueError(f"Unknown fishing order: {self.cgf.harvesting_order}")
        return tons_in_lake(num)

    def _prompt_pool_amount_of_resource_after_harvesting(self, agent):
        wanted = self.internal_global_state["wanted_resource"][agent]
        caught = self.internal_global_state["last_collected_resource"][agent]
        agent_name = self.agent_id_to_name[agent]
        return tons_caught(agent_name, wanted, caught)

    def _prompt_universalization(self, sustainability_threshold):
        return univ(sustainability_threshold)

    def _prompt_home_observe_agent_resource(self, agent):
        caught = self.internal_global_state["last_collected_resource"][agent]
        agent_name = self.agent_id_to_name[agent]
        return tons_caught_home(agent_name, caught)
