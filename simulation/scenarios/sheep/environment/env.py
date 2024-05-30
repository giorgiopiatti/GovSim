from omegaconf import DictConfig

from ...common import ConcurrentEnv, PerturbationEnv


def hectares_in_pool(num):
    return f"Before the shepherds take their flocks of sheep to the pasture, there are {num} hectares of grass available."


def tons_caught(agent_name, wanted, caught):
    return f"{agent_name} took {caught} flocks of sheep to the pasture, thus consuming {caught} hectares of grass."


def univ(sustainability_threshold):
    return (
        f"Given the current situation, if each shepherd take more than {sustainability_threshold} flocks of sheep to the pasture,"
        f" consuming {sustainability_threshold} hectares of grass, the available grass in the pasture will decrease next month"
    )


def tons_caught_home(agent_name, caught):
    return f"This month, {agent_name} took {caught} flocks of sheep to the pasture, thus consuming {caught} hectares of grass."


class SheepConcurrentEnv(ConcurrentEnv):
    def __init__(
        self, cfg: DictConfig, experiment_storage: str, map_id_to_name: dict[str, str]
    ) -> None:
        super().__init__(cfg, experiment_storage, map_id_to_name)
        self.POOL_LOCATION = "pasture"

    def _prompt_pool_amount_of_resource(self):
        if self.cfg.harvesting_order == "concurrent":
            num = self.internal_global_state["resource_in_pool"]
        else:
            raise ValueError(f"Unknown harvesting order: {self.cgf.harvesting_order}")
        return hectares_in_pool(num)

    def _prompt_pool_amount_of_resource_after_harvesting(self, agent):
        wanted = self.internal_global_state["wanted_resource"][agent]
        caught = self.internal_global_state["last_collected_resource"][agent]
        agent_name = self.agent_id_to_name[agent]
        return tons_caught(agent_name, wanted, caught)

    def _prompt_universalization(self, sustainability_threshold):
        return univ(sustainability_threshold)


class SheepPerturbationEnv(PerturbationEnv):
    def __init__(
        self, cfg: DictConfig, experiment_storage: str, map_id_to_name: dict[str, str]
    ) -> None:
        super().__init__(cfg, experiment_storage, map_id_to_name)
        self.POOL_LOCATION = "pasture"

    def _prompt_pool_amount_of_resource(self):
        if self.cfg.harvesting_order == "concurrent":
            num = self.internal_global_state["resource_in_pool"]
        else:
            raise ValueError(f"Unknown harvesting order: {self.cgf.harvesting_order}")
        return hectares_in_pool(num)

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
