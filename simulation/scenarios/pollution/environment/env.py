from omegaconf import DictConfig

from ...common import ConcurrentEnv, PerturbationEnv


def unpolluted_water_in_pool(num):
    return f"Before the factory owners start production for the month, the river is {num}% unpolluted."


def water_used(agent_name, wanted, caught):
    return f"{agent_name} produced {caught} widgets, thus consuming {caught}% of unpolluted water in the river."


def univ(sustainability_threshold):
    return (
        f"Given the current situation, if each factory owner produces more than {sustainability_threshold} widgets,"
        f" consuming {sustainability_threshold}% of unpolluted water, the unpolluted water in the river will decrease next month."
    )


def water_used_home(agent_name, caught):
    return f"This month, {agent_name} produced {caught} widgets, thus consuming {caught}% of unpolluted water in the river."


class PollutionConcurrentEnv(ConcurrentEnv):
    def __init__(
        self, cfg: DictConfig, experiment_storage: str, map_id_to_name: dict[str, str]
    ) -> None:
        super().__init__(cfg, experiment_storage, map_id_to_name)
        self.POOL_LOCATION = "factory"

    def _prompt_pool_amount_of_resource(self):
        if self.cfg.harvesting_order == "concurrent":
            num = self.internal_global_state["resource_in_pool"]
        else:
            raise ValueError(f"Unknown harvesting order: {self.cgf.harvesting_order}")
        return unpolluted_water_in_pool(num)

    def _prompt_pool_amount_of_resource_after_harvesting(self, agent):
        wanted = self.internal_global_state["wanted_resource"][agent]
        caught = self.internal_global_state["last_collected_resource"][agent]
        agent_name = self.agent_id_to_name[agent]
        return water_used(agent_name, wanted, caught)

    def _prompt_universalization(self, sustainability_threshold):
        return univ(sustainability_threshold)


class PollutionPerturbationEnv(PerturbationEnv):
    def __init__(
        self, cfg: DictConfig, experiment_storage: str, map_id_to_name: dict[str, str]
    ) -> None:
        super().__init__(cfg, experiment_storage, map_id_to_name)
        self.POOL_LOCATION = "factory"

    def _prompt_pool_amount_of_resource(self):
        if self.cfg.harvesting_order == "concurrent":
            num = self.internal_global_state["resource_in_pool"]
        else:
            raise ValueError(f"Unknown harvesting order: {self.cgf.harvesting_order}")
        return unpolluted_water_in_pool(num)

    def _prompt_pool_amount_of_resource_after_harvesting(self, agent):
        wanted = self.internal_global_state["wanted_resource"][agent]
        caught = self.internal_global_state["last_collected_resource"][agent]
        agent_name = self.agent_id_to_name[agent]
        return water_used(agent_name, wanted, caught)

    def _prompt_universalization(self, sustainability_threshold):
        return univ(sustainability_threshold)

    def _prompt_home_observe_agent_resource(self, agent):
        caught = self.internal_global_state["last_collected_resource"][agent]
        agent_name = self.agent_id_to_name[agent]
        return water_used_home(agent_name, caught)
