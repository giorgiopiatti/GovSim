import math
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
from omegaconf import DictConfig, OmegaConf
from pettingzoo.utils import agent_selector

from simulation.persona.common import (
    PersonaAction,
    PersonaActionChat,
    PersonaActionHarvesting,
    PersonaEvent,
    PersonaIdentity,
)

from .common import HarvestingObs
from .concurrent_env import (
    ConcurrentEnv,
    get_discussion_day,
    get_expiration_next_month,
    get_reflection_day,
)

"""
Uses the env.perturbations settins

Perturbation:
name: NAME
round: ROUND_AT_WHICH_TO_APPLY


"""


class PerturbationEnv(ConcurrentEnv):
    def __init__(
        self, cfg: DictConfig, experiment_storage: str, map_id_to_name: dict[str, str]
    ) -> None:
        super().__init__(cfg, experiment_storage, map_id_to_name)

        assert len(cfg.perturbations) == 1
        self.perturbation = cfg.perturbations[0].perturbation

    def _prompt_home_observe_agent_resource(self, agent):
        raise NotImplementedError

    def _observe_home(self, agent) -> HarvestingObs:
        if (
            self.cfg.language_nature == "none"
            or self.cfg.language_nature == "none_and_no_obs"
        ):
            # Inject what each person has fished
            events = []
            if self.cfg.language_nature == "none":
                for agent in self.agents:
                    events.append(
                        PersonaEvent(
                            self._prompt_home_observe_agent_resource(agent),
                            created=self.internal_global_state["next_time"][agent],
                            expiration=get_expiration_next_month(
                                self.internal_global_state["next_time"][agent]
                            ),
                        )
                    )

            state = HarvestingObs(
                phase=self.phase,
                current_location=self.internal_global_state["next_location"][agent],
                current_location_agents=self.internal_global_state["next_location"],
                current_time=self.internal_global_state["next_time"][agent],
                events=events,
                context="",
                chat=None,
                current_resource_num=self.internal_global_state["resource_in_pool"],
                agent_resource_num={agent: 0 for agent in self.agents},
                before_harvesting_sustainability_threshold=self.internal_global_state[
                    "sustainability_threshold"
                ],
            )
        else:
            state = super()._observe_home(agent)
        return state

    def _step_pool_after_harvesting(self, action: PersonaActionHarvesting):
        # We have no interaction with other agents at the lake
        self.internal_global_state["next_location"][self.agent_selection] = "restaurant"
        self.internal_global_state["next_time"][self.agent_selection] = (
            get_discussion_day(
                self.internal_global_state["next_time"][self.agent_selection]
            )
        )

        # Apply perturbations
        if (
            self.cfg.language_nature == "none"
            or self.cfg.language_nature == "none_and_no_obs"
        ):
            self.internal_global_state["next_location"][self.agent_selection] = "home"
            self.internal_global_state["next_time"][self.agent_selection] = (
                get_reflection_day(
                    self.internal_global_state["next_time"][self.agent_selection]
                )
            )

        # Next phase / next agent
        if self._agent_selector.is_last():
            self.phase = self._phase_selector.next()
        self.agent_selection = self._agent_selector.next()

    def step(self, action):
        if self.terminations[self.agent_selection]:
            return

        assert action.agent_id == self.agent_selection

        if self.phase == self.POOL_LOCATION:
            assert action.location == self.POOL_LOCATION
            assert type(action) == PersonaActionHarvesting
            self._step_lake_bet(action)
        elif self.phase == "pool_after_harvesting":
            assert action.location == self.POOL_LOCATION
            self._step_pool_after_harvesting(action)
        elif self.phase == "restaurant":
            assert action.location == "restaurant"
            self._step_restaurant(action)
        elif self.phase == "home":
            assert action.location == "home"
            self._step_home(action)
            if self._agent_selector.is_last():
                self.save_log()
                self.num_round += 1

                ## Apply perturbations, now we assume we have only 1

                if self.num_round == self.perturbation.round:
                    if self.perturbation.type == "change_language_nature" and (
                        self.perturbation.language_nature == "none"
                        or self.perturbation.language_nature == "none_and_no_obs"
                    ):
                        self.cfg.language_nature = "none"
                        self._phase_selector = agent_selector(
                            [self.POOL_LOCATION, "pool_after_harvesting", "home"]
                        )
                    elif self.perturbation.type == "insert_outsider":
                        self.agents = self.possible_agents
                        self._agent_selector = agent_selector(self.agents)
                        self._init_agent(self.agents[-1])

                self.phase = self._phase_selector.next()

                # We want to see also the discussion in case no fish remain
                self.terminations = {
                    agent: (
                        self.internal_global_state["resource_in_pool"] < 5
                        or self.num_round >= self.cfg.max_num_rounds
                    )
                    for agent in self.agents
                }

                self.internal_global_state["resource_in_pool"] = min(
                    self.cfg.initial_resource_in_pool,
                    self.internal_global_state["resource_in_pool"] * 2,
                )  # Double the fish in the lake, but cap at 100
                self.internal_global_state["resource_before_harvesting"] = (
                    self.internal_global_state["resource_in_pool"]
                )
                self.internal_global_state["sustainability_threshold"] = int(
                    (self.internal_global_state["resource_in_pool"] // 2)
                    // self.internal_global_state["num_agents"]
                )
                if self.cfg.harvesting_order == "random-sequential":
                    agents = list(np.random.permutation(self.agents))
                    self._agent_selector = agent_selector(agents)
            self.agent_selection = self._agent_selector.next()

        return (
            self.agent_selection,
            self._observe(self.agent_selection),
            self.rewards,
            self.terminations,
        )

    def reset(self):
        super().reset()

        if self.num_round == self.perturbation.round:
            if self.perturbation.type == "change_language_nature" and (
                self.perturbation.language_nature == "none"
                or self.perturbation.language_nature == "none_and_no_obs"
            ):
                self.cfg.language_nature = self.perturbation.language_nature
                self._phase_selector = agent_selector(
                    [self.POOL_LOCATION, "pool_after_harvesting", "home"]
                )
                self.phase = self._phase_selector.next()
            elif self.perturbation.type == "insert_outsider":
                self.agents = self.possible_agents
                self._agent_selector = agent_selector(self.agents)
                self._init_agent(self.agents[-1])
                self.agent_selection = self._agent_selector.next()

        return self.agent_selection, self._observe(self.agent_selection)
