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


def get_reflection_day(current_date):
    next_month = current_date.replace(day=28) + timedelta(days=4)
    last_day_of_current_month = next_month - timedelta(days=next_month.day)
    return last_day_of_current_month


def get_discussion_day(current_date):
    reflection = get_reflection_day(current_date)
    return reflection - timedelta(days=1)


def get_expiration_next_month(current_date):
    return get_reflection_day(current_date)


class ConcurrentEnv:
    def __init__(
        self, cfg: DictConfig, experiment_storage: str, map_id_to_name: dict[str, str]
    ) -> None:
        self.cfg = cfg
        self.experiment_storage = experiment_storage

        self.possible_agents = [f"persona_{i}" for i in range(5)]
        self.agent_name_mapping = dict(
            zip(self.possible_agents, list(range(len(self.possible_agents))))
        )
        self.agent_id_to_name = map_id_to_name

        self.POOL_LOCATION = "pool"

    ### Prompt text

    def _prompt_pool_amount_of_resource(self):
        raise NotImplementedError

    def _prompt_pool_amount_of_resource_after_harvesting(self, agent):
        raise NotImplementedError

    def _prompt_universalization(self, sustainability_threshold):
        raise NotImplementedError

    def _observe_pool(self, agent) -> HarvestingObs:
        sustainability_threshold = self.internal_global_state[
            "sustainability_threshold"
        ]
        events = [
            PersonaEvent(
                self._prompt_pool_amount_of_resource(),
                created=self.internal_global_state["next_time"][agent],
                expiration=get_expiration_next_month(
                    self.internal_global_state["next_time"][agent]
                ),
                always_include=True,
            )
        ]
        if self.cfg.inject_universalization:
            events.append(
                PersonaEvent(
                    self._prompt_universalization(sustainability_threshold),
                    created=self.internal_global_state["next_time"][agent],
                    expiration=get_expiration_next_month(
                        self.internal_global_state["next_time"][agent]
                    ),
                    always_include=True,
                )
            )
        obs = HarvestingObs(
            phase=self.phase,
            current_location=self.internal_global_state["next_location"][agent],
            current_location_agents=self.internal_global_state["next_location"],
            current_time=self.internal_global_state["next_time"][agent],
            events=events,
            context="",
            chat=None,
            current_resource_num=self.internal_global_state["resource_in_pool"],
            agent_resource_num={agent: 0 for agent in self.agents},
            before_harvesting_sustainability_threshold=sustainability_threshold,
        )
        return obs

    def _observe_pool_after_harvesting(self, agent) -> HarvestingObs:
        obs = HarvestingObs(
            phase=self.phase,
            current_location=self.internal_global_state["next_location"][agent],
            current_location_agents=self.internal_global_state["next_location"],
            current_time=self.internal_global_state["next_time"][agent],
            events=[
                PersonaEvent(
                    self._prompt_pool_amount_of_resource_after_harvesting(agent),
                    created=get_discussion_day(
                        self.internal_global_state["next_time"][agent]
                    )
                    - timedelta(minutes=1),  # hack to sort the events
                    expiration=get_expiration_next_month(
                        self.internal_global_state["next_time"][agent]
                    ),
                    always_include=True,
                )
            ],
            context="",
            chat=None,
            current_resource_num=self.internal_global_state["resource_in_pool"],
            agent_resource_num={agent: 0 for agent in self.agents},
            before_harvesting_sustainability_threshold=self.internal_global_state[
                "sustainability_threshold"
            ],
        )
        return obs

    def _observe_restaurant(self, agent) -> HarvestingObs:
        events = []
        state = HarvestingObs(
            phase=self.phase,
            current_location=self.internal_global_state["next_location"][agent],
            current_location_agents=self.internal_global_state["next_location"],
            current_time=self.internal_global_state["next_time"][agent],
            events=events,
            context="",
            chat=None,
            current_resource_num=self.internal_global_state["resource_in_pool"],
            agent_resource_num=self.internal_global_state["last_collected_resource"],
            before_harvesting_sustainability_threshold=self.internal_global_state[
                "sustainability_threshold"
            ],
        )
        return state

    def _observe_home(self, agent) -> HarvestingObs:
        state = HarvestingObs(
            phase=self.phase,
            current_location=self.internal_global_state["next_location"][agent],
            current_location_agents=self.internal_global_state["next_location"],
            current_time=self.internal_global_state["next_time"][agent],
            events=[],
            context="",
            chat=None,
            current_resource_num=self.internal_global_state["resource_in_pool"],
            agent_resource_num={agent: 0 for agent in self.agents},
            before_harvesting_sustainability_threshold=self.internal_global_state[
                "sustainability_threshold"
            ],
        )
        return state

    def _observe(self, agent) -> HarvestingObs:
        """
        Observe should return the observation of the specified agent.

        Depending on the current phase, the observation will be different
        """

        if self.phase == self.POOL_LOCATION:
            state = self._observe_pool(agent)
        elif self.phase == "pool_after_harvesting":
            state = self._observe_pool_after_harvesting(agent)
        elif self.phase == "restaurant":
            state = self._observe_restaurant(agent)
        elif self.phase == "home":
            state = self._observe_home(agent)
        return state

    def close(self):
        """
        Close should release any graphical displays, subprocesses, network connections
        or any other environment data which should not be kept around after the
        user is no longer using the environment.
        """
        pass

    def _init_agent(self, agent):
        self.internal_global_state["collected_resource"][agent] = 0
        self.internal_global_state["wanted_resource"][agent] = 0
        self.internal_global_state["last_collected_resource"][agent] = 0
        self.internal_global_state["next_location"][agent] = self.POOL_LOCATION
        self.internal_global_state["next_time"][agent] = datetime(2024, 1, 1, 1, 0, 0)

        self.rewards[agent] = 0.0
        self.terminations[agent] = False

    def reset(self, seed=None, options=None) -> tuple[str, HarvestingObs]:
        self.random = np.random.RandomState(seed)

        self.agents = self.possible_agents[: self.cfg.num_agents]

        self.num_round = 0
        self.df_acc = []

        # RL specific (for pettingzoo)
        self.rewards = {}
        self.terminations = {}

        # Environment specific
        self.internal_global_state = {
            "num_agents": float(self.cfg.num_agents),
            "resource_in_pool": self.cfg.initial_resource_in_pool,
            "resource_before_harvesting": self.cfg.initial_resource_in_pool,
            "sustainability_threshold": (
                10
            ),  # each day the fish double and cap at 100, so maximum 50 can be fished
            "collected_resource": {},
            "wanted_resource": {},
            "last_collected_resource": {},
            "next_location": {},
            "next_time": {},
            "action": {},
        }
        for agent in self.agents:
            self._init_agent(agent)

        self._agent_selector = agent_selector(self.agents)
        self.agent_selection = self._agent_selector.next()
        self._phase_selector = agent_selector(
            [self.POOL_LOCATION, "pool_after_harvesting", "restaurant", "home"]
        )
        self.phase = self._phase_selector.next()

        return self.agent_selection, self._observe(self.agent_selection)

    def save_log(self):
        pd.concat(self.df_acc).to_json(
            f"{self.experiment_storage}/log_env.json", orient="records"
        )

    def _assign_stochastic(self):
        resource_per_agent = {agent: 0 for agent in self.agents}

        wanted = self.internal_global_state["wanted_resource"].copy()
        remaining = self.internal_global_state["resource_in_pool"]
        while sum(wanted.values()) > 0 and remaining > 0:
            # filter agents which want more fish
            agents_to_assign = [agent for agent in self.agents if wanted[agent] > 0]
            if len(agents_to_assign) == 0:
                break
            # pick random agent
            agent = self.random.choice(agents_to_assign)
            wanted[agent] -= 1
            resource_per_agent[agent] += 1
            remaining -= 1

        self.internal_global_state["resource_in_pool"] = int(remaining)

        for agent in self.agents:
            # convert to int
            resource_per_agent[agent] = int(resource_per_agent[agent])

        return resource_per_agent

    def _assign_proportional(self):
        resource_per_agent = {agent: 0 for agent in self.agents}
        was_rounded_down = {agent: False for agent in self.agents}

        wanted = self.internal_global_state["wanted_resource"].copy()
        remaining = self.internal_global_state["resource_in_pool"]
        while sum(wanted.values()) > 0 and remaining > 0:
            total_wanted = sum(wanted.values())
            if total_wanted > remaining:
                remaining_res_now = remaining
                for agent in wanted.keys():
                    tmp = remaining_res_now * (wanted[agent] / total_wanted)
                    tmp = min(
                        tmp, wanted[agent]
                    )  # we cannot assign more than the agent wanted
                    if tmp == 0:
                        continue
                    if tmp == int(tmp):
                        resource_per_agent[agent] += int(tmp)
                        was_rounded_down[agent] = False
                    else:
                        tmp = np.floor(tmp)
                        was_rounded_down[agent] = True
                        resource_per_agent[agent] += tmp
                    remaining -= tmp
                    wanted[agent] -= tmp

                if remaining > 0:
                    # assign the remaining fish to the agents that were rounded down, randomly

                    if remaining > len([w for w in wanted.values() if w > 0]):
                        continue
                    else:
                        for _ in range(int(remaining)):
                            agents_to_assign = [
                                agent
                                for agent in self.agents
                                if was_rounded_down[agent] and wanted[agent] > 0
                            ]
                            if len(agents_to_assign) == 0:
                                break
                            total = sum(wanted[agent] for agent in agents_to_assign)
                            p = [wanted[agent] / total for agent in agents_to_assign]
                            agent = self.random.choice(agents_to_assign, p=p)
                            resource_per_agent[agent] += 1
                            wanted[agent] -= 1
                            agents_to_assign.remove(agent)
                            remaining -= 1

            else:
                for agent in self.agents:
                    resource_per_agent[agent] = wanted[agent]
                    remaining -= wanted[agent]
                    wanted[agent] = 0

        self.internal_global_state["resource_in_pool"] = int(remaining)

        for agent in self.agents:
            # convert to int
            resource_per_agent[agent] = int(resource_per_agent[agent])

        return resource_per_agent

    def _assign_resource(self):
        if self.cfg.assign_resource_strategy == "stochastic":
            resource_per_agent = self._assign_stochastic()
        elif self.cfg.assign_resource_strategy == "proportional":
            resource_per_agent = self._assign_proportional()
        else:
            raise ValueError(
                f"Unknown assign resource strategy: {self.cfg.assign_resource_strategy}"
            )

        for agent in self.agents:
            res = resource_per_agent[agent]
            action = self.internal_global_state["action"][agent]
            self.log_step_harvest(action, res)

        for agent in self.agents:
            res = resource_per_agent[agent]
            self.internal_global_state["collected_resource"][agent] += res
            self.internal_global_state["last_collected_resource"][agent] = res

            self.rewards[agent] += res

    def _step_lake_bet(self, action: PersonaActionHarvesting):
        res = action.quantity
        self.internal_global_state["wanted_resource"][self.agent_selection] = res
        self.internal_global_state["action"][self.agent_selection] = action
        self.internal_global_state["next_location"][
            self.agent_selection
        ] = self.POOL_LOCATION
        if self._agent_selector.is_last():
            self._assign_resource()
            self.phase = self._phase_selector.next()
        self.agent_selection = self._agent_selector.next()

    def _step_pool_after_harvesting(self, action: PersonaActionHarvesting):
        # We have no interaction with other agents at the lake
        self.internal_global_state["next_location"][self.agent_selection] = "restaurant"
        self.internal_global_state["next_time"][self.agent_selection] = (
            get_discussion_day(
                self.internal_global_state["next_time"][self.agent_selection]
            )
        )
        # We do nothing, we need to only ensure each of the agents has observe how much it has harvested
        if self._agent_selector.is_last():
            self.phase = self._phase_selector.next()
        self.agent_selection = self._agent_selector.next()

    def _step_restaurant(self, action: PersonaActionChat):
        # We have a group conversation, we register that each was there and go in the next phase for everyone, since we had a group conversation
        if type(action) == PersonaActionChat:
            self.log_step_conversation(action)
            # Advance to the next phase
            for a in self.agents:
                self.internal_global_state["next_location"][a] = "home"
                self.internal_global_state["next_time"][a] = get_reflection_day(
                    self.internal_global_state["next_time"][a]
                )
            self.phase = self._phase_selector.next()
            self.agent_selection = self._agent_selector.reset()

    def _step_home(self, action: PersonaAction):
        self.internal_global_state["next_location"][
            self.agent_selection
        ] = self.POOL_LOCATION
        self.internal_global_state["next_time"][self.agent_selection] += timedelta(
            days=1
        )
        # A possible idea here is to probe the agent for some reflection / thoughts
        # NOTE do we need to register something here?

    def step(self, action: PersonaAction) -> tuple[str, HarvestingObs, dict, dict]:
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
                self.phase = self._phase_selector.next()

                # We want to see also the discussion in case no fish remain
                self.terminations = {
                    agent: (
                        self.internal_global_state["resource_in_pool"]
                        < 5  # less than 5 fish remain, so we collapse
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

    ########################################
    # Logging
    ########################################

    def log_step_harvest(
        self,
        action: PersonaActionHarvesting,
        resource_collected: int,
    ):
        tmp = {
            "agent_id": [action.agent_id],
            "round": [self.num_round],
            "action": ["harvesting"],
            "resource_in_pool_before_harvesting": [
                self.internal_global_state["resource_before_harvesting"]
            ],
            "resource_in_pool_after_harvesting": [
                self.internal_global_state["resource_in_pool"]
            ],
            "concurrent_harvesting": [True],
            "resource_collected": [resource_collected],
            "wanted_resource": [action.quantity],
            "html_interactions": [action.html_interactions],
        }
        if "sustainable_intention" in action.stats:
            tmp["sustainable_intention"] = [action.stats["sustainable_intention"]]
        df_log = pd.DataFrame(tmp, index=[len(self.df_acc)])
        self.df_acc.append(df_log)

    def log_step_conversation(self, chat: PersonaActionChat):
        for i, (p, u) in enumerate(chat.conversation):
            df_log = pd.DataFrame(
                {
                    "agent_id": p.agent_id,
                    "agent_name": p.name,
                    "round": self.num_round,
                    "action": "utterance",
                    "resource_limit": chat.conversation_resource_limit,
                    "utterance": u,
                    "html_interactions": [chat.html_interactions[i]],
                },
                index=[len(self.df_acc)],
            )
            self.df_acc.append(df_log)
        df_log = pd.DataFrame(
            {
                "agent_id": "framework",
                "agent_name": "framework",
                "round": self.num_round,
                "action": "conversation_summary",
                "resource_limit": chat.conversation_resource_limit,
                "html_interactions": [chat.html_interactions[-2]],
            },
            index=[len(self.df_acc)],
        )
        self.df_acc.append(df_log)

        df_log = pd.DataFrame(
            {
                "agent_id": "framework",
                "agent_name": "framework",
                "round": self.num_round,
                "action": "conversation_resource_limit",
                "resource_limit": chat.conversation_resource_limit,
                "html_interactions": [chat.html_interactions[-1]],
            },
            index=[len(self.df_acc)],
        )
        self.df_acc.append(df_log)
