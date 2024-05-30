import os

import numpy as np
from simulation.persona import EmbeddingModel
from simulation.persona.common import PersonaIdentity
from simulation.utils import ModelWandbWrapper
from omegaconf import DictConfig, OmegaConf

from .environment import PollutionConcurrentEnv, PollutionPerturbationEnv


def run(
    cfg: DictConfig,
    logger: ModelWandbWrapper,
    wrapper: ModelWandbWrapper,
    embedding_model: EmbeddingModel,
    experiment_storage: str,
):
    if cfg.agent.agent_package == "persona_v3":
        from .agents.persona_v3 import PollutionPersona
        from .agents.persona_v3.cognition import utils as cognition_utils

        if cfg.agent.system_prompt == "v3":
            cognition_utils.SYS_VERSION = "v3"
        elif cfg.agent.system_prompt == "v3_nocom":
            cognition_utils.SYS_VERSION = "v3_nocom"
        else:
            cognition_utils.SYS_VERSION = "v1"
        if cfg.agent.cot_prompt == "think_step_by_step":
            cognition_utils.REASONING = "think_step_by_step"
        elif cfg.agent.cot_prompt == "deep_breath":
            cognition_utils.REASONING = "deep_breath"
    else:
        raise ValueError(f"Unknown agent package: {cfg.agent.agent_package}")

    personas = {
        f"persona_{i}": PollutionPersona(
            cfg.agent,
            wrapper,
            embedding_model,
            os.path.join(experiment_storage, f"persona_{i}"),
        )
        for i in range(5)
    }

    # NOTE persona characteristics, up to design choices
    num_personas = cfg.personas.num

    identities = {}
    for i in range(num_personas):
        persona_id = f"persona_{i}"
        identities[persona_id] = PersonaIdentity(
            agent_id=persona_id, **cfg.personas[persona_id]
        )

    # Standard setup
    agent_name_to_id = {obj.name: k for k, obj in identities.items()}
    agent_name_to_id["framework"] = "framework"
    agent_id_to_name = {v: k for k, v in agent_name_to_id.items()}

    for persona in personas:
        personas[persona].init_persona(persona, identities[persona], social_graph=None)

    for persona in personas:
        for other_persona in personas:
            # also add self reference, for conversation
            personas[persona].add_reference_to_other_persona(personas[other_persona])

    if cfg.env.class_name == "pollution_perturbation_env":
        env = PollutionPerturbationEnv(cfg.env, experiment_storage, agent_id_to_name)
    elif cfg.env.class_name == "pollution_perturbation_concurrent_env":
        env = PollutionConcurrentEnv(cfg.env, experiment_storage, agent_id_to_name)
    else:
        raise ValueError(f"Unknown environment class: {cfg.env.class_name}")
    agent_id, obs = env.reset()

    while True:
        agent = personas[agent_id]
        action = agent.loop(obs)

        (
            agent_id,
            obs,
            rewards,
            termination,
        ) = env.step(action)

        stats = {}
        STATS_KEYS = [
            "conversation_resource_limit",
            *[f"persona_{i}_collected_resource" for i in range(5)],
        ]
        for s in STATS_KEYS:
            if s in action.stats:
                stats[s] = action.stats[s]

        if np.any(list(termination.values())):
            logger.log_game(
                {
                    "num_resource": obs.current_resource_num,
                    **stats,
                },
                last_log=True,
            )
            break
        else:
            logger.log_game(
                {
                    "num_resource": obs.current_resource_num,
                    **stats,
                }
            )

        logger.save(experiment_storage, agent_name_to_id)

    env.save_log()
    for persona in personas:
        personas[persona].memory.save()
