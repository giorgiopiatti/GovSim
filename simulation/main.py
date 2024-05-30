import os
import shutil
import uuid

import hydra
import numpy as np
import wandb
from hydra import compose, initialize
from hydra.core.global_hydra import GlobalHydra
from omegaconf import DictConfig, OmegaConf
from transformers import set_seed

from simulation.utils import ModelWandbWrapper, WandbLogger
from pathfinder import get_model

from .persona import EmbeddingModel
from .scenarios.fishing.run import run as run_scenario_fishing
from .scenarios.pollution.run import run as run_scenario_pollution
from .scenarios.sheep.run import run as run_scenario_sheep


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))
    set_seed(cfg.seed)

    model = get_model(cfg.llm.path, cfg.llm.is_api, cfg.seed, cfg.llm.backend)
    logger = WandbLogger(cfg.experiment.name, OmegaConf.to_object(cfg), debug=cfg.debug)

    experiment_storage = os.path.join(
        os.path.dirname(__file__),
        f"./results/{cfg.experiment.name}/{logger.run_name}",
    )

    wrapper = ModelWandbWrapper(
        model,
        render=cfg.llm.render,
        wanbd_logger=logger,
        temperature=cfg.llm.temperature,
        top_p=cfg.llm.top_p,
        seed=cfg.seed,
        is_api=cfg.llm.is_api,
    )
    embedding_model = EmbeddingModel(device="cpu")

    if cfg.experiment.scenario == "fishing":
        run_scenario_fishing(
            cfg.experiment,
            logger,
            wrapper,
            embedding_model,
            experiment_storage,
        )
    elif cfg.experiment.scenario == "sheep":
        run_scenario_sheep(
            cfg.experiment,
            logger,
            wrapper,
            embedding_model,
            experiment_storage,
        )
    elif cfg.experiment.scenario == "pollution":
        run_scenario_pollution(
            cfg.experiment,
            logger,
            wrapper,
            embedding_model,
            experiment_storage,
        )
    else:
        raise ValueError(f"Unknown experiment.scenario: {cfg.experiment.scenario}")

    hydra_log_path = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
    shutil.copytree(f"{hydra_log_path}/.hydra/", f"{experiment_storage}/.hydra/")
    shutil.copy(f"{hydra_log_path}/main.log", f"{experiment_storage}/main.log")
    # shutil.rmtree(hydra_log_path)

    artifact = wandb.Artifact("hydra", type="log")
    artifact.add_dir(f"{experiment_storage}/.hydra/")
    artifact.add_file(f"{experiment_storage}/.hydra/config.yaml")
    artifact.add_file(f"{experiment_storage}/.hydra/hydra.yaml")
    artifact.add_file(f"{experiment_storage}/.hydra/overrides.yaml")
    wandb.run.log_artifact(artifact)


if __name__ == "__main__":
    OmegaConf.register_resolver("uuid", lambda: f"run_{uuid.uuid4()}")
    main()
