import json
import os
import shutil
import uuid

import hydra
import numpy as np
import statsmodels.stats.proportion as smprop
import tqdm
from hydra import compose, initialize
from hydra.core.global_hydra import GlobalHydra
from omegaconf import DictConfig, OmegaConf
from transformers import set_seed

import wandb
from simulation.persona.common import PersonaIdentity
from simulation.utils import ModelWandbWrapper, WandbLogger
from pathfinder import get_model


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))
    set_seed(cfg.seed)

    model = get_model(cfg.llm.path, cfg.llm.is_api, cfg.seed, cfg.llm.backend)
    logger = WandbLogger(
        f"subskills_check/fishing/{cfg.code_version}",
        OmegaConf.to_object(cfg),
        debug=cfg.debug,
    )

    experiment_storage = os.path.join(
        os.path.dirname(__file__),
        f"./results/subskills_check_{cfg.code_version}/{logger.run_name}",
    )
    os.makedirs(experiment_storage, exist_ok=True)

    wrapper = ModelWandbWrapper(
        model,
        render=cfg.llm.render,
        wanbd_logger=logger,
        temperature=cfg.llm.temperature,
        top_p=cfg.llm.top_p,
        seed=cfg.seed,
        is_api=cfg.llm.is_api,
    )

    if cfg.llm.out_format == "freeform":
        from .reasoning_free_format import (
            prompt_action_choose_amount_of_fish_to_catch,
            prompt_action_choose_amount_of_fish_to_catch_universalization,
            prompt_reflection_if_all_fisher_that_same_quantity,
            prompt_shrinking_limit,
            prompt_shrinking_limit_asumption,
            prompt_simple_reflection_if_all_fisher_that_same_quantity,
            prompt_simple_shrinking_limit,
            prompt_simple_shrinking_limit_assumption,
        )
    else:
        # We found freefrom makes more sense, since we don't destory the model's output probability distribqution
        raise ValueError(f"Unknown out_format: {cfg.llm.out_format}")

    if cfg.llm.cot_prompt == "deep_breath":
        cot_prompt = "Take a deep breath and work on this problem step-by-step."
    elif cfg.llm.cot_prompt == "think_step_by_step":
        cot_prompt = "Let's think step-by-step."
    else:
        raise ValueError(f"Unknown cot_prompt: {cfg.llm.cot_prompt}")

    NUM_RUNS = 150
    if cfg.debug:
        NUM_RUNS = 2

    class TestCase:
        name: str

        def __init__(self, name) -> None:
            self.name = name

        def run(
            self,
        ):
            logs = []
            for args in self.get_args_iterator():
                try:
                    answer, html_prompt = self.prompt(**args)
                    passed, correct_answer = self.pass_condition(answer, **args)
                    logs.append(
                        {
                            "args": self.serialize_args(args),
                            "answer": answer,
                            "passed": passed,
                            "correct_answer": correct_answer,
                            "error": "OK",
                            "html_prompt": html_prompt,
                        }
                    )
                except Exception as e:
                    print(f"Error: {e}")
                    _, correct_answer = self.pass_condition(0, **args)
                    logs.append(
                        {
                            "args": self.serialize_args(args),
                            "answer": None,
                            "correct_answer": correct_answer,
                            "passed": False,
                            "error": f"Error: {e}",
                            "html_prompt": "parse_error",
                        }
                    )

            ALPHA = 0.05
            ci = smprop.proportion_confint(
                sum([log["passed"] for log in logs]), len(logs), alpha=ALPHA
            )

            test = {
                "name": self.name,
                "instances": logs,
                "score_mean": np.mean([log["passed"] for log in logs]),
                "score_std": np.std([log["passed"] for log in logs]),
                "score_ci_lower": ci[0],
                "score_ci_upper": ci[1],
                "config": OmegaConf.to_object(cfg),
            }
            json.dump(test, open(f"{experiment_storage}/{self.name}.json", "w"))

        def get_args_iterator(self):
            raise NotImplementedError

        def prompt(self, *, args):
            raise NotImplementedError

        def serialize_args(self, args: dict[str, any]):
            res = {}
            for k, v in args.items():
                if isinstance(v, PersonaIdentity):
                    res[k] = v.agent_id
                else:
                    res[k] = v
            return res

    ############################
    # Test cases
    ############################

    class MathConsequenceAfterFishingSameAmount(TestCase):
        def __init__(self, name="math_consequence_after_fishing_same_amount") -> None:
            super().__init__(name)

        def get_args_iterator(self):
            return [
                {
                    "persona": PersonaIdentity(
                        "John",
                        "John",
                    ),
                    "num_tonnes_lake": 100,
                    "num_tonnes_fisher": 10,
                }
            ] * NUM_RUNS

        def prompt(self, *, persona, num_tonnes_lake, num_tonnes_fisher):
            return prompt_simple_reflection_if_all_fisher_that_same_quantity(
                wrapper,
                cot_prompt,
                persona,
                num_tonnes_lake,
                num_tonnes_fisher,
                cfg.llm.few_shots,
                cfg.llm.description_version,
            )

        def pass_condition(self, answer, persona, num_tonnes_lake, num_tonnes_fisher):
            correct_answer = max(
                0, min(100, (num_tonnes_lake - num_tonnes_fisher * 5) * 2)
            )
            return answer == correct_answer, correct_answer

    class SimConsequenceAfterFishingSameAmount(MathConsequenceAfterFishingSameAmount):
        def __init__(self, name="sim_consequence_after_fishing_same_amount") -> None:
            super().__init__(name)

        def prompt(self, *, persona, num_tonnes_lake, num_tonnes_fisher):
            return prompt_reflection_if_all_fisher_that_same_quantity(
                wrapper,
                cot_prompt,
                persona,
                num_tonnes_lake,
                num_tonnes_fisher,
                cfg.llm.few_shots,
                cfg.llm.description_version,
            )

    class MathShrinkingLimit(TestCase):
        def __init__(self, name="math_shrinking_limit") -> None:
            super().__init__(name)

        def get_args_iterator(self):
            return [
                {
                    "persona": PersonaIdentity(
                        "John",
                        "John",
                    ),
                    "num_tonnes_lake": 100,
                }
            ] * NUM_RUNS

        def prompt(
            self,
            *,
            persona,
            num_tonnes_lake,
        ):
            return prompt_simple_shrinking_limit(
                wrapper,
                cot_prompt,
                persona,
                num_tonnes_lake,
                cfg.llm.few_shots,
                cfg.llm.description_version,
            )

        def pass_condition(self, answer, persona, num_tonnes_lake):
            correct_answer = (num_tonnes_lake // 2) // 5
            return answer == correct_answer, correct_answer

    class MathShrinkingLimitAssumption(TestCase):
        def __init__(self, name="math_shrinking_limit_assumption") -> None:
            super().__init__(name)

        def get_args_iterator(self):
            return [
                {
                    "persona": PersonaIdentity(
                        "John",
                        "John",
                    ),
                    "num_tonnes_lake": 100,
                }
            ] * NUM_RUNS

        def prompt(
            self,
            *,
            persona,
            num_tonnes_lake,
        ):
            return prompt_simple_shrinking_limit_assumption(
                wrapper,
                cot_prompt,
                persona,
                num_tonnes_lake,
                cfg.llm.few_shots,
                cfg.llm.description_version,
            )

        def pass_condition(self, answer, persona, num_tonnes_lake):
            correct_answer = (num_tonnes_lake // 2) // 5
            return answer == correct_answer, correct_answer

    class SimShrinkingLimit(MathShrinkingLimit):
        def __init__(self, name="sim_shrinking_limit") -> None:
            super().__init__(name)

        def prompt(
            self,
            *,
            persona,
            num_tonnes_lake,
        ):
            return prompt_shrinking_limit(
                wrapper,
                cot_prompt,
                persona,
                num_tonnes_lake,
                cfg.llm.few_shots,
                cfg.llm.description_version,
            )

    class SimShrinkingLimitAssumption(MathShrinkingLimitAssumption):
        def __init__(self, name="sim_shrinking_limit_assumption") -> None:
            super().__init__(name)

        def prompt(
            self,
            *,
            persona,
            num_tonnes_lake,
        ):
            return prompt_shrinking_limit_asumption(
                wrapper,
                cot_prompt,
                persona,
                num_tonnes_lake,
                cfg.llm.few_shots,
                cfg.llm.description_version,
            )

    class SimCatchFishStandardPersona(TestCase):
        def __init__(self, name="sim_catch_fish_standard_persona") -> None:
            super().__init__(name)

        def get_args_iterator(self):
            return [
                {
                    "persona": PersonaIdentity(
                        "John",
                        "John",
                    ),
                    "num_tonnes_lake": 100,
                }
            ] * NUM_RUNS

        def prompt(
            self,
            *,
            persona,
            num_tonnes_lake,
        ):
            return prompt_action_choose_amount_of_fish_to_catch(
                wrapper,
                cot_prompt,
                persona,
                num_tonnes_lake,
                cfg.llm.few_shots,
                cfg.llm.description_version,
            )

        def pass_condition(self, answer, persona, num_tonnes_lake):
            correct_answer = (num_tonnes_lake // 2) // 5
            if correct_answer == 0:
                return answer == correct_answer, correct_answer
            return answer <= correct_answer and answer > 0, correct_answer

    class SimUnivCatchFishStandardPersona(TestCase):
        def __init__(self, name="sim_catch_fish_universalization") -> None:
            super().__init__(name)

        def get_args_iterator(self):
            return [
                {
                    "persona": PersonaIdentity(
                        "John",
                        "John",
                    ),
                    "num_tonnes_lake": 100,
                }
            ] * NUM_RUNS

        def prompt(
            self,
            *,
            persona,
            num_tonnes_lake,
        ):
            return prompt_action_choose_amount_of_fish_to_catch_universalization(
                wrapper,
                cot_prompt,
                persona,
                num_tonnes_lake,
                cfg.llm.few_shots,
                cfg.llm.description_version,
            )

        def pass_condition(self, answer, persona, num_tonnes_lake):
            correct_answer = (num_tonnes_lake // 2) // 5
            if correct_answer == 0:
                return answer == correct_answer, correct_answer
            return answer <= correct_answer and answer > 0, correct_answer

    ### Multiple

    def get_random_persona():
        persona_names = ["John", "Kate", "Jack", "Emma", "Luke"]
        name = persona_names[np.random.randint(0, len(persona_names))]
        return PersonaIdentity(name, name)

    class MultipleMathShrinkingLimit(MathShrinkingLimit):
        def __init__(self, name="multiple_math_shrinking_limit") -> None:
            super().__init__(name)

        def get_args_iterator(self):
            return [
                {
                    "persona": get_random_persona(),
                    "num_tonnes_lake": int(i),
                }
                for i in np.random.randint(10, 101, NUM_RUNS)
            ]

    class MultipleSimShrinkingLimit(SimShrinkingLimit):
        def __init__(self, name="multiple_sim_shrinking_limit") -> None:
            super().__init__(name)

        def get_args_iterator(self):
            return [
                {
                    "persona": get_random_persona(),
                    "num_tonnes_lake": int(i),
                }
                for i in np.random.randint(10, 101, NUM_RUNS)
            ]

    class MultipleMathShrinkingLimitAssumption(MathShrinkingLimitAssumption):
        def __init__(self, name="multiple_math_shrinking_limit_assumption") -> None:
            super().__init__(name)

        def get_args_iterator(self):
            return [
                {
                    "persona": get_random_persona(),
                    "num_tonnes_lake": int(i),
                }
                for i in np.random.randint(10, 101, NUM_RUNS)
            ]

    class MultipleSimShrinkingLimitAssumption(SimShrinkingLimitAssumption):
        def __init__(self, name="multiple_sim_shrinking_limit_assumption") -> None:
            super().__init__(name)

        def get_args_iterator(self):
            return [
                {
                    "persona": get_random_persona(),
                    "num_tonnes_lake": int(i),
                }
                for i in np.random.randint(10, 101, NUM_RUNS)
            ]

    class MultipleSimCatchFishStandardPersona(SimCatchFishStandardPersona):
        def __init__(self, name="multiple_sim_catch_fish_standard_persona") -> None:
            super().__init__(name)

        def get_args_iterator(self):
            return [
                {
                    "persona": get_random_persona(),
                    "num_tonnes_lake": int(i),
                }
                for i in np.random.randint(10, 101, NUM_RUNS)
            ]

    class MultipleSimUniverCatchFishStandardPersona(SimUnivCatchFishStandardPersona):
        def __init__(self, name="multiple_sim_universalization_catch_fish") -> None:
            super().__init__(name)

        def get_args_iterator(self):
            return [
                {
                    "persona": get_random_persona(),
                    "num_tonnes_lake": int(i),
                }
                for i in np.random.randint(10, 101, NUM_RUNS)
            ]

    class MultipleMathConsequenceAfterFishingSameAmount(
        MathConsequenceAfterFishingSameAmount
    ):
        def __init__(
            self, name="multiple_math_consequence_after_fishing_same_amount"
        ) -> None:
            super().__init__(name)

        def get_args_iterator(self):
            return [
                {
                    "persona": get_random_persona(),
                    "num_tonnes_lake": int(i),
                    "num_tonnes_fisher": int(np.random.randint(0, (i // 5) + 1)),
                }
                for i in np.random.randint(10, 101, NUM_RUNS)
            ]

    class MultipleSimConsequenceAfterFishingSameAmount(
        SimConsequenceAfterFishingSameAmount
    ):
        def __init__(
            self, name="multiple_sim_consequence_after_fishing_same_amount"
        ) -> None:
            super().__init__(name)

        def get_args_iterator(self):
            return [
                {
                    "persona": get_random_persona(),
                    "num_tonnes_lake": int(i),
                    "num_tonnes_fisher": int(np.random.randint(0, (i // 5) + 1)),
                }
                for i in np.random.randint(10, 101, NUM_RUNS)
            ]

    test_cases_2 = [
        MultipleMathShrinkingLimit(),
        MultipleSimShrinkingLimit(),
        MultipleMathConsequenceAfterFishingSameAmount(),
        MultipleSimConsequenceAfterFishingSameAmount(),
        MultipleSimCatchFishStandardPersona(),
        MultipleSimUniverCatchFishStandardPersona(),
        MultipleMathShrinkingLimitAssumption(),
        MultipleSimShrinkingLimitAssumption(),
    ]

    if cfg.split == "single":
        test_cases = test_cases_2
    elif int(cfg.split) == 1:
        test_cases = test_cases_2[:2]
    elif int(cfg.split) == 2:
        test_cases = test_cases_2[2:4]
    elif int(cfg.split) == 3:
        test_cases = test_cases_2[4:6]
    elif int(cfg.split) == 4:
        test_cases = test_cases_2[6:]

    for test_case in tqdm.tqdm(test_cases):
        test_case.run()


if __name__ == "__main__":
    OmegaConf.register_resolver("uuid", lambda: f"run_{uuid.uuid4()}")
    main()
