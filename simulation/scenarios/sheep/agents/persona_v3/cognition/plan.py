from pathfinder import assistant, system, user
from simulation.persona.cognition.plan import PlanComponent
from simulation.utils import ModelWandbWrapper


class SheepPlanComponent(PlanComponent):
    def __init__(
        self,
        model: ModelWandbWrapper,
        model_framework: ModelWandbWrapper,
    ):
        super().__init__(model, model_framework)
