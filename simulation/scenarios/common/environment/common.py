from datetime import datetime

from simulation.persona.common import ChatObservation, PersonaOberservation


class HarvestingObs(PersonaOberservation):
    current_resource_num: int

    agent_resource_num: dict[str, int]

    def __init__(
        self,
        phase: str,
        current_location: str,
        current_location_agents: dict[str, str],
        current_time: datetime,
        events: list,
        context: str,
        chat: ChatObservation,
        current_resource_num: int,
        agent_resource_num: dict[str, int],
        before_harvesting_sustainability_threshold: int,
    ) -> None:
        super().__init__(
            phase,
            current_location,
            current_location_agents,
            current_time,
            events,
            context,
            chat,
        )
        self.current_resource_num = current_resource_num
        self.agent_resource_num = agent_resource_num
        self.before_harvesting_sustainability_threshold = (
            before_harvesting_sustainability_threshold
        )
