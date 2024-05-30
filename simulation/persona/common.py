class PersonaIdentity:
    def __init__(
        self,
        agent_id: str,
        name: str,
        age: int = None,
        innate_traits: str = None,
        background: str = None,
        goals: str = None,
        behavior: str = None,
        customs: str = None,
    ) -> None:
        self.agent_id = agent_id
        self.name = name
        self.age = age
        self.innate_traits = innate_traits
        self.background = background
        self.goals = goals
        self.behavior = behavior
        self.customs = customs

    def get_identiy_stable_set(self) -> str:
        # Before we also used: lm += f"Here is a brief description of {persona.name}."
        res = "Consider the following persona:\n"
        res += f"- Name: {self.name}\n"
        if self.age:
            res += f"- Age: {self.age}\n"
        if self.background:
            res += f"- Background: {self.background}\n"
        if self.innate_traits:
            res += f"- Innate Traits: {self.innate_traits}\n"
        if self.goals:
            res += f"- Goals: {self.goals}\n"
        if self.behavior:
            res += f"- Behavior: {self.behavior}\n"
        if self.customs:
            res += f"- Customs: {self.customs}\n"
        return res


class ChatObservation:
    init_persona: PersonaIdentity
    other_personas: list[PersonaIdentity]
    conversation: list[tuple[str, str]]
    summary: str
    current_location: str

    def __init__(
        self,
        init_persona: PersonaIdentity,
        other_personas: list[PersonaIdentity],
        conversation: list[tuple[str, str]],
        summary: str,
        current_location: str,
    ) -> None:
        self.init_persona = init_persona
        self.other_personas = other_personas
        self.conversation = conversation
        self.summary = summary
        self.current_location = current_location


import datetime


class PersonaOberservation:
    phase: str

    current_location: str
    current_location_agents: dict[str, str]
    current_time: datetime.datetime

    events: list
    context: str
    chat: ChatObservation

    def __init__(
        self,
        phase: str,
        current_location: str,
        current_location_agents: dict[str, str],
        current_time: datetime.datetime,
        events: list,
        context: str,
        chat: ChatObservation,
    ) -> None:
        self.phase = phase
        self.current_location = current_location
        self.current_location_agents = current_location_agents
        self.current_time = current_time
        self.events = events
        self.context = context
        self.chat = chat


class PersonaAction:
    agent_id: str
    location: str
    stats = (
        dict  # NOTE used for wandb logging and csv logging, not so clean TODO refactor
    )
    html_interactions: list[str]

    def __init__(
        self, agent_id, location: str, stats={}, html_interactions=None
    ) -> None:
        self.agent_id = agent_id
        self.location = location
        self.stats = stats
        self.html_interactions = html_interactions


class PersonaActionHarvesting(PersonaAction):
    quantity: int

    def __init__(
        self, agent_id, location: str, quantity: int, stats={}, html_interactions=None
    ) -> None:
        super().__init__(agent_id, location, stats, html_interactions)
        self.quantity = quantity


class PersonaActionChat(PersonaAction):
    conversation: list[tuple[PersonaIdentity, str]]
    conversation_resource_limit: int

    def __init__(
        self,
        agent_id,
        location: str,
        conversation: list[tuple[PersonaIdentity, str]],
        conversation_resource_limit: int,
        stats={},
        html_interactions=None,
    ) -> None:
        super().__init__(agent_id, location, stats, html_interactions)
        self.conversation = conversation
        self.conversation_resource_limit = conversation_resource_limit


class PersonaEvent:
    description: str
    created: datetime
    expiration: datetime
    always_include: bool

    def __init__(
        self,
        description: str,
        created: datetime,
        expiration: datetime,
        always_include: bool = False,
    ) -> None:
        self.description = description
        self.created = created
        self.expiration = expiration
        self.always_include = always_include
