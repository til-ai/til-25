import functools

from gymnasium.spaces.utils import flatten, flatten_space
from pettingzoo.utils.env import ActionType, AECEnv, AgentID, ObsType
from pettingzoo.utils.wrappers.base import BaseWrapper


class FlattenDictWrapper(BaseWrapper[AgentID, ObsType, ActionType]):
    """This wrapper flattens the Dict observation space."""

    def __init__(
        self,
        env: AECEnv[AgentID, ObsType, ActionType],
    ):
        super().__init__(env)

    def observe(self, agent: AgentID) -> ObsType | None:
        obs = super().observe(agent)
        return flatten(self.env.observation_space(agent), obs)

    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent):
        space = super().observation_space(agent)
        return flatten_space(space)
