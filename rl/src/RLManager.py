class RLManager:
    def __init__(self):
        """
        Initializes the RLManager. Should initialize the model, and any other

        e.g. `self.model = ...`
        """

    def preprocess_observation(self, observation):
        """
        Modify any persistent state as necessary, and return your augmented observation.
        """
        return observation

    def get_action(self, observation: list[int]) -> int:
        """
        From input observation, get the action your agent will take.
        """
        _obs = self.preprocess_observation(observation)
        # feed _obs into model
        # return self.model.predict(_obs)
        return 0

    def reset():
        """
        Reset any persistent state information that was set previously.
        """
        return
