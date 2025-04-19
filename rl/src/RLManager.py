class RLManager:
    def __init__(self):
        # initialize the model here
        # self.model = ...
        pass

    def preprocess_observation(self, observation):
        # do nothing unless the model needs it
        return observation

    def get_action(self, observation) -> int:
        _obs = self.preprocess_observation(observation)
        # feed _obs into model
        # return self.model.predict(_obs)
        return 0
