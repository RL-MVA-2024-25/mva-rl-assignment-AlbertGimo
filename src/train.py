from gymnasium.wrappers import TimeLimit
from env_hiv import HIVPatient
import numpy as np
import numpy.random as rand
from FQI import *
from pathlib import Path

fixed_env = TimeLimit(
    env=FastHIVPatient(domain_randomization=False), max_episode_steps=200
)  # The time wrapper limits the number of steps in an episode at 200.
# Now the floor is yours to implement the agent and train it.
rand_env = TimeLimit(
    env=FastHIVPatient(domain_randomization=True), max_episode_steps=200
)

# You have to implement your own agent.
# Don't modify the methods names and signatures, but you can add methods.
# ENJOY!
class ProjectAgent:
    def __init__(self,path="models/best_FQI.pkl",augment_state=False):
        self.agent = HIV_FQI(augment_state=augment_state)
        self.path = path
        file = Path(path)
        if file.is_file():
            self.agent.load(path)

    def act(self, observation, use_random=False):
        return self.agent.greedy_action(observation)

    def save(self, path):
        self.path = path
        self.agent.save(path)

    def load(self):
        self.agent.load(self.path)

    if __name__ == '__main__':
        agent = HIV_FQI(augment_state=False)
        agent.train()
        # agent.train(path="models/FQI3.216e+10.pkl")
