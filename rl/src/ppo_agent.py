# env_setup.py
from til_environment import gridworld
from til_environment.flatten_dict import FlattenDictWrapper
import supersuit as ss

# train_ppo.py
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import DummyVecEnv
from stable_baselines3.common.utils import set_random_seed
from supersuit import pettingzoo_env_to_vec_env_v1

def make_env():
    env = gridworld.env(
        env_wrappers=[],          # clear default wrappers
        render_mode=None,         # disable rendering for training
        debug=False,
        novice=True               # True if you're a novice team
    )

    # Custom Wrapping
    env = FlattenDictWrapper(env)
    env = ss.pad_observations_v0(env)
    env = ss.pad_action_space_v0(env)
    env = ss.frame_stack_v2(env, 4)

    return env


def train(seed=0, timesteps=500_000):
    set_random_seed(seed)

    env = make_env()
    vec_env = pettingzoo_env_to_vec_env_v1(env)
    vec_env = DummyVecEnv([lambda: vec_env])

    model = PPO(
        "MlpPolicy",
        vec_env,
        verbose=1,
        tensorboard_log="./ppo_tilai_tensorboard/",
        n_steps=2048,
        batch_size=64,
        gae_lambda=0.95,
        gamma=0.99,
        ent_coef=0.01,
        learning_rate=2.5e-4,
        vf_coef=0.5,
        max_grad_norm=0.5,
    )

    model.learn(total_timesteps=timesteps)
    model.save("ppo_tilai")
    

if __name__ == "__main__":
    train()
