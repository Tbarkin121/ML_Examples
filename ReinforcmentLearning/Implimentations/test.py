
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.vec_env import VecFrameStack
from stable_baselines3 import A2C, DQN, PPO
import gym
from stable_baselines3.common.vec_env import VecVideoRecorder, DummyVecEnv
from stable_baselines3.common.env_util import make_vec_env

from modified_envs import AcrobotEnv

seed=0
def make_env(rank, seed=0):
    def _init():
        # env = CartPoleEnv()
        env = AcrobotEnv()
        # env.set_target(target=1.0, weight=0.2)
        env.seed(seed + rank)
        return env
    return _init


# Create the environment

def create_environment(n_env):

    env = DummyVecEnv([make_env(i, seed=seed) for i in range(n_env)])
    

    num_obs = env.observation_space.shape[0]
    num_actions = env.action_space.n
    
    # Box(4,) means that it is a Vector with 4 components
    print("Observation space:", env.observation_space)
    print("Observation Shape:", env.observation_space.shape[0])
    # Discrete(2) means that there is two discrete actions
    print("Action space:", env.action_space)

    print('num_obs = {}'.format(num_obs))
    print('num_actions = {}'.format(num_actions))

    # Wrap OpenAI Gym's `env.step` call as an operation in a TensorFlow function.
    # This would allow it to be included in a callable TensorFlow graph.
    return env, num_obs, num_actions

env, num_obs, num_actions = create_environment(n_env = 10)

eval_env = AcrobotEnv()
#%%
model = PPO('MlpPolicy', env, verbose=1, tensorboard_log="./logs/ppo_doubleP")
# Train the agent
model.learn(total_timesteps=int(1e6))
# Save the agent
model.save("PPO-AcrobotEnv")

#%%
from stable_baselines3.common.evaluation import evaluate_policy
model = PPO.load("PPO-AcrobotEnv", env=eval_env)
eval_env = AcrobotEnv()
# Evaluate the agent
# NOTE: If you use wrappers with your environment that modify rewards,
#       this will be reflected here. To evaluate with original rewards,
#       wrap environment in a "Monitor" wrapper before other wrappers.
mean_reward, std_reward = evaluate_policy(model, eval_env, n_eval_episodes=10)

# Enjoy trained agent
obs = eval_env.reset()
for i in range(1000):
    action, _states = model.predict(obs, deterministic=True)
    obs, rewards, dones, info = eval_env.step(action)
    eval_env.render()
eval_env.close()