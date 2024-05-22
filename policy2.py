from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.callbacks import BaseCallback
import rocket

class TrajectoryCallback(BaseCallback):
    def __init__(self, env, check_freq: int, verbose=1):
        super(TrajectoryCallback, self).__init__(verbose)
        self.check_freq = check_freq
        self.env = env
        self.episode_count = 0

    def _on_step(self) -> bool:
        if 'dones' in self.locals:
            for done in self.locals['dones']:
                if done:
                    self.episode_count += 1
                    if self.episode_count % self.check_freq == 0:
                        # 경로 시각화
                        self.env.rocket.show_path_from_state_buffer()
        return True

env = rocket.RocketEnv()
check_env(env, warn = True)

model = PPO("MlpPolicy", env, verbose = 1)

callback = TrajectoryCallback(env, check_freq=5)

model.learn(total_timesteps = 100000, callback=callback)

model.save("ppo_rocket")
