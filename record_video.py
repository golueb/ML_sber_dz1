import gym
import os

# Создаем папку для видео, если её нет
os.makedirs("videos", exist_ok=True)

# Создаем среду с rgb_array для рендеринга
env = gym.make('FrozenLake-v1', render_mode='rgb_array')
env = gym.wrappers.RecordVideo(env, video_folder='./videos')

# Сброс среды
observation, info = env.reset()

print("Запись видео начата...")

for _ in range(100):
    action = env.action_space.sample()
    observation, reward, terminated, truncated, info = env.step(action)

    if terminated or truncated:
        observation, info = env.reset()
        break

env.close()
print("Видео сохранено в папке './videos'")