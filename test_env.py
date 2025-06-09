import gym
import numpy as np
import os
from gym.wrappers import RecordVideo

# Проверяем наличие Q-таблицы
if not os.path.exists("q_table.npy"):
    print("Файл q_table.npy не найден. Обучите модель сначала.")
    exit()

# Загружаем Q-таблицу
Q = np.load("q_table.npy")

# Создаём среду с записью видео
video_folder = "videos"
os.makedirs(video_folder, exist_ok=True)

env = gym.make('FrozenLake-v1', render_mode='rgb_array')
env = RecordVideo(env, video_folder=video_folder, name_prefix="frozenlake-qlearning")

# Тестирование агента
state, _ = env.reset()
done = False

while not done:
    action = np.argmax(Q[state])
    next_state, reward, terminated, truncated, _ = env.step(action)
    done = terminated or truncated
    state = next_state

env.close()
print("Видео успешно записано в папку 'videos'")