import gym
import numpy as np
from collections import deque

# Создаём среду
env = gym.make('FrozenLake-v1')
num_states = env.observation_space.n
num_actions = env.action_space.n

# Гиперпараметры
alpha = 0.8     # скорость обучения
gamma = 0.95    # дисконтирование
epsilon = 1.0   # начальная вероятность случайного действия
episodes = 5000  # количество эпизодов

# Инициализируем Q-таблицу
Q = np.zeros([num_states, num_actions])

# Для логирования
rewards = []
steps_per_episode = []
window_size = 100
recent_successes = deque(maxlen=window_size)

print("Начало обучения...")

for i in range(episodes):
    state, _ = env.reset()
    done = False
    total_reward = 0
    step_count = 0

    while not done:
        # ε-greedy выбор действия
        if np.random.uniform() < epsilon:
            action = env.action_space.sample()  # случайное действие
        else:
            action = np.argmax(Q[state])        # жадное действие

        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

        # Обновление Q-значения
        Q[state, action] = Q[state, action] + alpha * (
            reward + gamma * np.max(Q[next_state]) - Q[state, action]
        )

        total_reward += reward
        state = next_state
        step_count += 1

        if done:
            break

    rewards.append(total_reward)
    steps_per_episode.append(step_count)
    recent_successes.append(total_reward)

    # Логирование прогресса
    if i % 100 == 0 and i > 0:
        avg_success = sum(recent_successes) / window_size
        print(f"Эпизод {i}, Средняя награда (окно {window_size}): {avg_success:.2f}")

    # Убираем epsilon, но не ниже 0.01
    epsilon = max(0.01, epsilon * 0.995)

env.close()

# Сохраняем результаты
np.save("q_table.npy", Q)
np.save("rewards.npy", np.array(rewards))
np.save("steps_per_episode.npy", np.array(steps_per_episode))

print("Обучение завершено.")