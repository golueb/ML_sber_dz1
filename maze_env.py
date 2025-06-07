import gym

# Указываем render_mode для отображения
env = gym.make('FrozenLake-v1', render_mode='human')

# Сбрасываем среду — получаем начальное состояние
observation, info = env.reset()

for _ in range(100):
    action = env.action_space.sample()  # случайное действие
    observation, reward, terminated, truncated, info = env.step(action)

    if terminated or truncated:
        observation, info = env.reset()
        break

env.close()