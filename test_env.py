import gym

env = gym.make('FrozenLake-v1', render_mode='human')
observation, info = env.reset()
print(f"Начальное состояние: {observation}")

step_count = 0
max_steps = 100

print("\n--- Начало тестирования ---")

terminated = False
truncated = False

while not (terminated or truncated) and step_count < max_steps:
    action = env.action_space.sample()
    observation, reward, terminated, truncated, info = env.step(action)
    step_count += 1

    print(f"Шаг {step_count}:")
    print(f"  Действие: {action}")
    print(f"  Состояние: {observation}")
    print(f"  Награда: {reward}")
    print(f"  Завершено: {terminated}, Прервано: {truncated}\n")

env.close()
print("--- Тестирование завершено ---")