import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Настройки
sns.set(style="whitegrid")
window_size = 100
plots_dir = "plots"
os.makedirs(plots_dir, exist_ok=True)

# Загружаем данные
try:
    rewards = np.load("rewards.npy")
except FileNotFoundError:
    print("Файл rewards.npy не найден. Обучите модель сначала.")
    exit()

# 1. Скользящее среднее награды
plt.figure(figsize=(12, 5))
plt.plot(np.convolve(rewards, np.ones(window_size)/window_size, mode='valid'), label="Средняя награда")
plt.title("Скользящее среднее награды (окно из 100 эпизодов)")
plt.xlabel("Эпизод")
plt.ylabel("Награда")
plt.legend()
plt.grid(True)
plt.savefig(os.path.join(plots_dir, "reward_curve.png"))
plt.show()

# 2. Доля успешных эпизодов
successes = [1 if r > 0 else 0 for r in rewards]
plt.figure(figsize=(12, 5))
plt.plot(np.convolve(successes, np.ones(window_size)/window_size, mode='valid'), label="Успех (%)")
plt.title("Доля успешных эпизодов (скользящее среднее)")
plt.xlabel("Эпизод")
plt.ylabel("Успех (%)")
plt.legend()
plt.grid(True)
plt.savefig(os.path.join(plots_dir, "success_rate.png"))
plt.show()

# 3. Шаги до завершения эпизода
try:
    steps = np.load("steps_per_episode.npy")
    plt.figure(figsize=(12, 5))
    plt.plot(steps, label="Шаги на эпизод")
    plt.title("Количество шагов на эпизод")
    plt.xlabel("Эпизод")
    plt.ylabel("Шаги")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(plots_dir, "steps_to_complete.png"))
    plt.show()
except FileNotFoundError:
    print("Файл steps_per_episode.npy не найден — пропускаем график шагов.")