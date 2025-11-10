import os
import sys
import gymnasium as gym
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from stable_baselines3 import PPO, DDPG, DQN, TD3
from stable_baselines3.common.vec_env import DummyVecEnv

# ──────────────────────────────────────────────
# SETUP LOGGING TO TERMINAL + FILE
# ──────────────────────────────────────────────
os.makedirs("models", exist_ok=True)
os.makedirs("results", exist_ok=True)

log_file = open("results/training_log.txt", "w")
sys.stdout = open("results/training_log.txt", "w")
sys.stderr = sys.stdout  # redirect errors too

# Also keep a copy in the console (optional)
class DualLogger:
    def __init__(self, *streams):
        self.streams = streams
    def write(self, message):
        for s in self.streams:
            s.write(message)
            s.flush()
    def flush(self):
        for s in self.streams:
            s.flush()

sys.stdout = DualLogger(sys.stdout, sys.__stdout__)
sys.stderr = sys.stdout

# ──────────────────────────────────────────────
# MODELS AND ENVIRONMENTS
# ──────────────────────────────────────────────
models = {
    "PPO": PPO,
    "DDPG": DDPG,
    "DQN": DQN,
    "TD3": TD3
}

env_model_map = {
    "CartPole-v1": ["DQN", "PPO"],
    "LunarLander-v2": ["DQN", "PPO"],
    "Pendulum-v1": ["DDPG", "TD3"],
    "BipedalWalker-v3": ["DDPG", "TD3"]
}

# ──────────────────────────────────────────────
# EVALUATION FUNCTION
# ──────────────────────────────────────────────
def evaluate(model, env, n_eval_episodes=5):
    rewards = []
    for _ in range(n_eval_episodes):
        obs, _ = env.reset()
        done = False
        total_reward = 0
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            if isinstance(action, np.ndarray) and action.shape == ():
                action = int(action)
            obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            total_reward += reward
        rewards.append(total_reward)
    return np.mean(rewards), np.std(rewards)

# ──────────────────────────────────────────────
# TRAINING FUNCTION
# ──────────────────────────────────────────────
def train_model(model_name, env_id, timesteps=50_000):
    ModelClass = models[model_name]
    path = f"models/{model_name}_{env_id}.zip"

    if os.path.exists(path):
        print(f"⏩ Skipping {model_name} on {env_id} (already trained)")
        model = ModelClass.load(path)
        env = gym.make(env_id)
        mean, _ = evaluate(model, env)
        return mean

    print(f"\n🚀 Training {model_name} on {env_id} ...")
    env = DummyVecEnv([lambda: gym.make(env_id)])
    model = ModelClass("MlpPolicy", env, verbose=0)
    model.learn(total_timesteps=timesteps)
    model.save(path)
    print(f"✅ Saved {path}")

    env = gym.make(env_id)
    mean, _ = evaluate(model, env)
    return mean

# ──────────────────────────────────────────────
# MAIN LOOP
# ──────────────────────────────────────────────
def main():
    results = []

    for env_id, model_list in env_model_map.items():
        for model_name in model_list:
            try:
                mean_reward = train_model(model_name, env_id, timesteps=50_000)
                results.append({"Model": model_name, "Environment": env_id, "MeanReward": mean_reward})
            except Exception as e:
                print(f"❌ {model_name} failed on {env_id}: {e}")
                results.append({"Model": model_name, "Environment": env_id, "MeanReward": None})

    df = pd.DataFrame(results)
    df.to_csv("results/classical_results.csv", index=False)
    print("\n📁 Saved results → results/classical_results.csv")

    plot_results(df)

# ──────────────────────────────────────────────
# PLOTTING
# ──────────────────────────────────────────────
def plot_results(df):
    plt.figure(figsize=(10, 6))
    envs = df["Environment"].unique()
    bar_width = 0.2
    x = np.arange(len(envs))
    offsets = {"PPO": -bar_width, "DQN": 0, "DDPG": bar_width, "TD3": 2 * bar_width}

    for model_name, offset in offsets.items():
        subset = df[df["Model"] == model_name]
        rewards = subset["MeanReward"].tolist()
        plt.bar(x + offset, rewards, width=bar_width, label=model_name)

    plt.xticks(x, envs)
    plt.ylabel("Mean Reward (5 eval episodes)")
    plt.title("RL Performance Comparison (PPO, DQN, DDPG, TD3)")
    plt.legend()
    plt.tight_layout()
    plt.savefig("results/classical_comparison.png")
    plt.show()

# ──────────────────────────────────────────────
if __name__ == "__main__":
    main()
    print("\n📝 Training log saved to → results/training_log.txt")
