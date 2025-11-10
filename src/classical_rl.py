# src/classical_rl.py
import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import gymnasium as gym
from stable_baselines3 import PPO, DQN, DDPG, TD3

# ──────────────────────────────────────────────
# CONFIG
# ──────────────────────────────────────────────
MODELS_DIR = "models"
RESULTS_DIR = "results"
EVAL_EPISODES = 100
os.makedirs(RESULTS_DIR, exist_ok=True)

MODEL_CLASSES = {
    "PPO": PPO,
    "DQN": DQN,
    "DDPG": DDPG,
    "TD3": TD3
}

# ──────────────────────────────────────────────
# EVALUATION UTILITIES
# ──────────────────────────────────────────────
def evaluate_model(model, env, n_eval_episodes=100):
    rewards, lengths = [], []
    for _ in range(n_eval_episodes):
        obs, _ = env.reset()
        done = False
        total_reward, ep_len = 0, 0
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            total_reward += reward
            ep_len += 1
        rewards.append(total_reward)
        lengths.append(ep_len)
    return np.array(rewards), np.array(lengths)


def bootstrap_ci(data, n_bootstrap=1000, ci=0.95):
    n = len(data)
    boot_means = [np.mean(np.random.choice(data, size=n, replace=True)) for _ in range(n_bootstrap)]
    lower = np.percentile(boot_means, (1 - ci) / 2 * 100)
    upper = np.percentile(boot_means, (1 + ci) / 2 * 100)
    return lower, upper


def detect_model_and_env(filename):
    base = os.path.splitext(filename)[0]
    for model_name in MODEL_CLASSES.keys():
        if base.startswith(model_name + "_"):
            env_id = base[len(model_name) + 1:]
            return model_name, env_id
    return None, None


# ──────────────────────────────────────────────
# BENCHMARK FUNCTION
# ──────────────────────────────────────────────
def benchmark_all():
    summary_rows, episode_rows = [], []
    model_files = [f for f in os.listdir(MODELS_DIR) if f.endswith(".zip")]

    if not model_files:
        print("❌ No trained models found in /models/")
        return

    for filename in model_files:
        model_path = os.path.join(MODELS_DIR, filename)
        model_name, env_id = detect_model_and_env(filename)

        if not model_name or not env_id:
            print(f"⚠️ Skipping unrecognized file: {filename}")
            continue

        print(f"\n🚀 Evaluating {model_name} on {env_id} ...")

        try:
            ModelClass = MODEL_CLASSES[model_name]
            model = ModelClass.load(model_path)
            env = gym.make(env_id)
        except Exception as e:
            print(f"❌ Failed to load {filename}: {e}")
            continue

        start = time.time()
        rewards, lengths = evaluate_model(model, env, n_eval_episodes=EVAL_EPISODES)
        elapsed = time.time() - start

        mean_r, std_r = rewards.mean(), rewards.std()
        median_r = np.median(rewards)
        p25, p75 = np.percentile(rewards, [25, 75])
        ci_low, ci_high = bootstrap_ci(rewards)
        mean_len = lengths.mean()

        print(f"✅ {model_name} | {env_id} | Mean={mean_r:.2f} ± {std_r:.2f} | Median={median_r:.2f}")

        summary_rows.append({
            "Model": model_name,
            "Environment": env_id,
            "MeanReward": mean_r,
            "StdReward": std_r,
            "MedianReward": median_r,
            "P25": p25,
            "P75": p75,
            "CI95_Lower": ci_low,
            "CI95_Upper": ci_high,
            "MeanEpisodeLen": mean_len,
            "EvalTimeSec": elapsed
        })

        for i, (r, l) in enumerate(zip(rewards, lengths)):
            episode_rows.append({
                "Model": model_name,
                "Environment": env_id,
                "Episode": i,
                "Reward": r,
                "Length": l
            })

        env.close()

    df_summary = pd.DataFrame(summary_rows)
    df_episode = pd.DataFrame(episode_rows)

    df_summary.to_csv(os.path.join(RESULTS_DIR, "benchmark_summary.csv"), index=False)
    df_episode.to_csv(os.path.join(RESULTS_DIR, "benchmark_per_episode.csv"), index=False)

    print("\n📊 Saved → benchmark_summary.csv & benchmark_per_episode.csv")
    plot_all(df_summary, df_episode)


# ──────────────────────────────────────────────
# PLOTTING
# ──────────────────────────────────────────────
def plot_all(df_summary, df_episode):
    envs = sorted(df_summary["Environment"].unique())
    models = sorted(df_summary["Model"].unique())
    bar_width = 0.15
    x = np.arange(len(envs))

    plt.figure(figsize=(10, 6))
    for i, model_name in enumerate(models):
        subset = df_summary[df_summary["Model"] == model_name]
        # Align rewards to environment order
        rewards = []
        for env in envs:
            match = subset[subset["Environment"] == env]
            rewards.append(match["MeanReward"].values[0] if not match.empty else np.nan)
        plt.bar(x + i * bar_width, rewards, width=bar_width, label=model_name)

    plt.xticks(x + bar_width, envs)
    plt.ylabel("Mean Reward (100 eval episodes)")
    plt.title("Average Performance Across Environments")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "benchmark_mean_rewards.png"))

    # ── 2️⃣ Reward variability (box plot)
    plt.figure(figsize=(10, 6))
    for env_id in envs:
        subset = df_episode[df_episode["Environment"] == env_id]
        data = [subset[subset["Model"] == m]["Reward"].values for m in models]
        positions = np.arange(len(models)) + list(envs).index(env_id) * (len(models) + 1)
        plt.boxplot(data, positions=positions, widths=0.6)
    plt.xticks([])
    plt.ylabel("Episode Reward")
    plt.title("Reward Distribution per Environment")
    plt.savefig(os.path.join(RESULTS_DIR, "benchmark_boxplot.png"))

    # ── 3️⃣ Episode length comparison
    plt.figure(figsize=(10, 6))
    for model_name in models:
        subset = df_summary[df_summary["Model"] == model_name]
        plt.plot(subset["Environment"], subset["MeanEpisodeLen"], marker="o", label=model_name)

    plt.ylabel("Mean Episode Length")
    plt.title("Episode Length Comparison")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "benchmark_episode_length.png"))

    plt.show()
    print("✅ Saved 3 plots in /results/: mean_rewards, boxplot, and episode_length.")



# ──────────────────────────────────────────────
if __name__ == "__main__":
    benchmark_all()
