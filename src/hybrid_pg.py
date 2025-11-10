"""
Optimized Quantum Advantage Actor-Critic (A2C) for LunarLander-v3
Includes: CSV logging, plotting, and model checkpointing

Based on 2024-2025 research findings:
1. Actor-Critic with baseline (reduces variance)
2. Entropy regularization (improves exploration)
3. Generalized Advantage Estimation (GAE) (bias-variance tradeoff)
4. Hardware-efficient ansatz (shallow circuits)
5. Hybrid quantum-classical architecture
"""

import gymnasium as gym
import numpy as np
import pennylane as qml
from pennylane import numpy as pnp
from collections import deque
import csv
import pickle
import matplotlib.pyplot as plt
from datetime import datetime


class ClassicalPreprocessor:
    """Dimensionality reduction with better initialization"""
    def __init__(self, input_dim, output_dim):
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        hidden_dim = 16
        # He initialization
        scale1 = np.sqrt(2.0 / input_dim)
        scale2 = np.sqrt(2.0 / hidden_dim)
        
        self.w1 = pnp.random.randn(input_dim, hidden_dim) * scale1
        self.b1 = pnp.zeros(hidden_dim)
        self.w2 = pnp.random.randn(hidden_dim, output_dim) * scale2
        self.b2 = pnp.zeros(output_dim)
        
        for p in [self.w1, self.b1, self.w2, self.b2]:
            p.requires_grad = True
    
    def forward(self, x):
        h = pnp.tanh(pnp.dot(x, self.w1) + self.b1)
        out = pnp.tanh(pnp.dot(h, self.w2) + self.b2)
        return out * np.pi
    
    def parameters(self):
        return [self.w1, self.b1, self.w2, self.b2]


class QuantumActor:
    """Quantum policy network with hardware-efficient ansatz"""
    def __init__(self, state_dim=8, reduced_dim=3, n_actions=4, n_layers=2):
        self.state_dim = state_dim
        self.reduced_dim = reduced_dim
        self.n_actions = n_actions
        self.n_layers = n_layers
        
        # Preprocessor
        self.preprocessor = ClassicalPreprocessor(state_dim, reduced_dim)
        
        # Quantum circuit - Hardware-efficient ansatz
        self.dev = qml.device('default.qubit', wires=reduced_dim)
        self.circuit = self._create_circuit()
        
        # Quantum weights - shallow initialization
        self.q_weights = pnp.random.uniform(
            -np.pi/4, np.pi/4,
            size=(n_layers, reduced_dim, 2),
            requires_grad=True
        )
        
        # Action-specific output weights
        self.output_weights = pnp.random.randn(n_actions, reduced_dim) * 0.1
        self.output_weights.requires_grad = True
    
    def _create_circuit(self):
        """Create quantum circuit as a class method (pickle-friendly)"""
        @qml.qnode(self.dev, interface='autograd')
        def quantum_circuit(inputs, weights):
            """Hardware-efficient ansatz (proven best for NISQ devices)"""
            # State encoding
            for i in range(self.reduced_dim):
                qml.RY(inputs[i], wires=i)
            
            # Variational layers
            for layer in range(self.n_layers):
                # Single-qubit rotations
                for i in range(self.reduced_dim):
                    qml.RY(weights[layer, i, 0], wires=i)
                    qml.RZ(weights[layer, i, 1], wires=i)
                
                # Linear entanglement
                for i in range(self.reduced_dim - 1):
                    qml.CNOT(wires=[i, i + 1])
            
            return [qml.expval(qml.PauliZ(i)) for i in range(self.reduced_dim)]
        
        return quantum_circuit
    
    def forward(self, state):
        """Get action logits"""
        reduced = self.preprocessor.forward(state)
        q_out = pnp.array(self.circuit(reduced, self.q_weights))
        logits = pnp.dot(self.output_weights, q_out)
        return logits
    
    def get_action_probs(self, state):
        """Softmax policy"""
        logits = self.forward(state)
        exp_logits = pnp.exp(logits - pnp.max(logits))
        probs = exp_logits / pnp.sum(exp_logits)
        return probs
    
    def parameters(self):
        return self.preprocessor.parameters() + [self.q_weights, self.output_weights]


class ClassicalCritic:
    """Classical value function (critic)"""
    def __init__(self, state_dim=8):
        # Two-layer network
        h1 = 64
        h2 = 32
        
        scale1 = np.sqrt(2.0 / state_dim)
        scale2 = np.sqrt(2.0 / h1)
        scale3 = np.sqrt(2.0 / h2)
        
        self.w1 = pnp.random.randn(state_dim, h1) * scale1
        self.b1 = pnp.zeros(h1)
        self.w2 = pnp.random.randn(h1, h2) * scale2
        self.b2 = pnp.zeros(h2)
        self.w3 = pnp.random.randn(h2, 1) * scale3
        self.b3 = pnp.zeros(1)
        
        for p in [self.w1, self.b1, self.w2, self.b2, self.w3, self.b3]:
            p.requires_grad = True
    
    def forward(self, state):
        """Estimate state value V(s)"""
        h1 = pnp.maximum(0, pnp.dot(state, self.w1) + self.b1)
        h2 = pnp.maximum(0, pnp.dot(h1, self.w2) + self.b2)
        v = pnp.dot(h2, self.w3) + self.b3
        return v[0]
    
    def parameters(self):
        return [self.w1, self.b1, self.w2, self.b2, self.w3, self.b3]


class TrainingLogger:
    """Logs training metrics"""
    def __init__(self, model_name):
        self.model_name = model_name
        self.timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.csv_filename = f'{model_name}_training_{self.timestamp}.csv'
        self.data = []
        
    def log_episode(self, episode, reward, steps, recent_avg):
        self.data.append({
            'episode': episode,
            'reward': reward,
            'steps': steps,
            'recent_avg': recent_avg
        })
    
    def save_csv(self):
        with open(self.csv_filename, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=[
                'episode', 'reward', 'steps', 'recent_avg'
            ])
            writer.writeheader()
            writer.writerows(self.data)
        print(f"\n✓ Training log saved: {self.csv_filename}")
        return self.csv_filename
    
    def plot_results(self, test_results=None):
        """Create training and test plots"""
        episodes = [d['episode'] for d in self.data]
        rewards = [d['reward'] for d in self.data]
        recent_avgs = [d['recent_avg'] for d in self.data]
        steps = [d['steps'] for d in self.data]
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(f'{self.model_name} - Training Results', fontsize=16)
        
        # Rewards
        ax1 = axes[0, 0]
        ax1.plot(episodes, rewards, alpha=0.3, color='blue', label='Episode Reward')
        ax1.plot(episodes, recent_avgs, linewidth=2, color='red', label='Recent 100 Avg')
        ax1.axhline(y=200, color='green', linestyle='--', label='Solved Threshold')
        ax1.set_xlabel('Episode')
        ax1.set_ylabel('Reward')
        ax1.set_title('Training Rewards')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Steps
        ax2 = axes[0, 1]
        ax2.plot(episodes, steps, linewidth=1, color='orange', alpha=0.6)
        ax2.set_xlabel('Episode')
        ax2.set_ylabel('Steps')
        ax2.set_title('Steps per Episode')
        ax2.grid(True, alpha=0.3)
        
        # Rolling statistics
        ax3 = axes[1, 0]
        window = 10
        if len(rewards) >= window:
            rolling_mean = np.convolve(rewards, np.ones(window)/window, mode='valid')
            rolling_episodes = episodes[window-1:]
            ax3.plot(rolling_episodes, rolling_mean, linewidth=2, color='purple')
            ax3.set_xlabel('Episode')
            ax3.set_ylabel('Reward (10-episode avg)')
            ax3.set_title('Rolling Average (10 episodes)')
            ax3.grid(True, alpha=0.3)
        
        # Test results
        ax4 = axes[1, 1]
        if test_results:
            test_rewards = test_results['rewards']
            ax4.hist(test_rewards, bins=15, alpha=0.7, color='green', edgecolor='black')
            ax4.axvline(x=np.mean(test_rewards), color='red', linestyle='--', linewidth=2,
                       label=f"Mean: {np.mean(test_rewards):.2f}")
            ax4.axvline(x=200, color='orange', linestyle='--', linewidth=2,
                       label='Solved (200)')
            ax4.set_xlabel('Reward')
            ax4.set_ylabel('Frequency')
            ax4.set_title(f"Test Reward Distribution ({len(test_rewards)} episodes)")
            ax4.legend()
            ax4.grid(True, alpha=0.3, axis='y')
        else:
            ax4.text(0.5, 0.5, 'Test results not available', 
                    ha='center', va='center', transform=ax4.transAxes)
            ax4.axis('off')
        
        plt.tight_layout()
        plot_filename = f'{self.model_name}_plots_{self.timestamp}.png'
        plt.savefig(plot_filename, dpi=150, bbox_inches='tight')
        print(f"✓ Plots saved: {plot_filename}")
        return plot_filename


def compute_gae(rewards, values, next_value, dones, gamma=0.99, lambda_=0.95):
    """
    Generalized Advantage Estimation (GAE)
    Balances bias-variance tradeoff better than simple returns
    """
    advantages = []
    gae = 0
    
    for t in reversed(range(len(rewards))):
        if t == len(rewards) - 1:
            next_val = next_value
        else:
            next_val = values[t + 1]
        
        # TD error
        delta = rewards[t] + gamma * next_val * (1 - dones[t]) - values[t]
        
        # GAE accumulation
        gae = delta + gamma * lambda_ * (1 - dones[t]) * gae
        advantages.insert(0, gae)
    
    return np.array(advantages)


def main():
    model_name = "QuantumA2C_LunarLander"
    
    print("="*70)
    print(f"Training: {model_name}")
    print("Quantum Advantage Actor-Critic for LunarLander-v3")
    print("="*70)
    
    env = gym.make('LunarLander-v3')
    
    # Hybrid quantum-classical A2C
    actor = QuantumActor(
        state_dim=8,
        reduced_dim=3,  # 3 qubits (research shows 2-4 is optimal)
        n_actions=4,
        n_layers=2
    )
    
    critic = ClassicalCritic(state_dim=8)
    
    # Separate optimizers (common practice in A2C)
    actor_optimizer = qml.AdamOptimizer(stepsize=3e-4)
    critic_optimizer = qml.AdamOptimizer(stepsize=1e-3)
    
    # Hyperparameters
    gamma = 0.99
    lambda_gae = 0.95  # GAE parameter
    entropy_coef = 0.01  # Entropy regularization
    value_loss_coef = 0.5
    max_grad_norm = 0.5  # Gradient clipping
    
    total_episodes = 20
    episodes_per_update = 5  # Update every 5 episodes
    max_steps = 500
    
    logger = TrainingLogger(model_name)
    
    best_reward = -float('inf')
    episode_count = 0
    recent_rewards = deque(maxlen=100)
    
    print(f"\nHyperparameters:")
    print(f"  Total Episodes: {total_episodes}")
    print(f"  Episodes per Update: {episodes_per_update}")
    print(f"  Actor LR: 3e-4 | Critic LR: 1e-3")
    print(f"  Gamma: {gamma} | GAE Lambda: {lambda_gae}")
    print(f"  Entropy Coef: {entropy_coef} | Value Loss Coef: {value_loss_coef}")
    print("="*70)
    
    while episode_count < total_episodes:
        # Collect trajectories
        trajectories = []
        batch_rewards = []
        
        for _ in range(episodes_per_update):
            state, _ = env.reset()
            trajectory = {
                'states': [],
                'actions': [],
                'rewards': [],
                'log_probs': [],
                'values': [],
                'dones': []
            }
            
            episode_reward = 0
            episode_steps = 0
            
            for step in range(max_steps):
                # Get action probabilities and value
                probs = actor.get_action_probs(state)
                value = critic.forward(state)
                
                # Sample action
                probs_np = np.array(probs)
                action = np.random.choice(actor.n_actions, p=probs_np)
                
                # Take action
                next_state, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                
                # Store transition
                trajectory['states'].append(state)
                trajectory['actions'].append(action)
                trajectory['rewards'].append(reward)
                trajectory['log_probs'].append(np.log(probs_np[action] + 1e-10))
                trajectory['values'].append(float(value))
                trajectory['dones'].append(done)
                
                episode_reward += reward
                episode_steps += 1
                state = next_state
                
                if done:
                    break
            
            # Get bootstrap value for last state
            if not done:
                next_value = float(critic.forward(state))
            else:
                next_value = 0.0
            
            # Compute advantages using GAE
            advantages = compute_gae(
                trajectory['rewards'],
                trajectory['values'],
                next_value,
                trajectory['dones'],
                gamma,
                lambda_gae
            )
            trajectory['advantages'] = advantages
            trajectory['returns'] = advantages + np.array(trajectory['values'])
            
            trajectories.append(trajectory)
            batch_rewards.append(episode_reward)
            recent_rewards.append(episode_reward)
            episode_count += 1
            
            # Log individual episode
            recent_avg = np.mean(recent_rewards) if len(recent_rewards) > 0 else episode_reward
            logger.log_episode(episode_count, episode_reward, episode_steps, recent_avg)
        
        # Update networks
        # Combine all trajectories
        all_states = []
        all_actions = []
        all_advantages = []
        all_returns = []
        all_old_log_probs = []
        
        for traj in trajectories:
            all_states.extend(traj['states'])
            all_actions.extend(traj['actions'])
            all_advantages.extend(traj['advantages'])
            all_returns.extend(traj['returns'])
            all_old_log_probs.extend(traj['log_probs'])
        
        # Normalize advantages
        all_advantages = np.array(all_advantages)
        all_advantages = (all_advantages - all_advantages.mean()) / (all_advantages.std() + 1e-8)
        
        # Update critic
        def critic_loss(w1, b1, w2, b2, w3, b3):
            total_loss = 0
            for i, state in enumerate(all_states):
                h1 = pnp.maximum(0, pnp.dot(state, w1) + b1)
                h2 = pnp.maximum(0, pnp.dot(h1, w2) + b2)
                v_pred = pnp.dot(h2, w3) + b3
                
                v_target = all_returns[i]
                total_loss += (v_pred[0] - v_target) ** 2
            
            return value_loss_coef * total_loss / len(all_states)
        
        critic_params = critic.parameters()
        critic_params = critic_optimizer.step(critic_loss, *critic_params)
        for i, p in enumerate(critic_params):
            critic.parameters()[i] = p
        
        # Update actor
        def actor_loss(w1, b1, w2, b2, q_weights, output_weights):
            total_loss = 0
            total_entropy = 0
            
            for i, state in enumerate(all_states):
                # Forward pass
                h = pnp.tanh(pnp.dot(state, w1) + b1)
                reduced = pnp.tanh(pnp.dot(h, w2) + b2) * np.pi
                q_out = pnp.array(actor.circuit(reduced, q_weights))
                logits = pnp.dot(output_weights, q_out)
                
                # Softmax
                exp_logits = pnp.exp(logits - pnp.max(logits))
                probs = exp_logits / pnp.sum(exp_logits)
                
                # Log probability of taken action
                log_prob = pnp.log(probs[all_actions[i]] + 1e-10)
                
                # Policy gradient loss (negative because we're minimizing)
                advantage = all_advantages[i]
                total_loss -= log_prob * advantage
                
                # Entropy bonus (encourages exploration)
                entropy = -pnp.sum(probs * pnp.log(probs + 1e-10))
                total_entropy += entropy
            
            # Combine policy loss and entropy bonus
            avg_loss = total_loss / len(all_states)
            avg_entropy = total_entropy / len(all_states)
            
            return avg_loss - entropy_coef * avg_entropy
        
        actor_params = actor.parameters()
        actor_params = actor_optimizer.step(actor_loss, *actor_params)
        for i, p in enumerate(actor_params):
            actor.parameters()[i] = p
        
        # Logging
        avg_reward = np.mean(batch_rewards)
        max_reward = np.max(batch_rewards)
        recent_avg = np.mean(recent_rewards) if len(recent_rewards) > 0 else avg_reward
        
        if avg_reward > best_reward:
            best_reward = avg_reward
        
        if episode_count % 50 == 0:
            print(f"Episode {episode_count:5d} | "
                  f"Batch Avg: {avg_reward:7.2f} | "
                  f"Recent 100: {recent_avg:7.2f} | "
                  f"Best: {best_reward:7.2f}")
        
        # Success criteria
        if recent_avg > 200:
            print(f"\n🎉 Solved at episode {episode_count}!")
            print(f"   Recent 100 episodes average: {recent_avg:.2f}")
            break
    
    env.close()
    
    # Save training CSV
    logger.save_csv()
    
    # Testing
    print("\n" + "="*70)
    print("Testing Phase")
    print("="*70)
    
    test_env = gym.make('LunarLander-v3')
    test_rewards = []
    test_steps = []
    successes = 0
    
    for test_ep in range(50):
        state, _ = test_env.reset()
        total_reward = 0
        done = False
        steps = 0
        
        while not done and steps < 500:
            probs = actor.get_action_probs(state)
            action = int(np.argmax(np.array(probs)))  # Greedy at test time
            state, reward, terminated, truncated, _ = test_env.step(action)
            total_reward += reward
            done = terminated or truncated
            steps += 1
        
        test_rewards.append(total_reward)
        test_steps.append(steps)
        if total_reward >= 200:
            successes += 1
        
        if (test_ep + 1) % 10 == 0:
            print(f"Test {test_ep + 1:2d}/50 | Recent avg: {np.mean(test_rewards[-10:]):7.2f}")
    
    test_avg = np.mean(test_rewards)
    test_std = np.std(test_rewards)
    test_median = np.median(test_rewards)
    test_max = np.max(test_rewards)
    test_min = np.min(test_rewards)
    
    print("="*70)
    print(f"Test Results (50 episodes):")
    print(f"  Average:  {test_avg:7.2f} ± {test_std:.2f}")
    print(f"  Median:   {test_median:7.2f}")
    print(f"  Best:     {test_max:7.2f}")
    print(f"  Worst:    {test_min:7.2f}")
    print(f"  Avg Steps: {np.mean(test_steps):6.1f}")
    print(f"  Success:  {successes}/50 ({successes/50*100:.0f}%)")
    print("="*70)
    
    test_env.close()
    
    # Save test results to CSV
    test_csv = f'{model_name}_test_results_{logger.timestamp}.csv'
    with open(test_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['episode', 'reward', 'steps'])
        for i, (reward, steps) in enumerate(zip(test_rewards, test_steps), 1):
            writer.writerow([i, reward, steps])
        writer.writerow(['average', test_avg, np.mean(test_steps)])
        writer.writerow(['std', test_std, np.std(test_steps)])
        writer.writerow(['median', test_median, '-'])
        writer.writerow(['max', test_max, '-'])
        writer.writerow(['min', test_min, '-'])
        writer.writerow(['success_rate', f'{successes}/50', f'{successes/50*100:.1f}%'])
    print(f"✓ Test results saved: {test_csv}")
    
    # Create plots
    test_results = {'rewards': test_rewards}
    logger.plot_results(test_results)
    
    # Save model (save only parameters, not the circuit - pickle-friendly)
    model_data = {
        'config': {
            'state_dim': 8,
            'reduced_dim': 3,
            'n_actions': 4,
            'n_layers': 2,
            'actor_lr': 3e-4,
            'critic_lr': 1e-3,
            'gamma': gamma,
            'lambda_gae': lambda_gae,
            'entropy_coef': entropy_coef,
            'value_loss_coef': value_loss_coef,
            'episodes_per_update': episodes_per_update
        },
        'actor_parameters': {
            'preprocessor_w1': actor.preprocessor.w1,
            'preprocessor_b1': actor.preprocessor.b1,
            'preprocessor_w2': actor.preprocessor.w2,
            'preprocessor_b2': actor.preprocessor.b2,
            'q_weights': actor.q_weights,
            'output_weights': actor.output_weights
        },
        'critic_parameters': {
            'w1': critic.w1,
            'b1': critic.b1,
            'w2': critic.w2,
            'b2': critic.b2,
            'w3': critic.w3,
            'b3': critic.b3
        },
        'test_results': {
            'avg': test_avg,
            'std': test_std,
            'median': test_median,
            'max': test_max,
            'min': test_min,
            'success_rate': successes/50
        }
    }
    model_file = f'{model_name}_model_{logger.timestamp}.pkl'
    with open(model_file, 'wb') as f:
        pickle.dump(model_data, f)
    print(f"✓ Model saved: {model_file}")
    
    print("\n" + "="*70)
    print("✓ Training Complete!")
    print(f"  - Training CSV: {logger.csv_filename}")
    print(f"  - Test CSV: {test_csv}")
    print(f"  - Plots: {model_name}_plots_{logger.timestamp}.png")
    print(f"  - Model: {model_file}")
    print("="*70)


if __name__ == '__main__':
    main()