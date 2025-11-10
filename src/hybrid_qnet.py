"""
Optimized Quantum Dueling DQN for Taxi-v3 (~5-6 minutes runtime)
Includes: CSV logging, plotting, and model checkpointing

Key optimizations:
- Reduced epochs: 30 instead of 100
- Fewer episodes per epoch: 15 instead of 25
- Fewer updates per epoch: 5 instead of 10
- Maintains all original architecture improvements
- Fixed pickle serialization for quantum circuit
"""

import gymnasium as gym
import numpy as np
import pennylane as qml
from pennylane import numpy as pnp
from collections import deque
import random
import csv
import pickle
import matplotlib.pyplot as plt
from datetime import datetime


class PrioritizedReplayBuffer:
    """Prioritized Experience Replay for more efficient learning"""
    def __init__(self, capacity=10000, alpha=0.6, beta_start=0.4, beta_frames=10000):
        self.capacity = capacity
        self.alpha = alpha
        self.beta_start = beta_start
        self.beta_frames = beta_frames
        self.frame = 1
        
        self.buffer = []
        self.priorities = np.zeros(capacity, dtype=np.float32)
        self.position = 0
    
    def push(self, state, action, reward, next_state, done):
        """Add with max priority initially"""
        max_priority = self.priorities.max() if self.buffer else 1.0
        
        if len(self.buffer) < self.capacity:
            self.buffer.append((state, action, reward, next_state, done))
        else:
            self.buffer[self.position] = (state, action, reward, next_state, done)
        
        self.priorities[self.position] = max_priority
        self.position = (self.position + 1) % self.capacity
    
    def sample(self, batch_size):
        """Sample with priorities"""
        N = len(self.buffer)
        if N == self.capacity:
            priorities = self.priorities
        else:
            priorities = self.priorities[:N]
        
        probs = priorities ** self.alpha
        probs /= probs.sum()
        
        indices = np.random.choice(N, batch_size, p=probs, replace=False)
        
        beta = min(1.0, self.beta_start + self.frame * (1.0 - self.beta_start) / self.beta_frames)
        weights = (N * probs[indices]) ** (-beta)
        weights /= weights.max()
        self.frame += 1
        
        samples = [self.buffer[idx] for idx in indices]
        states, actions, rewards, next_states, dones = zip(*samples)
        
        return (
            np.array(states, dtype=np.int32),
            np.array(actions, dtype=np.int32),
            np.array(rewards, dtype=np.float32),
            np.array(next_states, dtype=np.int32),
            np.array(dones, dtype=bool),
            indices,
            weights
        )
    
    def update_priorities(self, indices, td_errors):
        """Update priorities based on TD errors"""
        for idx, error in zip(indices, td_errors):
            self.priorities[idx] = abs(error) + 1e-6
    
    def __len__(self):
        return len(self.buffer)


class OptimizedQuantumDuelingDQN:
    """
    Quantum Dueling DQN with RealAmplitudes ansatz
    Separates state value V(s) from action advantages A(s,a)
    """
    def __init__(self, n_states=500, n_qubits=4, n_actions=6, n_layers=2):
        self.n_states = n_states
        self.n_qubits = n_qubits
        self.n_actions = n_actions
        self.n_layers = n_layers
        
        # State embedding with He initialization
        embedding_dim = 16
        self.embeddings = pnp.random.randn(n_states, embedding_dim) * np.sqrt(2.0 / n_states)
        self.embeddings.requires_grad = True
        
        # Projection to quantum space with He init
        self.w_proj = pnp.random.randn(embedding_dim, n_qubits) * np.sqrt(2.0 / embedding_dim)
        self.b_proj = pnp.zeros(n_qubits)
        self.w_proj.requires_grad = True
        self.b_proj.requires_grad = True
        
        # Quantum circuit - RealAmplitudes ansatz
        self.dev = qml.device('default.qubit', wires=n_qubits)
        self.circuit = self._create_circuit()
        
        # Quantum weights - shallow initialization
        self.q_weights = pnp.random.uniform(
            -np.pi/4, np.pi/4,
            size=(n_layers, n_qubits),
            requires_grad=True
        )
        
        # Dueling architecture: separate V(s) and A(s,a) streams
        self.v_w = pnp.random.randn(n_qubits) * np.sqrt(2.0 / n_qubits)
        self.v_b = pnp.array([0.0])
        self.v_w.requires_grad = True
        self.v_b.requires_grad = True
        
        # Advantage heads A(s,a) for each action
        self.adv_heads = []
        for _ in range(n_actions):
            w = pnp.random.randn(n_qubits) * np.sqrt(2.0 / n_qubits)
            b = pnp.array([0.0])
            w.requires_grad = True
            b.requires_grad = True
            self.adv_heads.append((w, b))
        
        # Target network parameters
        self.target_embeddings = pnp.copy(self.embeddings)
        self.target_w_proj = pnp.copy(self.w_proj)
        self.target_b_proj = pnp.copy(self.b_proj)
        self.target_q_weights = pnp.copy(self.q_weights)
        self.target_v_w = pnp.copy(self.v_w)
        self.target_v_b = pnp.copy(self.v_b)
        self.target_adv_heads = [(pnp.copy(w), pnp.copy(b)) for w, b in self.adv_heads]
    
    def _create_circuit(self):
        """Create quantum circuit as a class method (pickle-friendly)"""
        @qml.qnode(self.dev, interface='autograd')
        def quantum_circuit(inputs, weights):
            """RealAmplitudes ansatz - proven best for DQN in 2025 research"""
            for i in range(self.n_qubits):
                qml.RY(inputs[i], wires=i)
            
            for layer in range(self.n_layers):
                for i in range(self.n_qubits):
                    qml.RY(weights[layer, i], wires=i)
                
                for i in range(self.n_qubits - 1):
                    qml.CZ(wires=[i, i + 1])
            
            return [qml.expval(qml.PauliZ(i)) for i in range(self.n_qubits)]
        
        return quantum_circuit
    
    def get_q_values(self, state_idx, use_target=False):
        """Dueling architecture: Q(s,a) = V(s) + (A(s,a) - mean(A(s,·)))"""
        if use_target:
            embedded = self.target_embeddings[state_idx]
            encoded = pnp.tanh(pnp.dot(embedded, self.target_w_proj) + self.target_b_proj) * np.pi
            q_features = pnp.array(self.circuit(encoded, self.target_q_weights))
            
            v = pnp.dot(self.target_v_w, q_features) + self.target_v_b[0]
            
            advantages = pnp.array([
                pnp.dot(self.target_adv_heads[a][0], q_features) + self.target_adv_heads[a][1][0]
                for a in range(self.n_actions)
            ])
            
            q_values = v + (advantages - pnp.mean(advantages))
        else:
            embedded = self.embeddings[state_idx]
            encoded = pnp.tanh(pnp.dot(embedded, self.w_proj) + self.b_proj) * np.pi
            q_features = pnp.array(self.circuit(encoded, self.q_weights))
            
            v = pnp.dot(self.v_w, q_features) + self.v_b[0]
            
            advantages = pnp.array([
                pnp.dot(self.adv_heads[a][0], q_features) + self.adv_heads[a][1][0]
                for a in range(self.n_actions)
            ])
            
            q_values = v + (advantages - pnp.mean(advantages))
        
        return q_values
    
    def select_action(self, state_idx, epsilon):
        """Epsilon-greedy with main network"""
        if np.random.random() < epsilon:
            return np.random.randint(0, self.n_actions)
        else:
            q_values = self.get_q_values(state_idx, use_target=False)
            return int(np.argmax(np.array(q_values)))
    
    def soft_update_target(self, tau=0.005):
        """Soft update target network"""
        self.target_embeddings = (1 - tau) * self.target_embeddings + tau * self.embeddings
        self.target_w_proj = (1 - tau) * self.target_w_proj + tau * self.w_proj
        self.target_b_proj = (1 - tau) * self.target_b_proj + tau * self.b_proj
        self.target_q_weights = (1 - tau) * self.target_q_weights + tau * self.q_weights
        self.target_v_w = (1 - tau) * self.target_v_w + tau * self.v_w
        self.target_v_b = (1 - tau) * self.target_v_b + tau * self.v_b
        
        for i in range(len(self.adv_heads)):
            w, b = self.adv_heads[i]
            tw, tb = self.target_adv_heads[i]
            self.target_adv_heads[i] = ((1 - tau) * tw + tau * w, (1 - tau) * tb + tau * b)
    
    def parameters(self):
        """All trainable parameters"""
        params = [self.embeddings, self.w_proj, self.b_proj, self.q_weights, self.v_w, self.v_b]
        for w, b in self.adv_heads:
            params.extend([w, b])
        return params


class TrainingLogger:
    """Logs training metrics"""
    def __init__(self, model_name):
        self.model_name = model_name
        self.timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.csv_filename = f'{model_name}_training_{self.timestamp}.csv'
        self.data = []
        
    def log_epoch(self, epoch, avg_reward, max_reward, avg_steps, success_rate, epsilon):
        self.data.append({
            'epoch': epoch,
            'avg_reward': avg_reward,
            'max_reward': max_reward,
            'avg_steps': avg_steps,
            'success_rate': success_rate,
            'epsilon': epsilon
        })
    
    def save_csv(self):
        with open(self.csv_filename, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=[
                'epoch', 'avg_reward', 'max_reward', 'avg_steps', 'success_rate', 'epsilon'
            ])
            writer.writeheader()
            writer.writerows(self.data)
        print(f"\n✓ Training log saved: {self.csv_filename}")
        return self.csv_filename
    
    def plot_results(self, test_results=None):
        """Create training and test plots"""
        epochs = [d['epoch'] for d in self.data]
        avg_rewards = [d['avg_reward'] for d in self.data]
        max_rewards = [d['max_reward'] for d in self.data]
        success_rates = [d['success_rate'] * 100 for d in self.data]
        avg_steps = [d['avg_steps'] for d in self.data]
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(f'{self.model_name} - Training Results', fontsize=16)
        
        # Rewards
        ax1 = axes[0, 0]
        ax1.plot(epochs, avg_rewards, label='Average', linewidth=2, color='blue')
        ax1.plot(epochs, max_rewards, label='Max', linewidth=1, alpha=0.7, color='green')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Reward')
        ax1.set_title('Training Rewards')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Success rate
        ax2 = axes[0, 1]
        ax2.plot(epochs, success_rates, linewidth=2, color='green')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Success Rate (%)')
        ax2.set_title('Success Rate Over Time')
        ax2.grid(True, alpha=0.3)
        
        # Average steps
        ax3 = axes[1, 0]
        ax3.plot(epochs, avg_steps, linewidth=2, color='orange')
        ax3.set_xlabel('Epoch')
        ax3.set_ylabel('Steps')
        ax3.set_title('Average Steps per Episode')
        ax3.grid(True, alpha=0.3)
        
        # Test results
        ax4 = axes[1, 1]
        if test_results:
            test_rewards = test_results['rewards']
            ax4.hist(test_rewards, bins=15, alpha=0.7, color='purple', edgecolor='black')
            ax4.axvline(x=np.mean(test_rewards), color='red', linestyle='--', linewidth=2,
                       label=f"Mean: {np.mean(test_rewards):.2f}")
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


def main():
    model_name = "OptimizedQuantumDuelingDQN"
    
    print("="*70)
    print(f"Training: {model_name}")
    print("Optimized for ~5-6 minute runtime")
    print("="*70)
    
    env = gym.make('Taxi-v3')
    
    # Optimized agent
    agent = OptimizedQuantumDuelingDQN(
        n_states=500,
        n_qubits=4,
        n_actions=6,
        n_layers=2
    )
    
    # Prioritized replay
    replay_buffer = PrioritizedReplayBuffer(
        capacity=10000,
        alpha=0.6,
        beta_start=0.4,
        beta_frames=8000
    )
    
    logger = TrainingLogger(model_name)
    
    # Optimized hyperparameters for speed
    optimizer = qml.AdamOptimizer(stepsize=5e-4)
    
    epochs = 30  # Reduced from 100
    episodes_per_epoch = 15  # Reduced from 25
    batch_size = 64
    gamma = 0.99
    epsilon_start = 1.0
    epsilon_end = 0.01
    epsilon_decay = 0.996
    tau = 0.005
    
    warmup_episodes = 10  # Reduced from 20
    n_updates = 5  # Reduced from 10
    
    epsilon = epsilon_start
    best_reward = -float('inf')
    
    print(f"\nHyperparameters:")
    print(f"  Epochs: {epochs} | Episodes/Epoch: {episodes_per_epoch} | Updates: {n_updates}")
    print(f"  Learning Rate: {5e-4} | Batch Size: {batch_size} | Warmup: {warmup_episodes}")
    print("="*70)
    
    total_episodes = 0
    
    for epoch in range(epochs):
        epoch_rewards = []
        epoch_steps = []
        
        # Collect experience
        for ep in range(episodes_per_epoch):
            state, _ = env.reset()
            done = False
            total_reward = 0
            steps = 0
            
            while not done and steps < 100:
                if total_episodes < warmup_episodes:
                    action = np.random.randint(0, 6)
                else:
                    action = agent.select_action(state, epsilon)
                
                next_state, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                
                replay_buffer.push(state, action, reward, next_state, done)
                
                total_reward += reward
                state = next_state
                steps += 1
            
            epoch_rewards.append(total_reward)
            epoch_steps.append(steps)
            total_episodes += 1
        
        # Training after warmup
        if total_episodes >= warmup_episodes and len(replay_buffer) >= batch_size:
            for _ in range(n_updates):
                states, actions, rewards, next_states, dones, indices, weights = replay_buffer.sample(batch_size)
                
                # Loss function with 16 parameters
                def loss_fn(embeddings, w_proj, b_proj, q_weights, v_w, v_b,
                           adv0_w, adv0_b, adv1_w, adv1_b, adv2_w, adv2_b,
                           adv3_w, adv3_b, adv4_w, adv4_b, adv5_w, adv5_b):
                    
                    adv_params = [adv0_w, adv0_b, adv1_w, adv1_b, adv2_w, adv2_b,
                                 adv3_w, adv3_b, adv4_w, adv4_b, adv5_w, adv5_b]
                    
                    total_loss = 0
                    
                    for i in range(len(states)):
                        embedded = embeddings[states[i]]
                        encoded = pnp.tanh(pnp.dot(embedded, w_proj) + b_proj) * np.pi
                        q_features = pnp.array(agent.circuit(encoded, q_weights))
                        
                        v = pnp.dot(v_w, q_features) + v_b[0]
                        advantages = pnp.array([
                            pnp.dot(adv_params[a*2], q_features) + adv_params[a*2 + 1][0]
                            for a in range(6)
                        ])
                        q_values = v + (advantages - pnp.mean(advantages))
                        
                        current_q = q_values[actions[i]]
                        
                        if dones[i]:
                            target_q = rewards[i]
                        else:
                            next_q_main = agent.get_q_values(next_states[i], use_target=False)
                            best_action = int(np.argmax(np.array(next_q_main)))
                            
                            next_q_target = agent.get_q_values(next_states[i], use_target=True)
                            target_q = rewards[i] + gamma * next_q_target[best_action]
                        
                        td_error = current_q - target_q
                        
                        weight = weights[i] if i < len(weights) else 1.0
                        if pnp.abs(td_error) <= 1.0:
                            loss = 0.5 * (td_error ** 2) * weight
                        else:
                            loss = (pnp.abs(td_error) - 0.5) * weight
                        
                        total_loss += loss
                    
                    return total_loss / len(states)
                
                # Update parameters
                params = agent.parameters()
                params = optimizer.step(loss_fn, *params)
                
                agent.embeddings = params[0]
                agent.w_proj = params[1]
                agent.b_proj = params[2]
                agent.q_weights = params[3]
                agent.v_w = params[4]
                agent.v_b = params[5]
                
                for i in range(6):
                    agent.adv_heads[i] = (params[6 + i*2], params[6 + i*2 + 1])
                
                # Compute TD errors for priority update
                td_errors = []
                for i in range(len(states)):
                    q_values = agent.get_q_values(states[i], use_target=False)
                    current_q = q_values[actions[i]]
                    
                    if dones[i]:
                        target_q = rewards[i]
                    else:
                        next_q_main = agent.get_q_values(next_states[i], use_target=False)
                        best_action = int(np.argmax(np.array(next_q_main)))
                        next_q_target = agent.get_q_values(next_states[i], use_target=True)
                        target_q = rewards[i] + gamma * next_q_target[best_action]
                    
                    td_errors.append(float(current_q - target_q))
                
                replay_buffer.update_priorities(indices, td_errors)
                agent.soft_update_target(tau=tau)
        
        # Decay epsilon
        if total_episodes >= warmup_episodes:
            epsilon = max(epsilon_end, epsilon * epsilon_decay)
        
        # Logging
        avg_reward = np.mean(epoch_rewards)
        max_reward = np.max(epoch_rewards)
        avg_steps = np.mean(epoch_steps)
        success_rate = sum(1 for r in epoch_rewards if r > 0) / len(epoch_rewards)
        
        if avg_reward > best_reward:
            best_reward = avg_reward
        
        logger.log_epoch(epoch + 1, avg_reward, max_reward, avg_steps, success_rate, epsilon)
        
        status = "[WARMUP]" if total_episodes < warmup_episodes else "[TRAIN]"
        
        print(f"Epoch {epoch + 1:3d} {status} | "
              f"Avg: {avg_reward:6.2f} | "
              f"Max: {max_reward:6.1f} | "
              f"Steps: {avg_steps:4.1f} | "
              f"Success: {success_rate:4.0%} | "
              f"Best: {best_reward:6.2f} | "
              f"ε: {epsilon:.3f}")
        
        # Early stopping
        if success_rate >= 0.95 and avg_reward > 8.0:
            print(f"\n🎉 Solved at epoch {epoch + 1}!")
            break
    
    env.close()
    
    # Save CSV
    logger.save_csv()
    
    # Testing
    print("\n" + "="*70)
    print("Testing Phase")
    print("="*70)
    
    test_env = gym.make('Taxi-v3')
    test_rewards = []
    test_steps = []
    successes = 0
    
    for test_ep in range(50):
        state, _ = test_env.reset()
        total_reward = 0
        done = False
        steps = 0
        
        while not done and steps < 100:
            q_values = agent.get_q_values(state, use_target=False)
            action = int(np.argmax(np.array(q_values)))
            
            state, reward, terminated, truncated, _ = test_env.step(action)
            total_reward += reward
            done = terminated or truncated
            steps += 1
        
        test_rewards.append(total_reward)
        test_steps.append(steps)
        if total_reward > 0:
            successes += 1
        
        if (test_ep + 1) % 10 == 0:
            print(f"Test {test_ep + 1:2d}/50 | Recent avg: {np.mean(test_rewards[-10:]):6.2f}")
    
    test_avg = np.mean(test_rewards)
    test_std = np.std(test_rewards)
    test_median = np.median(test_rewards)
    test_max = np.max(test_rewards)
    test_min = np.min(test_rewards)
    
    print("="*70)
    print(f"Test Results (50 episodes):")
    print(f"  Average reward:  {test_avg:6.2f} ± {test_std:.2f}")
    print(f"  Median reward:   {test_median:6.2f}")
    print(f"  Average steps:   {np.mean(test_steps):6.1f}")
    print(f"  Success rate:    {successes}/50 ({successes/50*100:.0f}%)")
    print(f"  Best reward:     {test_max:6.1f}")
    print(f"  Worst reward:    {test_min:6.1f}")
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
            'n_states': 500,
            'n_qubits': 4,
            'n_actions': 6,
            'n_layers': 2,
            'learning_rate': 5e-4,
            'gamma': gamma,
            'tau': tau,
            'batch_size': batch_size,
            'epochs': epochs,
            'episodes_per_epoch': episodes_per_epoch
        },
        'parameters': {
            'embeddings': agent.embeddings,
            'w_proj': agent.w_proj,
            'b_proj': agent.b_proj,
            'q_weights': agent.q_weights,
            'v_w': agent.v_w,
            'v_b': agent.v_b,
            'adv_heads': agent.adv_heads,
            'target_embeddings': agent.target_embeddings,
            'target_w_proj': agent.target_w_proj,
            'target_b_proj': agent.target_b_proj,
            'target_q_weights': agent.target_q_weights,
            'target_v_w': agent.target_v_w,
            'target_v_b': agent.target_v_b,
            'target_adv_heads': agent.target_adv_heads,
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