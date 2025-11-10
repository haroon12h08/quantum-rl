"""
Optimized Quantum DDPG for Fast Training (~7 minutes)
Includes: CSV logging, plotting, and model checkpointing

Key optimizations:
- Reduced epochs: 25 instead of 300
- Fewer episodes per epoch: 3 instead of 5
- Fewer training steps: 20 instead of 50
- Maintains architecture quality from original
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
import os


class QuantumActor:
    """Improved quantum actor with deeper circuit"""
    def __init__(self, state_dim=24, n_qubits=4, n_actions=4, n_layers=2):
        self.state_dim = state_dim
        self.n_qubits = n_qubits
        self.n_actions = n_actions
        self.n_layers = n_layers
        
        # Classical encoder with better initialization
        hidden = 16
        scale1 = np.sqrt(2.0 / state_dim)
        scale2 = np.sqrt(2.0 / hidden)
        
        self.w_enc1 = pnp.random.randn(state_dim, hidden) * scale1
        self.b_enc1 = pnp.zeros(hidden)
        self.w_enc2 = pnp.random.randn(hidden, n_qubits) * scale2
        self.b_enc2 = pnp.zeros(n_qubits)
        
        for p in [self.w_enc1, self.b_enc1, self.w_enc2, self.b_enc2]:
            p.requires_grad = True
        
        # Quantum circuit
        self.dev = qml.device('default.qubit', wires=n_qubits)
        
        @qml.qnode(self.dev, interface='autograd')
        def q_circuit(inputs, weights):
            # State encoding
            for i in range(n_qubits):
                qml.RY(inputs[i], wires=i)
            
            # Multiple variational layers
            for layer in range(n_layers):
                # Rotation layer
                for i in range(n_qubits):
                    qml.RY(weights[layer, i, 0], wires=i)
                    qml.RZ(weights[layer, i, 1], wires=i)
                
                # Entanglement layer
                for i in range(n_qubits - 1):
                    qml.CNOT(wires=[i, i + 1])
                # Circular entanglement
                qml.CNOT(wires=[n_qubits - 1, 0])
            
            return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]
        
        self.circuit = q_circuit
        # Better initialization for quantum weights
        self.q_weights = pnp.random.uniform(0, np.pi/2, size=(n_layers, n_qubits, 2), requires_grad=True)
        
        # Post-processing
        scale_out = np.sqrt(2.0 / n_qubits)
        self.w_out = pnp.random.randn(n_actions, n_qubits) * scale_out
        self.b_out = pnp.zeros(n_actions)
        
        for p in [self.w_out, self.b_out]:
            p.requires_grad = True
    
    def forward(self, state):
        """Forward pass with gradient clipping"""
        # Encode state
        h = pnp.tanh(pnp.dot(state, self.w_enc1) + self.b_enc1)
        encoded = pnp.tanh(pnp.dot(h, self.w_enc2) + self.b_enc2) * np.pi
        
        # Quantum processing
        q_out = pnp.array(self.circuit(encoded, self.q_weights))
        
        # Map to action space
        actions = pnp.tanh(pnp.dot(self.w_out, q_out) + self.b_out)
        return actions
    
    def parameters(self):
        return [self.w_enc1, self.b_enc1, self.w_enc2, self.b_enc2,
                self.q_weights, self.w_out, self.b_out]


class ClassicalCritic:
    """Improved critic with better initialization"""
    def __init__(self, state_dim=24, action_dim=4):
        h1 = 128
        h2 = 64
        
        scale1 = np.sqrt(2.0 / (state_dim + action_dim))
        scale2 = np.sqrt(2.0 / h1)
        scale3 = np.sqrt(2.0 / h2)
        
        self.w1 = pnp.random.randn(state_dim + action_dim, h1) * scale1
        self.b1 = pnp.zeros(h1)
        self.w2 = pnp.random.randn(h1, h2) * scale2
        self.b2 = pnp.zeros(h2)
        self.w3 = pnp.random.randn(h2, 1) * scale3
        self.b3 = pnp.zeros(1)
        
        for p in [self.w1, self.b1, self.w2, self.b2, self.w3, self.b3]:
            p.requires_grad = True
    
    def forward(self, state, action):
        x = pnp.concatenate([state, action])
        h1 = pnp.maximum(0, pnp.dot(x, self.w1) + self.b1)
        h2 = pnp.maximum(0, pnp.dot(h1, self.w2) + self.b2)
        q = pnp.dot(h2, self.w3) + self.b3
        return q[0]
    
    def parameters(self):
        return [self.w1, self.b1, self.w2, self.b2, self.w3, self.b3]


class ReplayBuffer:
    def __init__(self, capacity=50000):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (np.array(states), np.array(actions), np.array(rewards),
                np.array(next_states), np.array(dones))
    
    def __len__(self):
        return len(self.buffer)


class TrainingLogger:
    """Logs training metrics"""
    def __init__(self, model_name):
        self.model_name = model_name
        self.timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.csv_filename = f'{model_name}_training_{self.timestamp}.csv'
        self.data = []
        
    def log_epoch(self, epoch, avg_reward, max_reward, min_reward, std_reward, noise):
        self.data.append({
            'epoch': epoch,
            'avg_reward': avg_reward,
            'max_reward': max_reward,
            'min_reward': min_reward,
            'std_reward': std_reward,
            'exploration_noise': noise
        })
    
    def save_csv(self):
        with open(self.csv_filename, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=[
                'epoch', 'avg_reward', 'max_reward', 'min_reward', 'std_reward', 'exploration_noise'
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
        min_rewards = [d['min_reward'] for d in self.data]
        
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))
        fig.suptitle(f'{self.model_name} - Training Results', fontsize=16)
        
        # Training progress
        ax1 = axes[0]
        ax1.plot(epochs, avg_rewards, label='Average', linewidth=2, color='blue')
        ax1.plot(epochs, max_rewards, label='Max', linewidth=1, alpha=0.7, color='green')
        ax1.plot(epochs, min_rewards, label='Min', linewidth=1, alpha=0.7, color='red')
        ax1.fill_between(epochs, min_rewards, max_rewards, alpha=0.2)
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Reward')
        ax1.set_title('Training Progress')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Test results (if provided)
        ax2 = axes[1]
        if test_results:
            test_rewards = test_results['rewards']
            ax2.bar(range(len(test_rewards)), test_rewards, alpha=0.7, color='purple')
            ax2.axhline(y=np.mean(test_rewards), color='red', linestyle='--', 
                       label=f"Mean: {np.mean(test_rewards):.2f}")
            ax2.set_xlabel('Test Episode')
            ax2.set_ylabel('Reward')
            ax2.set_title(f"Test Performance ({len(test_rewards)} episodes)")
            ax2.legend()
            ax2.grid(True, alpha=0.3, axis='y')
        else:
            ax2.text(0.5, 0.5, 'Test results not available', 
                    ha='center', va='center', transform=ax2.transAxes)
            ax2.axis('off')
        
        plt.tight_layout()
        plot_filename = f'{self.model_name}_plots_{self.timestamp}.png'
        plt.savefig(plot_filename, dpi=150, bbox_inches='tight')
        print(f"✓ Plots saved: {plot_filename}")
        return plot_filename


def main():
    model_name = "ImprovedQuantumDDPG"
    
    print("="*70)
    print(f"Training: {model_name}")
    print("Optimized for ~7 minute runtime")
    print("="*70)
    
    env = gym.make('BipedalWalker-v3')
    
    # Improved architecture
    actor = QuantumActor(state_dim=24, n_qubits=4, n_actions=4, n_layers=2)
    critic = ClassicalCritic(state_dim=24, action_dim=4)
    
    target_actor = QuantumActor(state_dim=24, n_qubits=4, n_actions=4, n_layers=2)
    target_critic = ClassicalCritic(state_dim=24, action_dim=4)
    
    for i, p in enumerate(actor.parameters()):
        target_actor.parameters()[i] = pnp.copy(p)
    for i, p in enumerate(critic.parameters()):
        target_critic.parameters()[i] = pnp.copy(p)
    
    replay_buffer = ReplayBuffer(capacity=50000)
    logger = TrainingLogger(model_name)
    
    # Adjusted hyperparameters for speed
    actor_optimizer = qml.AdamOptimizer(stepsize=5e-4)
    critic_optimizer = qml.AdamOptimizer(stepsize=1e-3)
    
    gamma = 0.99
    tau = 0.005
    batch_size = 64
    warmup_episodes = 5  # Reduced from 10
    
    epochs = 25  # Reduced from 300
    episodes_per_epoch = 3  # Reduced from 5
    train_steps = 20  # Reduced from 50
    exploration_noise = 0.2
    
    best_reward = -float('inf')
    
    print(f"\nHyperparameters:")
    print(f"  Epochs: {epochs} | Episodes/Epoch: {episodes_per_epoch} | Train Steps: {train_steps}")
    print(f"  Actor LR: {5e-4} | Critic LR: {1e-3} | Batch Size: {batch_size}")
    print(f"  Warmup Episodes: {warmup_episodes}")
    print("="*70)
    
    total_episodes = 0
    
    for epoch in range(epochs):
        epoch_rewards = []
        
        # Collect experience
        for ep in range(episodes_per_epoch):
            state, _ = env.reset()
            total_reward = 0
            done = False
            steps = 0
            
            while not done and steps < 1600:
                # Random actions during warmup
                if total_episodes < warmup_episodes:
                    action = np.random.uniform(-1, 1, size=4)
                else:
                    action = np.array(actor.forward(state))
                    action = np.clip(action + np.random.normal(0, exploration_noise, size=4), -1, 1)
                
                next_state, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                
                replay_buffer.push(state, action, reward, next_state, done)
                
                total_reward += reward
                state = next_state
                steps += 1
            
            epoch_rewards.append(total_reward)
            total_episodes += 1
        
        # Train only after warmup
        if total_episodes >= warmup_episodes and len(replay_buffer) >= batch_size:
            for _ in range(train_steps):
                states, actions, rewards, next_states, dones = replay_buffer.sample(batch_size)
                
                # Update Critic
                def critic_loss(w1, b1, w2, b2, w3, b3):
                    total_loss = 0
                    for i in range(len(states)):
                        x = pnp.concatenate([states[i], actions[i]])
                        h1 = pnp.maximum(0, pnp.dot(x, w1) + b1)
                        h2 = pnp.maximum(0, pnp.dot(h1, w2) + b2)
                        current_q = pnp.dot(h2, w3) + b3
                        
                        if dones[i]:
                            target_q = rewards[i]
                        else:
                            next_action = target_actor.forward(next_states[i])
                            target_q_val = target_critic.forward(next_states[i], next_action)
                            target_q = rewards[i] + gamma * target_q_val
                        
                        total_loss += (current_q[0] - target_q) ** 2
                    
                    return total_loss / len(states)
                
                critic_params = critic.parameters()
                critic_params = critic_optimizer.step(critic_loss, *critic_params)
                for i, p in enumerate(critic_params):
                    critic.parameters()[i] = p
                
                # Update Actor
                def actor_loss(w_enc1, b_enc1, w_enc2, b_enc2, q_weights, w_out, b_out):
                    total_loss = 0
                    for i in range(len(states)):
                        h = pnp.tanh(pnp.dot(states[i], w_enc1) + b_enc1)
                        encoded = pnp.tanh(pnp.dot(h, w_enc2) + b_enc2) * np.pi
                        q_out = pnp.array(actor.circuit(encoded, q_weights))
                        action = pnp.tanh(pnp.dot(w_out, q_out) + b_out)
                        
                        q_val = critic.forward(states[i], action)
                        total_loss -= q_val
                    
                    return total_loss / len(states)
                
                actor_params = actor.parameters()
                actor_params = actor_optimizer.step(actor_loss, *actor_params)
                for i, p in enumerate(actor_params):
                    actor.parameters()[i] = p
                
                # Soft update targets
                for i in range(len(target_actor.parameters())):
                    target_actor.parameters()[i] = (1 - tau) * target_actor.parameters()[i] + tau * actor.parameters()[i]
                for i in range(len(target_critic.parameters())):
                    target_critic.parameters()[i] = (1 - tau) * target_critic.parameters()[i] + tau * critic.parameters()[i]
        
        # Logging
        avg_reward = np.mean(epoch_rewards)
        max_reward = np.max(epoch_rewards)
        min_reward = np.min(epoch_rewards)
        std_reward = np.std(epoch_rewards)
        
        logger.log_epoch(epoch + 1, avg_reward, max_reward, min_reward, std_reward, exploration_noise)
        
        if avg_reward > best_reward:
            best_reward = avg_reward
        
        status = "[WARMUP]" if total_episodes < warmup_episodes else "[TRAIN]"
        print(f"Epoch {epoch + 1:3d} {status} | Avg: {avg_reward:7.2f} | Max: {max_reward:7.2f} | Best: {best_reward:7.2f} | Noise: {exploration_noise:.3f}")
        
        # Decay exploration more slowly
        if total_episodes >= warmup_episodes:
            exploration_noise = max(0.05, exploration_noise * 0.998)
        
        if avg_reward > 250:
            print(f"\n🎉 Solved at epoch {epoch + 1}!")
            break
    
    env.close()
    
    # Save CSV
    logger.save_csv()
    
    # Test
    print("\n" + "="*70)
    print("Testing Phase")
    print("="*70)
    test_env = gym.make('BipedalWalker-v3')
    test_rewards = []
    
    for i in range(20):
        state, _ = test_env.reset()
        total_reward = 0
        done = False
        steps = 0
        
        while not done and steps < 1600:
            action = np.array(actor.forward(state))
            state, reward, terminated, truncated, _ = test_env.step(action)
            total_reward += reward
            done = terminated or truncated
            steps += 1
        
        test_rewards.append(total_reward)
        print(f"Test {i+1:2d}/20 | Reward: {total_reward:7.2f}")
    
    test_avg = np.mean(test_rewards)
    test_std = np.std(test_rewards)
    test_max = np.max(test_rewards)
    test_min = np.min(test_rewards)
    
    print("="*70)
    print(f"Test Results: {test_avg:.2f} ± {test_std:.2f} | Best: {test_max:.2f} | Worst: {test_min:.2f}")
    print("="*70)
    
    test_env.close()
    
    # Save test results to CSV
    test_csv = f'{model_name}_test_results_{logger.timestamp}.csv'
    with open(test_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['episode', 'reward'])
        for i, reward in enumerate(test_rewards, 1):
            writer.writerow([i, reward])
        writer.writerow(['average', test_avg])
        writer.writerow(['std', test_std])
        writer.writerow(['max', test_max])
        writer.writerow(['min', test_min])
    print(f"✓ Test results saved: {test_csv}")
    
    # Create plots
    test_results = {'rewards': test_rewards}
    logger.plot_results(test_results)
    
    # Save model
    model_data = {
        'actor': actor,
        'critic': critic,
        'target_actor': target_actor,
        'target_critic': target_critic,
        'replay_buffer': replay_buffer,
        'config': {
            'state_dim': 24,
            'n_qubits': 4,
            'n_actions': 4,
            'n_layers': 2,
            'actor_lr': 5e-4,
            'critic_lr': 1e-3,
            'gamma': gamma,
            'tau': tau,
            'batch_size': batch_size,
            'epochs': epochs,
            'episodes_per_epoch': episodes_per_epoch
        },
        'test_results': {
            'avg': test_avg,
            'std': test_std,
            'max': test_max,
            'min': test_min
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