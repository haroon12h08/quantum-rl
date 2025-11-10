"""
Data Re-uploading Quantum Deep Q-Learning (Optimized)
Based on: Coelho et al. "VQC-Based Reinforcement Learning with Data Re-uploading" (2024)
Includes: CSV logging, plotting, and model checkpointing

Key features from paper:
- Data re-uploading: encode state multiple times in circuit
- Trainable input/output scaling
- Moving targets in DQN help avoid barren plateaus
- Works even with single qubit for CartPole
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


class DataReuploadingQNetwork:
    """
    Quantum Q-Network with data re-uploading
    Paper: "One qubit as a universal approximant" (Perez-Salinas et al. 2021)
    """
    def __init__(self, state_dim=4, n_actions=2, n_qubits=1, n_layers=5):
        self.state_dim = state_dim
        self.n_actions = n_actions
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        
        # Trainable input scaling (CRITICAL for performance)
        self.input_scaling = pnp.ones(state_dim, requires_grad=True)
        
        # Quantum device
        self.dev = qml.device('default.qubit', wires=n_qubits)
        self.circuit = self._create_circuit()
        
        # Quantum weights: (n_layers, n_qubits, 2)
        self.q_weights = pnp.random.uniform(
            0, 2*np.pi,
            size=(n_layers, n_qubits, 2),
            requires_grad=True
        )
        
        # Trainable output scaling (CRITICAL per paper)
        self.output_weights = pnp.random.randn(n_actions, n_qubits) * 0.1
        self.output_weights.requires_grad = True
        
        # Output bias
        self.output_bias = pnp.zeros(n_actions)
        self.output_bias.requires_grad = True
        
        # Target network for stable training
        self.target_input_scaling = pnp.copy(self.input_scaling)
        self.target_q_weights = pnp.copy(self.q_weights)
        self.target_output_weights = pnp.copy(self.output_weights)
        self.target_output_bias = pnp.copy(self.output_bias)
    
    def _create_circuit(self):
        """Create quantum circuit as a class method (pickle-friendly)"""
        @qml.qnode(self.dev, interface='autograd')
        def q_circuit(inputs, weights):
            """
            Data re-uploading circuit:
            Alternates between data encoding and variational layers
            This is the KEY insight from the paper
            """
            for layer in range(self.n_layers):
                # Data re-uploading: encode state again at each layer
                for i in range(self.n_qubits):
                    # Encode all state features into each qubit
                    for j in range(self.state_dim):
                        qml.RY(inputs[j], wires=i)
                
                # Variational layer
                for i in range(self.n_qubits):
                    qml.RY(weights[layer, i, 0], wires=i)
                    qml.RZ(weights[layer, i, 1], wires=i)
                
                # Entanglement (if more than 1 qubit)
                if self.n_qubits > 1:
                    for i in range(self.n_qubits - 1):
                        qml.CNOT(wires=[i, i + 1])
            
            # Measure all qubits
            return [qml.expval(qml.PauliZ(i)) for i in range(self.n_qubits)]
        
        return q_circuit
    
    def get_q_values(self, state, use_target=False):
        """Get Q-values for all actions"""
        # Apply input scaling
        if use_target:
            scaled_state = state * self.target_input_scaling
            q_features = pnp.array(self.circuit(scaled_state, self.target_q_weights))
            q_values = pnp.dot(self.target_output_weights, q_features) + self.target_output_bias
        else:
            scaled_state = state * self.input_scaling
            q_features = pnp.array(self.circuit(scaled_state, self.q_weights))
            q_values = pnp.dot(self.output_weights, q_features) + self.output_bias
        
        return q_values
    
    def select_action(self, state, epsilon):
        """Epsilon-greedy action selection"""
        if np.random.random() < epsilon:
            return np.random.randint(0, self.n_actions)
        else:
            q_values = self.get_q_values(state, use_target=False)
            return int(np.argmax(np.array(q_values)))
    
    def update_target_network(self):
        """Hard update of target network"""
        self.target_input_scaling = pnp.copy(self.input_scaling)
        self.target_q_weights = pnp.copy(self.q_weights)
        self.target_output_weights = pnp.copy(self.output_weights)
        self.target_output_bias = pnp.copy(self.output_bias)
    
    def parameters(self):
        """All trainable parameters"""
        return [self.input_scaling, self.q_weights, 
                self.output_weights, self.output_bias]


class ReplayBuffer:
    def __init__(self, capacity=10000):
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
        
    def log_episode(self, episode, reward, steps, epsilon, avg_10, avg_100=None):
        entry = {
            'episode': episode,
            'reward': reward,
            'steps': steps,
            'epsilon': epsilon,
            'avg_10': avg_10
        }
        if avg_100 is not None:
            entry['avg_100'] = avg_100
        self.data.append(entry)
    
    def save_csv(self):
        if not self.data:
            return None
        
        fieldnames = ['episode', 'reward', 'steps', 'epsilon', 'avg_10']
        if 'avg_100' in self.data[0]:
            fieldnames.append('avg_100')
        
        with open(self.csv_filename, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(self.data)
        print(f"\n✓ Training log saved: {self.csv_filename}")
        return self.csv_filename
    
    def plot_results(self, test_results=None):
        """Create training and test plots"""
        episodes = [d['episode'] for d in self.data]
        rewards = [d['reward'] for d in self.data]
        avg_10 = [d['avg_10'] for d in self.data]
        steps = [d['steps'] for d in self.data]
        epsilon = [d['epsilon'] for d in self.data]
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(f'{self.model_name} - Training Results', fontsize=16)
        
        # Rewards
        ax1 = axes[0, 0]
        ax1.plot(episodes, rewards, alpha=0.3, color='blue', label='Episode Reward')
        ax1.plot(episodes, avg_10, linewidth=2, color='red', label='10-episode Avg')
        if 'avg_100' in self.data[0]:
            avg_100 = [d['avg_100'] for d in self.data]
            ax1.plot(episodes, avg_100, linewidth=2, color='green', label='100-episode Avg')
        ax1.axhline(y=475, color='orange', linestyle='--', label='Solved (475)')
        ax1.set_xlabel('Episode')
        ax1.set_ylabel('Reward')
        ax1.set_title('Training Rewards')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Steps per episode
        ax2 = axes[0, 1]
        ax2.plot(episodes, steps, linewidth=1, color='purple', alpha=0.6)
        ax2.set_xlabel('Episode')
        ax2.set_ylabel('Steps')
        ax2.set_title('Episode Length')
        ax2.grid(True, alpha=0.3)
        
        # Epsilon decay
        ax3 = axes[1, 0]
        ax3.plot(episodes, epsilon, linewidth=2, color='orange')
        ax3.set_xlabel('Episode')
        ax3.set_ylabel('Epsilon')
        ax3.set_title('Exploration Rate (ε)')
        ax3.grid(True, alpha=0.3)
        
        # Test results
        ax4 = axes[1, 1]
        if test_results:
            test_rewards = test_results['rewards']
            ax4.hist(test_rewards, bins=15, alpha=0.7, color='green', edgecolor='black')
            ax4.axvline(x=np.mean(test_rewards), color='red', linestyle='--', linewidth=2,
                       label=f"Mean: {np.mean(test_rewards):.1f}")
            ax4.axvline(x=475, color='orange', linestyle='--', linewidth=2,
                       label='Solved (475)')
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
    model_name = "DataReuploadingDQN_CartPole"
    
    print("="*70)
    print(f"Training: {model_name}")
    print("Data Re-uploading Quantum DQN for CartPole-v1")
    print("="*70)
    
    # CartPole environment
    env = gym.make('CartPole-v1')
    
    # Paper's architecture: even 1 qubit works!
    agent = DataReuploadingQNetwork(
        state_dim=4,
        n_actions=2,
        n_qubits=1,  # Paper shows 1 qubit is enough for CartPole
        n_layers=5   # Paper uses 5 layers
    )
    
    replay_buffer = ReplayBuffer(capacity=10000)
    logger = TrainingLogger(model_name)
    
    # Optimized hyperparameters
    optimizer = qml.AdamOptimizer(stepsize=0.01)
    
    gamma = 0.99
    epsilon_start = 1.0
    epsilon_end = 0.01
    epsilon_decay = 0.995
    target_update_freq = 10
    batch_size = 32
    
    max_episodes = 20  # Reduced from 500 for faster training
    min_buffer_size = batch_size * 2  # Start training sooner
    
    epsilon = epsilon_start
    episode_rewards = []
    best_avg_reward = -float('inf')
    
    print(f"\nArchitecture:")
    print(f"  Qubits: {agent.n_qubits}")
    print(f"  Layers: {agent.n_layers}")
    print(f"  Data re-uploads per forward pass: {agent.n_layers}x")
    print(f"\nHyperparameters:")
    print(f"  Max Episodes: {max_episodes}")
    print(f"  Learning Rate: 0.01")
    print(f"  Batch Size: {batch_size}")
    print(f"  Target Update Freq: {target_update_freq}")
    print(f"  Epsilon: {epsilon_start} → {epsilon_end} (decay: {epsilon_decay})")
    print("="*70)
    
    for episode in range(max_episodes):
        state, _ = env.reset()
        total_reward = 0
        done = False
        steps = 0
        
        while not done and steps < 500:
            action = agent.select_action(state, epsilon)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            replay_buffer.push(state, action, reward, next_state, done)
            
            total_reward += reward
            state = next_state
            steps += 1
            
            # Train if enough samples
            if len(replay_buffer) >= min_buffer_size:
                states, actions, rewards, next_states, dones = replay_buffer.sample(batch_size)
                
                # DQN loss function
                def loss_fn(input_scaling, q_weights, output_weights, output_bias):
                    total_loss = 0
                    
                    for i in range(len(states)):
                        # Current Q-value
                        scaled_state = states[i] * input_scaling
                        q_features = pnp.array(agent.circuit(scaled_state, q_weights))
                        q_values = pnp.dot(output_weights, q_features) + output_bias
                        current_q = q_values[actions[i]]
                        
                        # Target Q-value
                        if dones[i]:
                            target_q = rewards[i]
                        else:
                            next_q_values = agent.get_q_values(next_states[i], use_target=True)
                            target_q = rewards[i] + gamma * pnp.max(next_q_values)
                        
                        # MSE loss
                        loss = (current_q - target_q) ** 2
                        total_loss += loss
                    
                    return total_loss / len(states)
                
                # Update parameters
                params = agent.parameters()
                params = optimizer.step(loss_fn, *params)
                
                agent.input_scaling = params[0]
                agent.q_weights = params[1]
                agent.output_weights = params[2]
                agent.output_bias = params[3]
        
        episode_rewards.append(total_reward)
        
        # Decay epsilon
        epsilon = max(epsilon_end, epsilon * epsilon_decay)
        
        # Update target network
        if (episode + 1) % target_update_freq == 0:
            agent.update_target_network()
        
        # Calculate averages
        avg_10 = np.mean(episode_rewards[-10:]) if len(episode_rewards) >= 10 else np.mean(episode_rewards)
        avg_100 = np.mean(episode_rewards[-100:]) if len(episode_rewards) >= 100 else None
        
        if avg_10 > best_avg_reward:
            best_avg_reward = avg_10
        
        # Log episode
        logger.log_episode(episode + 1, total_reward, steps, epsilon, avg_10, avg_100)
        
        # Print progress
        if (episode + 1) % 10 == 0:
            status = f"Episode {episode + 1:3d} | "
            status += f"Reward: {total_reward:3.0f} | "
            status += f"Avg(10): {avg_10:6.1f} | "
            if avg_100 is not None:
                status += f"Avg(100): {avg_100:6.1f} | "
            status += f"Best: {best_avg_reward:6.1f} | "
            status += f"ε: {epsilon:.3f}"
            print(status)
        
        # Success criteria (CartPole-v1 is solved at 475)
        if avg_100 is not None and avg_100 >= 475:
            print(f"\n🎉 Solved at episode {episode + 1}!")
            print(f"Average reward over last 100 episodes: {avg_100:.1f}")
            break
    
    env.close()
    
    # Save training CSV
    logger.save_csv()
    
    # Testing
    print("\n" + "="*70)
    print("Testing Phase")
    print("="*70)
    
    test_env = gym.make('CartPole-v1')
    test_rewards = []
    test_steps = []
    
    for test_ep in range(30):
        state, _ = test_env.reset()
        total_reward = 0
        done = False
        steps = 0
        
        while not done and steps < 500:
            q_values = agent.get_q_values(state, use_target=False)
            action = int(np.argmax(np.array(q_values)))
            
            state, reward, terminated, truncated, _ = test_env.step(action)
            total_reward += reward
            done = terminated or truncated
            steps += 1
        
        test_rewards.append(total_reward)
        test_steps.append(steps)
        
        if (test_ep + 1) % 10 == 0:
            print(f"Test {test_ep + 1:2d}/30 | Recent avg: {np.mean(test_rewards[-10:]):6.1f}")
    
    test_avg = np.mean(test_rewards)
    test_std = np.std(test_rewards)
    test_median = np.median(test_rewards)
    test_max = np.max(test_rewards)
    test_min = np.min(test_rewards)
    solved_count = sum(1 for r in test_rewards if r >= 475)
    
    print("="*70)
    print(f"Test Results (30 episodes):")
    print(f"  Average:  {test_avg:6.1f} ± {test_std:.1f}")
    print(f"  Median:   {test_median:6.1f}")
    print(f"  Best:     {test_max:3.0f}")
    print(f"  Worst:    {test_min:3.0f}")
    print(f"  Avg Steps: {np.mean(test_steps):6.1f}")
    print(f"  Solved:   {solved_count}/30 ({solved_count/30*100:.0f}%)")
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
        writer.writerow(['solved_rate', f'{solved_count}/30', f'{solved_count/30*100:.1f}%'])
    print(f"✓ Test results saved: {test_csv}")
    
    # Create plots
    test_results = {'rewards': test_rewards}
    logger.plot_results(test_results)
    
    # Save model (pickle-friendly - no circuit)
    model_data = {
        'config': {
            'state_dim': agent.state_dim,
            'n_actions': agent.n_actions,
            'n_qubits': agent.n_qubits,
            'n_layers': agent.n_layers,
            'learning_rate': 0.01,
            'gamma': gamma,
            'batch_size': batch_size,
            'target_update_freq': target_update_freq
        },
        'parameters': {
            'input_scaling': agent.input_scaling,
            'q_weights': agent.q_weights,
            'output_weights': agent.output_weights,
            'output_bias': agent.output_bias,
            'target_input_scaling': agent.target_input_scaling,
            'target_q_weights': agent.target_q_weights,
            'target_output_weights': agent.target_output_weights,
            'target_output_bias': agent.target_output_bias
        },
        'training_info': {
            'total_episodes': len(episode_rewards),
            'final_avg_10': float(avg_10),
            'final_avg_100': float(avg_100) if avg_100 is not None else None,
            'best_avg_reward': float(best_avg_reward)
        },
        'test_results': {
            'avg': test_avg,
            'std': test_std,
            'median': test_median,
            'max': test_max,
            'min': test_min,
            'solved_rate': solved_count/30
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


def load_trained_model(filename):
    """Load a trained quantum model from pickle file"""
    with open(filename, 'rb') as f:
        model_data = pickle.load(f)
    
    # Recreate agent with saved architecture
    config = model_data['config']
    agent = DataReuploadingQNetwork(
        state_dim=config['state_dim'],
        n_actions=config['n_actions'],
        n_qubits=config['n_qubits'],
        n_layers=config['n_layers']
    )
    
    # Load trained parameters
    params = model_data['parameters']
    agent.input_scaling = pnp.array(params['input_scaling'], requires_grad=True)
    agent.q_weights = pnp.array(params['q_weights'], requires_grad=True)
    agent.output_weights = pnp.array(params['output_weights'], requires_grad=True)
    agent.output_bias = pnp.array(params['output_bias'], requires_grad=True)
    
    agent.target_input_scaling = pnp.array(params['target_input_scaling'])
    agent.target_q_weights = pnp.array(params['target_q_weights'])
    agent.target_output_weights = pnp.array(params['target_output_weights'])
    agent.target_output_bias = pnp.array(params['target_output_bias'])
    
    print(f"✓ Model loaded from: {filename}")
    print(f"  Training episodes: {model_data['training_info']['total_episodes']}")
    if model_data['training_info']['final_avg_100']:
        print(f"  Final avg (100): {model_data['training_info']['final_avg_100']:.1f}")
    print(f"  Test avg: {model_data['test_results']['avg']:.1f}")
    
    return agent


if __name__ == '__main__':
    main()
    
    # Example of loading and using the model:
    # agent = load_trained_model('DataReuploadingDQN_CartPole_model_TIMESTAMP.pkl')
    # # Then use agent.get_q_values(state) for inference