"""
Data Re-uploading Quantum Deep Q-Learning
Based on: Coelho et al. "VQC-Based Reinforcement Learning with Data Re-uploading" (2024)
Successfully solved CartPole-v0 and Acrobot-v1

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
        
        @qml.qnode(self.dev, interface='autograd')
        def q_circuit(inputs, weights):
            """
            Data re-uploading circuit:
            Alternates between data encoding and variational layers
            This is the KEY insight from the paper
            """
            for layer in range(n_layers):
                # Data re-uploading: encode state again at each layer
                for i in range(n_qubits):
                    # Encode all state features into each qubit
                    for j in range(state_dim):
                        qml.RY(inputs[j], wires=i)
                
                # Variational layer
                for i in range(n_qubits):
                    qml.RY(weights[layer, i, 0], wires=i)
                    qml.RZ(weights[layer, i, 1], wires=i)
                
                # Entanglement (if more than 1 qubit)
                if n_qubits > 1:
                    for i in range(n_qubits - 1):
                        qml.CNOT(wires=[i, i + 1])
            
            # Measure all qubits
            return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]
        
        self.circuit = q_circuit
        
        # Quantum weights: (n_layers, n_qubits, 2)
        self.q_weights = pnp.random.uniform(
            0, 2*np.pi,
            size=(n_layers, n_qubits, 2),
            requires_grad=True
        )
        
        # Trainable output scaling (CRITICAL per paper)
        # One weight per action-qubit pair
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
        params = [self.input_scaling, self.q_weights, 
                  self.output_weights, self.output_bias]
        return params


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


def main():
    # Start with CartPole (paper's benchmark)
    env = gym.make('CartPole-v1')
    
    # Paper's architecture: even 1 qubit works!
    agent = DataReuploadingQNetwork(
        state_dim=4,
        n_actions=2,
        n_qubits=1,  # Paper shows 1 qubit is enough for CartPole
        n_layers=5   # Paper uses 5 layers
    )
    
    replay_buffer = ReplayBuffer(capacity=10000)
    
    # Paper's hyperparameters
    optimizer = qml.AdamOptimizer(stepsize=0.01)
    
    gamma = 0.99
    epsilon_start = 1.0
    epsilon_end = 0.01
    epsilon_decay = 0.995
    target_update_freq = 10
    batch_size = 32
    
    max_episodes = 500
    epsilon = epsilon_start
    
    episode_rewards = []
    best_avg_reward = -float('inf')
    
    print("Training Data Re-uploading Quantum DQN on CartPole-v1")
    print("=" * 60)
    print(f"Architecture: {agent.n_qubits} qubit(s), {agent.n_layers} layers")
    print(f"Data re-uploading: {agent.n_layers}x per forward pass")
    print("=" * 60)
    
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
            if len(replay_buffer) >= batch_size:
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
        
        # Logging
        if (episode + 1) % 10 == 0:
            avg_reward = np.mean(episode_rewards[-10:])
            if avg_reward > best_avg_reward:
                best_avg_reward = avg_reward
            
            print(f"Episode {episode + 1:3d} | "
                  f"Avg (last 10): {avg_reward:6.1f} | "
                  f"Last: {total_reward:3.0f} | "
                  f"Best Avg: {best_avg_reward:6.1f} | "
                  f"ε: {epsilon:.3f}")
        
        # Success criteria (CartPole-v1 is solved at 475)
        if len(episode_rewards) >= 100:
            avg_100 = np.mean(episode_rewards[-100:])
            if avg_100 >= 475:
                print(f"\n🎉 Solved at episode {episode + 1}!")
                print(f"Average reward over last 100 episodes: {avg_100:.1f}")
                break
    
    env.close()
    
    # Test the trained agent
    print("\n" + "=" * 60)
    print("Testing trained agent (no exploration)")
    print("=" * 60)
    
    test_env = gym.make('CartPole-v1')
    test_rewards = []
    
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
    
    print(f"\nTest Results (30 episodes):")
    print(f"  Average:  {np.mean(test_rewards):.1f} ± {np.std(test_rewards):.1f}")
    print(f"  Median:   {np.median(test_rewards):.1f}")
    print(f"  Best:     {np.max(test_rewards):.0f}")
    print(f"  Worst:    {np.min(test_rewards):.0f}")
    
    test_env.close()
    
    # Save the trained model
    print("\n" + "=" * 60)
    print("Saving trained model...")
    print("=" * 60)
    
    model_data = {
        'input_scaling': np.array(agent.input_scaling),
        'q_weights': np.array(agent.q_weights),
        'output_weights': np.array(agent.output_weights),
        'output_bias': np.array(agent.output_bias),
        'architecture': {
            'state_dim': agent.state_dim,
            'n_actions': agent.n_actions,
            'n_qubits': agent.n_qubits,
            'n_layers': agent.n_layers
        },
        'training_info': {
            'total_episodes': len(episode_rewards),
            'final_avg_reward': float(np.mean(episode_rewards[-100:])),
            'best_avg_reward': float(best_avg_reward)
        }
    }
    
    np.save('quantum_cartpole_model.npy', model_data)
    print("✓ Model saved to: quantum_cartpole_model.npy")
    
    # Training summary
    print("\n" + "=" * 60)
    print("Training Summary:")
    print(f"  Total episodes: {len(episode_rewards)}")
    print(f"  Final average (last 100): {np.mean(episode_rewards[-100:]):.1f}")
    print(f"  Best average: {best_avg_reward:.1f}")
    print(f"  Test average: {np.mean(test_rewards):.1f}")
    print("=" * 60)


def load_trained_model(filename='quantum_cartpole_model.npy'):
    """Load a trained quantum model"""
    model_data = np.load(filename, allow_pickle=True).item()
    
    # Recreate agent with saved architecture
    arch = model_data['architecture']
    agent = DataReuploadingQNetwork(
        state_dim=arch['state_dim'],
        n_actions=arch['n_actions'],
        n_qubits=arch['n_qubits'],
        n_layers=arch['n_layers']
    )
    
    # Load trained parameters
    agent.input_scaling = pnp.array(model_data['input_scaling'], requires_grad=True)
    agent.q_weights = pnp.array(model_data['q_weights'], requires_grad=True)
    agent.output_weights = pnp.array(model_data['output_weights'], requires_grad=True)
    agent.output_bias = pnp.array(model_data['output_bias'], requires_grad=True)
    
    print(f"✓ Model loaded from: {filename}")
    print(f"  Training episodes: {model_data['training_info']['total_episodes']}")
    print(f"  Final performance: {model_data['training_info']['final_avg_reward']:.1f}")
    
    return agent


if __name__ == '__main__':
    main()
    
    # Example of loading and using the model:
    # agent = load_trained_model('quantum_cartpole_model.npy')
    # # Then use agent.get_q_values(state) for inference