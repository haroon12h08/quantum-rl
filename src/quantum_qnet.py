"""
Optimized Quantum DQN for Taxi-v3
Uses data re-uploading and state encoding
"""

import gymnasium as gym
import numpy as np
import pennylane as qml
from pennylane import numpy as pnp
from collections import deque
import random
import csv
from datetime import datetime


class QuantumDQN:
    """Quantum Deep Q-Network with data re-uploading"""
    def __init__(self, n_qubits=5, n_actions=6, n_layers=3):
        self.n_qubits = n_qubits
        self.n_actions = n_actions
        self.n_layers = n_layers
        
        self.dev = qml.device('default.qubit', wires=n_qubits)
        
        @qml.qnode(self.dev, interface='autograd')
        def circuit(inputs, weights, scaling):
            """Data re-uploading circuit"""
            for layer in range(n_layers):
                # Re-upload data at each layer
                for i in range(n_qubits):
                    qml.RY(scaling[layer, i] * inputs[i], wires=i)
                
                # Variational layer
                for i in range(n_qubits):
                    qml.RY(weights[layer, i, 0], wires=i)
                    qml.RZ(weights[layer, i, 1], wires=i)
                
                # Entanglement
                for i in range(n_qubits - 1):
                    qml.CNOT(wires=[i, i + 1])
                if n_qubits > 2:
                    qml.CNOT(wires=[n_qubits - 1, 0])
            
            return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]
        
        self.circuit = circuit
        
        # Trainable parameters
        self.weights = pnp.random.uniform(0, 2*np.pi, size=(n_layers, n_qubits, 2), requires_grad=True)
        self.scaling = pnp.ones((n_layers, n_qubits), requires_grad=True)
        self.output_weights = pnp.random.randn(n_actions, n_qubits) * 0.1
        self.output_weights.requires_grad = True
        
        # Target network
        self.target_weights = pnp.copy(self.weights)
        self.target_scaling = pnp.copy(self.scaling)
        self.target_output_weights = pnp.copy(self.output_weights)
    
    def get_q_values(self, state, use_target=False):
        """Get Q-values for all actions"""
        if use_target:
            expectations = pnp.array(self.circuit(state, self.target_weights, self.target_scaling))
            q_values = pnp.dot(self.target_output_weights, expectations)
        else:
            expectations = pnp.array(self.circuit(state, self.weights, self.scaling))
            q_values = pnp.dot(self.output_weights, expectations)
        return q_values
    
    def select_action(self, state, epsilon):
        """Epsilon-greedy action selection"""
        if np.random.random() < epsilon:
            return np.random.randint(0, self.n_actions)
        else:
            q_values = self.get_q_values(state, use_target=False)
            return int(np.argmax(np.array(q_values)))
    
    def update_target_network(self):
        """Copy to target network"""
        self.target_weights = pnp.copy(self.weights)
        self.target_scaling = pnp.copy(self.scaling)
        self.target_output_weights = pnp.copy(self.output_weights)
    
    def parameters(self):
        return [self.weights, self.scaling, self.output_weights]


class ReplayBuffer:
    def __init__(self, capacity=2000):
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


def encode_taxi_state(state_int):
    """Encode Taxi state to continuous vector"""
    # Decode Taxi state components
    taxi_row = (state_int // 100) % 5
    taxi_col = (state_int // 20) % 5
    pass_idx = (state_int // 4) % 5
    dest_idx = state_int % 4
    
    # Normalize to [-π, π]
    encoded = np.array([
        taxi_row / 4.0 * np.pi,
        taxi_col / 4.0 * np.pi,
        pass_idx / 4.0 * np.pi,
        dest_idx / 3.0 * np.pi,
        (taxi_row + taxi_col) / 8.0 * np.pi
    ])
    
    return encoded


def main():
    env = gym.make('Taxi-v3')
    
    agent = QuantumDQN(n_qubits=5, n_actions=6, n_layers=3)
    replay_buffer = ReplayBuffer(capacity=2000)
    optimizer = qml.AdamOptimizer(stepsize=0.01)
    
    epochs = 10
    episodes_per_epoch = 10
    batch_size = 32
    gamma = 0.95
    epsilon_start = 1.0
    epsilon_end = 0.05
    epsilon_decay = 0.995
    target_update_freq = 5
    
    epsilon = epsilon_start
    results = []
    best_reward = -float('inf')
    
    print("Training Quantum DQN on Taxi-v3")
    print("=" * 60)
    
    for epoch in range(epochs):
        epoch_rewards = []
        epoch_steps = []
        
        for ep in range(episodes_per_epoch):
            state, _ = env.reset()
            done = False
            total_reward = 0
            steps = 0
            
            while not done and steps < 100:
                encoded_state = encode_taxi_state(state)
                action = agent.select_action(encoded_state, epsilon)
                
                next_state, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                
                encoded_next_state = encode_taxi_state(next_state)
                replay_buffer.push(encoded_state, action, reward, encoded_next_state, done)
                
                total_reward += reward
                state = next_state
                steps += 1
            
            epoch_rewards.append(total_reward)
            epoch_steps.append(steps)
        
        # Training
        if len(replay_buffer) >= batch_size:
            for _ in range(5):  # Multiple updates
                states, actions, rewards, next_states, dones = replay_buffer.sample(batch_size)
                
                def loss_fn(weights, scaling, output_weights):
                    total_loss = 0
                    
                    for i in range(len(states)):
                        # Current Q
                        expectations = pnp.array(agent.circuit(states[i], weights, scaling))
                        q_values = pnp.dot(output_weights, expectations)
                        current_q = q_values[actions[i]]
                        
                        # Target Q
                        if dones[i]:
                            target_q = rewards[i]
                        else:
                            next_q = agent.get_q_values(next_states[i], use_target=True)
                            target_q = rewards[i] + gamma * pnp.max(next_q)
                        
                        # MSE loss
                        total_loss += (current_q - target_q) ** 2
                    
                    return total_loss / len(states)
                
                params = agent.parameters()
                params = optimizer.step(loss_fn, *params)
                agent.weights, agent.scaling, agent.output_weights = params
        
        # Decay epsilon
        epsilon = max(epsilon_end, epsilon * epsilon_decay)
        
        # Update target
        if (epoch + 1) % target_update_freq == 0:
            agent.update_target_network()
        
        # Logging
        avg_reward = np.mean(epoch_rewards)
        max_reward = np.max(epoch_rewards)
        avg_steps = np.mean(epoch_steps)
        
        if avg_reward > best_reward:
            best_reward = avg_reward
        
        success_rate = sum(1 for r in epoch_rewards if r > 0) / len(epoch_rewards)
        
        results.append({
            'epoch': epoch + 1,
            'avg_reward': avg_reward,
            'max_reward': max_reward,
            'avg_steps': avg_steps,
            'best_reward': best_reward,
            'success_rate': success_rate,
            'epsilon': epsilon
        })
        
        print(f"Epoch {epoch + 1:3d} | Avg: {avg_reward:6.1f} | Max: {max_reward:6.1f} | "
              f"Steps: {avg_steps:5.1f} | Best: {best_reward:6.1f} | "
              f"Success: {success_rate:.1%} | ε: {epsilon:.3f}")
        
        if avg_reward > 7:
            print(f"\nGood performance at epoch {epoch + 1}!")
            break
    
    env.close()
    
    # Save results to CSV
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_filename = f'results_taxi_{timestamp}.csv'
    
    with open(csv_filename, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['epoch', 'avg_reward', 'max_reward', 'avg_steps', 
                                                'best_reward', 'success_rate', 'epsilon'])
        writer.writeheader()
        writer.writerows(results)
    
    print(f"\n✓ Results saved to: {csv_filename}")
    
    # Save model
    model_data = {
        'weights': np.array(agent.weights),
        'scaling': np.array(agent.scaling),
        'output_weights': np.array(agent.output_weights),
        'architecture': {
            'n_qubits': agent.n_qubits,
            'n_actions': agent.n_actions,
            'n_layers': agent.n_layers
        },
        'final_performance': {
            'best_reward': float(best_reward),
            'total_epochs': epochs
        }
    }
    
    model_filename = f'model_taxi_{timestamp}.npy'
    np.save(model_filename, model_data)
    print(f"✓ Model saved to: {model_filename}")
    
    return results


if __name__ == '__main__':
    main()