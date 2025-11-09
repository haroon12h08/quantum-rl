import gymnasium as gym
import numpy as np
import pennylane as qml
from pennylane import numpy as pnp
from collections import deque
import random


class QuantumQNetwork:
    """
    Simplified Quantum Q-Network for Taxi-v3
    Uses direct state encoding without complex transformations
    """
    def __init__(self, n_qubits=5, n_actions=6, n_layers=2):
        self.n_qubits = n_qubits
        self.n_actions = n_actions
        self.n_layers = n_layers
        
        self.dev = qml.device('default.qubit', wires=n_qubits)
        
        @qml.qnode(self.dev, interface='autograd')
        def q_circuit(inputs, weights):
            """Simplified quantum circuit"""
            for layer in range(self.n_layers):
                # State encoding
                for i in range(self.n_qubits):
                    qml.RY(inputs[i], wires=i)
                
                # Variational layer
                for i in range(self.n_qubits):
                    qml.RY(weights[layer, i, 0], wires=i)
                    qml.RZ(weights[layer, i, 1], wires=i)
                
                # Entanglement
                for i in range(self.n_qubits - 1):
                    qml.CNOT(wires=[i, i + 1])
                if self.n_qubits > 2:
                    qml.CNOT(wires=[self.n_qubits - 1, 0])
            
            return [qml.expval(qml.PauliZ(i)) for i in range(self.n_qubits)]
        
        self.q_circuit = q_circuit
        
        # Network parameters
        self.weights = pnp.random.uniform(
            -np.pi, np.pi,
            size=(n_layers, n_qubits, 2),
            requires_grad=True
        )
        self.output_weights = pnp.random.uniform(
            -1, 1,
            size=(n_actions, n_qubits),
            requires_grad=True
        )
        
        # Target network
        self.target_weights = pnp.copy(self.weights)
        self.target_output_weights = pnp.copy(self.output_weights)
    
    def get_q_values(self, state, use_target=False):
        """Get Q-values for all actions"""
        if use_target:
            expectations = pnp.array(self.q_circuit(state, self.target_weights))
            q_values = pnp.array([
                pnp.sum(self.target_output_weights[a] * expectations) * 5  # Scale factor
                for a in range(self.n_actions)
            ])
        else:
            expectations = pnp.array(self.q_circuit(state, self.weights))
            q_values = pnp.array([
                pnp.sum(self.output_weights[a] * expectations) * 5
                for a in range(self.n_actions)
            ])
        return q_values
    
    def select_action(self, state, epsilon):
        """Epsilon-greedy action selection"""
        if np.random.random() < epsilon:
            return np.random.randint(0, self.n_actions)
        else:
            q_values = self.get_q_values(state, use_target=False)
            return int(np.argmax(np.array(q_values)))
    
    def update_target_network(self):
        """Copy weights to target network"""
        self.target_weights = pnp.copy(self.weights)
        self.target_output_weights = pnp.copy(self.output_weights)


class ReplayBuffer:
    """Experience replay buffer"""
    def __init__(self, capacity=1000):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            np.array(states),
            np.array(actions),
            np.array(rewards),
            np.array(next_states),
            np.array(dones)
        )
    
    def __len__(self):
        return len(self.buffer)


def encode_taxi_state(state_int):
    """
    Simple one-hot style encoding for Taxi-v3
    Taxi has 500 states, we use binary + positional encoding
    """
    # Extract components (Taxi-v3 specific)
    taxi_row = (state_int // 100) % 5
    taxi_col = (state_int // 20) % 5  
    pass_idx = (state_int // 4) % 5
    dest_idx = state_int % 4
    
    # Simple normalized encoding
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
    
    agent = QuantumQNetwork(
        n_qubits=5,
        n_actions=6,
        n_layers=2
    )
    
    replay_buffer = ReplayBuffer(capacity=1000)
    
    # Optimizers
    optimizer_weights = qml.AdamOptimizer(stepsize=0.02)  # Increased learning rate
    optimizer_output = qml.AdamOptimizer(stepsize=0.02)
    
    epochs = 100
    episodes_per_epoch = 10  # More episodes
    batch_size = 16  # Smaller batch
    gamma = 0.95
    epsilon_start = 1.0
    epsilon_end = 0.05  # Higher min exploration
    epsilon_decay = 0.99
    target_update_freq = 5
    
    epsilon = epsilon_start
    best_reward = -float('inf')
    
    for epoch in range(epochs):
        epoch_rewards = []
        epoch_steps = []
        
        for ep in range(episodes_per_epoch):
            state, _ = env.reset()
            done = False
            total_reward = 0
            steps = 0
            max_steps = 100  # Taxi episodes should be short
            
            while not done and steps < max_steps:
                encoded_state = encode_taxi_state(state)
                action = agent.select_action(encoded_state, epsilon)
                
                next_state, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                
                encoded_next_state = encode_taxi_state(next_state)
                
                replay_buffer.push(
                    encoded_state, action, reward,
                    encoded_next_state, done
                )
                
                total_reward += reward
                state = next_state
                steps += 1
            
            epoch_rewards.append(total_reward)
            epoch_steps.append(steps)
        
        # Train multiple times per epoch
        if len(replay_buffer) >= batch_size:
            for _ in range(3):  # Multiple updates
                states, actions, rewards, next_states, dones = replay_buffer.sample(batch_size)
                
                def loss_fn(params):
                    weights, output_weights = params
                    total_loss = 0
                    
                    for i in range(len(states)):
                        # Current Q
                        expectations = pnp.array(agent.q_circuit(states[i], weights))
                        q_values = pnp.array([
                            pnp.sum(output_weights[a] * expectations) * 5
                            for a in range(agent.n_actions)
                        ])
                        current_q = q_values[actions[i]]
                        
                        # Target Q
                        if dones[i]:
                            target_q = rewards[i]
                        else:
                            next_q = agent.get_q_values(next_states[i], use_target=True)
                            target_q = rewards[i] + gamma * pnp.max(next_q)
                        
                        # Huber loss
                        error = current_q - target_q
                        if pnp.abs(error) <= 1.0:
                            total_loss += 0.5 * error ** 2
                        else:
                            total_loss += pnp.abs(error) - 0.5
                    
                    return total_loss / len(states)
                
                # Update
                def weights_cost(w):
                    return loss_fn([w, agent.output_weights])
                agent.weights = optimizer_weights.step(weights_cost, agent.weights)
                
                def output_cost(o):
                    return loss_fn([agent.weights, o])
                agent.output_weights = optimizer_output.step(output_cost, agent.output_weights)
        
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
        
        print(f"Epoch {epoch + 1:3d} | "
              f"Avg: {avg_reward:6.1f} | "
              f"Max: {max_reward:6.1f} | "
              f"Steps: {avg_steps:5.1f} | "
              f"Best: {best_reward:6.1f} | "
              f"ε: {epsilon:.3f}")
        
        # Success check
        if avg_reward > 7:
            print(f"\nGood performance at epoch {epoch + 1}!")
            break
    
    env.close()
    
    # Test
    print("\n--- Testing trained agent ---")
    test_env = gym.make('Taxi-v3')
    test_rewards = []
    test_steps = []
    successes = 0
    
    for test_ep in range(20):
        state, _ = test_env.reset()
        total_reward = 0
        done = False
        steps = 0
        
        while not done and steps < 100:
            encoded_state = encode_taxi_state(state)
            q_values = agent.get_q_values(encoded_state, use_target=False)
            action = int(np.argmax(np.array(q_values)))
            
            state, reward, terminated, truncated, _ = test_env.step(action)
            total_reward += reward
            done = terminated or truncated
            steps += 1
        
        test_rewards.append(total_reward)
        test_steps.append(steps)
        if total_reward > 0:
            successes += 1
    
    print(f"\nTest Results:")
    print(f"  Avg reward: {np.mean(test_rewards):.2f} ± {np.std(test_rewards):.2f}")
    print(f"  Avg steps: {np.mean(test_steps):.1f}")
    print(f"  Success rate: {successes}/20 ({successes/20*100:.0f}%)")
    test_env.close()


if __name__ == '__main__':
    main()