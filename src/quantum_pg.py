"""
Optimized Quantum Policy Gradient for LunarLander-v3
Uses data re-uploading and trainable input scaling
"""

import gymnasium as gym
import numpy as np
import pennylane as qml
from pennylane import numpy as pnp
import csv
from datetime import datetime


class QuantumPolicyGradient:
    """Quantum policy with data re-uploading"""
    def __init__(self, n_qubits=4, n_actions=4, n_layers=3):
        self.n_qubits = n_qubits
        self.n_actions = n_actions
        self.n_layers = n_layers
        self.dev = qml.device('default.qubit', wires=n_qubits)
        
        @qml.qnode(self.dev, interface='autograd')
        def circuit(inputs, weights, scaling):
            """Data re-uploading circuit with trainable input scaling"""
            for layer in range(n_layers):
                # Re-upload scaled data at each layer
                for i in range(n_qubits):
                    qml.RY(scaling[layer, i] * inputs[i], wires=i)
                
                # Variational layer
                for i in range(n_qubits):
                    qml.RY(weights[layer, i, 0], wires=i)
                    qml.RZ(weights[layer, i, 1], wires=i)
                
                # Entanglement
                for i in range(n_qubits):
                    qml.CNOT(wires=[i, (i + 1) % n_qubits])
            
            return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]
        
        self.circuit = circuit
        
        # Initialize parameters
        self.weights = pnp.random.uniform(-np.pi, np.pi, size=(n_layers, n_qubits, 2), requires_grad=True)
        self.scaling = pnp.ones((n_layers, n_qubits), requires_grad=True)
        self.output_weights = pnp.random.randn(n_actions, n_qubits) * 0.1
        self.output_weights.requires_grad = True
    
    def get_action_probs(self, state):
        """Get action probabilities"""
        expectations = pnp.array(self.circuit(state, self.weights, self.scaling))
        logits = pnp.array([pnp.sum(self.output_weights[a] * expectations) for a in range(self.n_actions)])
        exp_logits = pnp.exp(logits - pnp.max(logits))
        return exp_logits / pnp.sum(exp_logits)
    
    def select_action(self, state):
        """Sample action from policy"""
        probs = self.get_action_probs(state)
        return int(np.random.choice(self.n_actions, p=np.array(probs)))
    
    def parameters(self):
        return [self.weights, self.scaling, self.output_weights]


def normalize_state(state):
    """Normalize LunarLander state to quantum range"""
    bounds = np.array([1.5, 1.5, 5.0, 5.0, 3.14, 5.0, 1.0, 1.0])
    return np.clip(state / bounds, -1, 1) * np.pi


def discount_rewards(rewards, gamma=0.99):
    """Calculate discounted returns"""
    discounted = np.zeros_like(rewards, dtype=np.float64)
    cumulative = 0
    for i in reversed(range(len(rewards))):
        cumulative = rewards[i] + gamma * cumulative
        discounted[i] = cumulative
    return discounted


def main():
    env = gym.make('LunarLander-v3')
    
    agent = QuantumPolicyGradient(n_qubits=4, n_actions=4, n_layers=3)
    
    # Single optimizer for all parameters
    optimizer = qml.AdamOptimizer(stepsize=0.01)
    
    epochs = 10
    episodes_per_epoch = 10
    gamma = 0.99
    
    results = []
    best_reward = -float('inf')
    
    print("Training Quantum Policy Gradient on LunarLander-v3")
    print("=" * 60)
    
    for epoch in range(epochs):
        epoch_trajectories = []
        epoch_rewards = []
        
        for ep in range(episodes_per_epoch):
            state, _ = env.reset()
            trajectory = {'states': [], 'actions': [], 'rewards': []}
            done = False
            steps = 0
            
            while not done and steps < 500:
                norm_state = normalize_state(state)
                action = agent.select_action(norm_state)
                
                next_state, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                
                trajectory['states'].append(norm_state)
                trajectory['actions'].append(action)
                trajectory['rewards'].append(reward)
                
                state = next_state
                steps += 1
            
            # Calculate returns
            returns = discount_rewards(trajectory['rewards'], gamma)
            
            # Normalize returns
            if len(returns) > 1:
                returns = (returns - np.mean(returns)) / (np.std(returns) + 1e-8)
            
            trajectory['returns'] = returns
            epoch_trajectories.append(trajectory)
            epoch_rewards.append(sum(trajectory['rewards']))
        
        # Policy gradient update
        def loss_fn(weights, scaling, output_weights):
            total_loss = 0
            n_steps = 0
            
            for traj in epoch_trajectories:
                for state, action, ret in zip(traj['states'], traj['actions'], traj['returns']):
                    expectations = pnp.array(agent.circuit(state, weights, scaling))
                    logits = pnp.array([pnp.sum(output_weights[a] * expectations) for a in range(agent.n_actions)])
                    exp_logits = pnp.exp(logits - pnp.max(logits))
                    probs = exp_logits / pnp.sum(exp_logits)
                    log_prob = pnp.log(probs[action] + 1e-8)
                    total_loss -= log_prob * ret
                    n_steps += 1
            
            return total_loss / n_steps if n_steps > 0 else 0
        
        params = agent.parameters()
        params = optimizer.step(loss_fn, *params)
        agent.weights, agent.scaling, agent.output_weights = params
        
        # Logging
        avg_reward = np.mean(epoch_rewards)
        max_reward = np.max(epoch_rewards)
        min_reward = np.min(epoch_rewards)
        if avg_reward > best_reward:
            best_reward = avg_reward
        
        results.append({
            'epoch': epoch + 1,
            'avg_reward': avg_reward,
            'max_reward': max_reward,
            'min_reward': min_reward,
            'best_reward': best_reward
        })
        
        print(f"Epoch {epoch + 1:2d} | Avg: {avg_reward:7.2f} | Max: {max_reward:7.2f} | "
              f"Min: {min_reward:7.2f} | Best: {best_reward:7.2f}")
        
        if avg_reward > 200:
            print(f"\nEnvironment solved at epoch {epoch + 1}!")
            break
    
    env.close()
    
    # Save results to CSV
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_filename = f'results_lunarlander_{timestamp}.csv'
    
    with open(csv_filename, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['epoch', 'avg_reward', 'max_reward', 'min_reward', 'best_reward'])
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
    
    model_filename = f'model_lunarlander_{timestamp}.npy'
    np.save(model_filename, model_data)
    print(f"✓ Model saved to: {model_filename}")
    
    return results


if __name__ == '__main__':
    main()