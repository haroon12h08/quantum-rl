"""
Optimized Quantum Actor-Critic for BipedalWalker-v3
Simplified from complex version while keeping key improvements
"""

import gymnasium as gym
import numpy as np
import pennylane as qml
from pennylane import numpy as pnp
import csv
from datetime import datetime


class QuantumActorCritic:
    """Quantum Actor-Critic with data re-uploading inspiration"""
    def __init__(self, n_qubits=4, n_actions=4, n_layers=2):
        self.n_qubits = n_qubits
        self.n_actions = n_actions
        self.n_layers = n_layers
        
        self.dev = qml.device('default.qubit', wires=n_qubits)
        
        # Actor circuit with data re-uploading
        @qml.qnode(self.dev, interface='autograd')
        def actor_circuit(inputs, weights):
            for layer in range(n_layers):
                # Re-upload data at each layer
                for i in range(n_qubits):
                    qml.RY(inputs[i], wires=i)
                
                # Variational layer
                for i in range(n_qubits):
                    qml.RY(weights[layer, i, 0], wires=i)
                    qml.RZ(weights[layer, i, 1], wires=i)
                
                # Entanglement
                for i in range(n_qubits - 1):
                    qml.CNOT(wires=[i, i + 1])
            
            return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]
        
        # Critic circuit
        @qml.qnode(self.dev, interface='autograd')
        def critic_circuit(inputs, weights):
            for layer in range(n_layers):
                for i in range(n_qubits):
                    qml.RY(inputs[i], wires=i)
                
                for i in range(n_qubits):
                    qml.RY(weights[layer, i, 0], wires=i)
                    qml.RZ(weights[layer, i, 1], wires=i)
                
                for i in range(n_qubits - 1):
                    qml.CNOT(wires=[i, i + 1])
            
            return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]
        
        self.actor_circuit = actor_circuit
        self.critic_circuit = critic_circuit
        
        # Trainable input scaling (key from data re-uploading)
        self.actor_input_scale = pnp.ones(n_qubits, requires_grad=True)
        self.critic_input_scale = pnp.ones(n_qubits, requires_grad=True)
        
        # Quantum weights
        self.actor_weights = pnp.random.uniform(0, 2*np.pi, size=(n_layers, n_qubits, 2), requires_grad=True)
        self.critic_weights = pnp.random.uniform(0, 2*np.pi, size=(n_layers, n_qubits, 2), requires_grad=True)
        
        # Output layers
        self.actor_output = pnp.random.randn(n_actions, n_qubits) * 0.1
        self.actor_output.requires_grad = True
        self.critic_output = pnp.random.randn(n_qubits) * 0.1
        self.critic_output.requires_grad = True
    
    def get_action_probs(self, state):
        scaled_state = state * self.actor_input_scale
        q_out = pnp.array(self.actor_circuit(scaled_state, self.actor_weights))
        logits = pnp.dot(self.actor_output, q_out)
        exp_logits = pnp.exp(logits - pnp.max(logits))
        return exp_logits / pnp.sum(exp_logits)
    
    def get_value(self, state):
        scaled_state = state * self.critic_input_scale
        q_out = pnp.array(self.critic_circuit(scaled_state, self.critic_weights))
        return pnp.dot(self.critic_output, q_out)
    
    def actor_parameters(self):
        return [self.actor_input_scale, self.actor_weights, self.actor_output]
    
    def critic_parameters(self):
        return [self.critic_input_scale, self.critic_weights, self.critic_output]


def normalize_state(state):
    """Reduce 24D to 4D using most critical features"""
    # Hull angle, angular velocity, and first leg's joint angles
    indices = [0, 2, 4, 6]
    reduced = state[indices]
    # Normalize
    bounds = np.array([3.14, 5.0, 3.14, 3.14])
    return np.clip(reduced / bounds, -1, 1) * np.pi


def main():
    env = gym.make('BipedalWalker-v3')
    
    agent = QuantumActorCritic(n_qubits=4, n_actions=4, n_layers=2)
    
    actor_opt = qml.AdamOptimizer(stepsize=0.003)
    critic_opt = qml.AdamOptimizer(stepsize=0.01)
    
    epochs = 10
    episodes_per_epoch = 5
    gamma = 0.99
    
    # Action mapping
    action_map = {
        0: np.array([0.0, 0.0, 0.0, 0.0]),
        1: np.array([1.0, 0.0, 1.0, 0.0]),
        2: np.array([-1.0, 0.0, -1.0, 0.0]),
        3: np.array([0.5, 1.0, 0.5, 1.0])
    }
    
    # Track results for CSV
    results = []
    best_reward = -float('inf')
    
    print("Training Quantum Actor-Critic on BipedalWalker-v3")
    print("=" * 60)
    
    for epoch in range(epochs):
        epoch_trajectories = []
        epoch_rewards = []
        
        for ep in range(episodes_per_epoch):
            state, _ = env.reset()
            trajectory = {'states': [], 'actions': [], 'rewards': [], 'values': []}
            done = False
            steps = 0
            
            while not done and steps < 1600:
                norm_state = normalize_state(state)
                probs = agent.get_action_probs(norm_state)
                action = int(np.random.choice(agent.n_actions, p=np.array(probs)))
                value = agent.get_value(norm_state)
                
                next_state, reward, terminated, truncated, _ = env.step(action_map[action])
                done = terminated or truncated
                
                trajectory['states'].append(norm_state)
                trajectory['actions'].append(action)
                trajectory['rewards'].append(reward)
                trajectory['values'].append(float(value))
                
                state = next_state
                steps += 1
            
            # Calculate advantages
            returns = []
            R = 0
            for r in reversed(trajectory['rewards']):
                R = r + gamma * R
                returns.insert(0, R)
            returns = np.array(returns)
            values = np.array(trajectory['values'])
            advantages = returns - values
            if len(advantages) > 1:
                advantages = (advantages - np.mean(advantages)) / (np.std(advantages) + 1e-8)
            
            trajectory['returns'] = returns
            trajectory['advantages'] = advantages
            epoch_trajectories.append(trajectory)
            epoch_rewards.append(sum(trajectory['rewards']))
        
        # Update actor
        def actor_loss(input_scale, weights, output):
            loss = 0
            n = 0
            for traj in epoch_trajectories:
                for state, action, adv in zip(traj['states'], traj['actions'], traj['advantages']):
                    scaled = state * input_scale
                    q_out = pnp.array(agent.actor_circuit(scaled, weights))
                    logits = pnp.dot(output, q_out)
                    exp_logits = pnp.exp(logits - pnp.max(logits))
                    probs = exp_logits / pnp.sum(exp_logits)
                    log_prob = pnp.log(probs[action] + 1e-8)
                    loss -= log_prob * adv
                    n += 1
            return loss / n if n > 0 else 0
        
        # Update critic
        def critic_loss(input_scale, weights, output):
            loss = 0
            n = 0
            for traj in epoch_trajectories:
                for state, ret in zip(traj['states'], traj['returns']):
                    scaled = state * input_scale
                    q_out = pnp.array(agent.critic_circuit(scaled, weights))
                    value = pnp.dot(output, q_out)
                    loss += (value - ret) ** 2
                    n += 1
            return loss / n if n > 0 else 0
        
        actor_params = agent.actor_parameters()
        actor_params = actor_opt.step(actor_loss, *actor_params)
        agent.actor_input_scale, agent.actor_weights, agent.actor_output = actor_params
        
        critic_params = agent.critic_parameters()
        critic_params = critic_opt.step(critic_loss, *critic_params)
        agent.critic_input_scale, agent.critic_weights, agent.critic_output = critic_params
        
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
    
    env.close()
    
    # Save results to CSV
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_filename = f'results_bipedal_{timestamp}.csv'
    
    with open(csv_filename, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['epoch', 'avg_reward', 'max_reward', 'min_reward', 'best_reward'])
        writer.writeheader()
        writer.writerows(results)
    
    print(f"\n✓ Results saved to: {csv_filename}")
    
    # Save model
    model_data = {
        'actor_input_scale': np.array(agent.actor_input_scale),
        'actor_weights': np.array(agent.actor_weights),
        'actor_output': np.array(agent.actor_output),
        'critic_input_scale': np.array(agent.critic_input_scale),
        'critic_weights': np.array(agent.critic_weights),
        'critic_output': np.array(agent.critic_output),
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
    
    model_filename = f'model_bipedal_{timestamp}.npy'
    np.save(model_filename, model_data)
    print(f"✓ Model saved to: {model_filename}")
    
    return results


if __name__ == '__main__':
    main()