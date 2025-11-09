import gymnasium as gym
import numpy as np
import pennylane as qml
from pennylane import numpy as pnp


class ImprovedQuantumPolicyGradient:
    def __init__(self, n_qubits=4, n_actions=4, n_layers=3):
        """
        Improved quantum circuit with data re-uploading and input scaling.
        
        Based on recent research:
        - Data re-uploading for expressiveness
        - Input scaling parameters for better training
        - Multiple observables for action selection
        """
        self.n_qubits = n_qubits
        self.n_actions = n_actions
        self.n_layers = n_layers
        self.dev = qml.device('default.qubit', wires=self.n_qubits)
        
        @qml.qnode(self.dev, interface='autograd')
        def circuit(inputs, weights, scaling):
            """
            Data re-uploading circuit with input scaling.
            
            Structure per layer:
            1. Scaled state encoding (RY gates)
            2. Variational rotations (RY, RZ)
            3. Entangling gates (CNOT ring)
            """
            for layer in range(self.n_layers):
                # Data re-uploading: encode state in each layer
                for i in range(self.n_qubits):
                    # Apply scaled input
                    scaled_input = scaling[layer, i] * inputs[i]
                    qml.RY(scaled_input, wires=i)
                
                # Variational layer
                for i in range(self.n_qubits):
                    qml.RY(weights[layer, i, 0], wires=i)
                    qml.RZ(weights[layer, i, 1], wires=i)
                
                # Entangling layer (ring topology)
                for i in range(self.n_qubits):
                    qml.CNOT(wires=[i, (i + 1) % self.n_qubits])
            
            # Return measurements for all qubits (used for different actions)
            return [qml.expval(qml.PauliZ(i)) for i in range(self.n_qubits)]
        
        self.circuit = circuit
        
        # Initialize trainable parameters
        self.weights = pnp.random.uniform(
            -np.pi, np.pi, 
            size=(n_layers, n_qubits, 2), 
            requires_grad=True
        )
        
        # Input scaling parameters (crucial for training)
        self.scaling = pnp.ones(
            (n_layers, n_qubits), 
            requires_grad=True
        )
        
        # Output weights for each action
        self.output_weights = pnp.random.uniform(
            -1, 1,
            size=(n_actions, n_qubits),
            requires_grad=True
        )
    
    def get_action_logits(self, state):
        """Get raw logits for all actions using output weights"""
        # Get expectation values for all qubits
        expectations = pnp.array(self.circuit(state, self.weights, self.scaling))
        
        # Combine expectations using learned output weights for each action
        logits = pnp.array([
            pnp.sum(self.output_weights[action] * expectations)
            for action in range(self.n_actions)
        ])
        
        return logits
    
    def get_action_probs(self, state):
        """Get probability distribution over all actions"""
        logits = self.get_action_logits(state)
        
        # Softmax with temperature scaling
        exp_logits = pnp.exp(logits - pnp.max(logits))
        probs = exp_logits / pnp.sum(exp_logits)
        return probs
    
    def select_action(self, state):
        """Sample action from policy"""
        probs = self.get_action_probs(state)
        probs_np = np.array(probs)
        action = np.random.choice(self.n_actions, p=probs_np)
        return action, probs
    
    def get_trainable_params(self):
        """Return all trainable parameters"""
        return [self.weights, self.scaling, self.output_weights]


def normalize_state(state):
    """Normalize state to appropriate range for quantum encoding"""
    # LunarLander state space ranges
    state_bounds = np.array([1.5, 1.5, 5.0, 5.0, 3.14, 5.0, 1.0, 1.0])
    normalized = np.clip(state / state_bounds, -1, 1) * np.pi
    return normalized


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
    
    # Improved agent with data re-uploading
    agent = ImprovedQuantumPolicyGradient(
        n_qubits=4, 
        n_actions=4,
        n_layers=3  # More layers for better expressiveness
    )
    
    # Separate optimizers with different learning rates (proven technique)
    opt_weights = qml.AdamOptimizer(stepsize=0.01)
    opt_scaling = qml.AdamOptimizer(stepsize=0.005)
    opt_output = qml.AdamOptimizer(stepsize=0.01)
    
    epochs = 30  # Reduced for faster training
    episodes_per_epoch = 5
    
    best_reward = -float('inf')
    
    for epoch in range(epochs):
        epoch_rewards = []
        epoch_trajectories = []
        
        for ep in range(episodes_per_epoch):
            state, _ = env.reset()
            done = False
            
            trajectory = {
                'states': [],
                'actions': [],
                'rewards': []
            }
            
            step_count = 0
            max_steps = 500
            
            while not done and step_count < max_steps:
                norm_state = normalize_state(state)
                action, probs = agent.select_action(norm_state)
                
                next_state, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                
                trajectory['states'].append(norm_state)
                trajectory['actions'].append(action)
                trajectory['rewards'].append(reward)
                
                state = next_state
                step_count += 1
            
            # Calculate returns with baseline subtraction
            returns = discount_rewards(trajectory['rewards'])
            
            # Normalize returns (reduces variance)
            if len(returns) > 1:
                returns = (returns - np.mean(returns)) / (np.std(returns) + 1e-8)
            
            trajectory['returns'] = returns
            epoch_trajectories.append(trajectory)
            epoch_rewards.append(sum(trajectory['rewards']))
        
        # Policy gradient update
        def cost_fn(params):
            weights, scaling, output_weights = params
            total_loss = 0
            n_steps = 0
            
            for traj in epoch_trajectories:
                for state, action, ret in zip(
                    traj['states'], 
                    traj['actions'], 
                    traj['returns']
                ):
                    # Get expectations
                    expectations = pnp.array(agent.circuit(state, weights, scaling))
                    
                    # Calculate logits for all actions
                    logits = pnp.array([
                        pnp.sum(output_weights[a] * expectations)
                        for a in range(agent.n_actions)
                    ])
                    
                    # Softmax
                    exp_logits = pnp.exp(logits - pnp.max(logits))
                    action_probs = exp_logits / pnp.sum(exp_logits)
                    
                    # Log probability of taken action
                    log_prob = pnp.log(action_probs[action] + 1e-8)
                    
                    # Policy gradient loss
                    total_loss -= log_prob * ret
                    n_steps += 1
            
            return total_loss / n_steps if n_steps > 0 else 0
        
        # Update weights
        def cost_weights(w):
            return cost_fn([w, agent.scaling, agent.output_weights])
        agent.weights = opt_weights.step(cost_weights, agent.weights)
        
        # Update scaling
        def cost_scaling(s):
            return cost_fn([agent.weights, s, agent.output_weights])
        agent.scaling = opt_scaling.step(cost_scaling, agent.scaling)
        
        # Update output weights
        def cost_output(o):
            return cost_fn([agent.weights, agent.scaling, o])
        agent.output_weights = opt_output.step(cost_output, agent.output_weights)
        
        # Logging
        avg_reward = np.mean(epoch_rewards)
        max_reward = np.max(epoch_rewards)
        min_reward = np.min(epoch_rewards)
        
        if avg_reward > best_reward:
            best_reward = avg_reward
        
        print(f"Epoch {epoch + 1:2d} | "
              f"Avg: {avg_reward:7.2f} | "
              f"Max: {max_reward:7.2f} | "
              f"Min: {min_reward:7.2f} | "
              f"Best: {best_reward:7.2f}")
        
        # Early stopping
        if avg_reward > 200:
            print(f"\nEnvironment solved in {epoch + 1} epochs!")
            break
    
    env.close()
    
    # Test the trained agent
    print("\n--- Testing trained agent ---")
    test_env = gym.make('LunarLander-v3', render_mode='rgb_array')
    test_rewards = []
    
    for test_ep in range(10):
        state, _ = test_env.reset()
        total_reward = 0
        done = False
        steps = 0
        
        while not done and steps < 500:
            norm_state = normalize_state(state)
            probs = agent.get_action_probs(norm_state)
            action = np.argmax(np.array(probs))  # Greedy
            state, reward, terminated, truncated, _ = test_env.step(action)
            total_reward += reward
            done = terminated or truncated
            steps += 1
        
        test_rewards.append(total_reward)
        print(f"Test episode {test_ep + 1}: {total_reward:.2f}")
    
    print(f"\nTest average: {np.mean(test_rewards):.2f} ± {np.std(test_rewards):.2f}")
    test_env.close()


if __name__ == '__main__':
    main()