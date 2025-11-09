import gymnasium as gym
import numpy as np
import pennylane as qml
from pennylane import numpy as pnp


class ClassicalPreprocessor:
    """
    Classical neural network for dimensionality reduction
    Reduces high-dimensional state to quantum-friendly size
    """
    def __init__(self, input_dim, output_dim):
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        # Two-layer network
        hidden_dim = 16
        self.w1 = pnp.random.randn(input_dim, hidden_dim) * 0.1
        self.b1 = pnp.zeros(hidden_dim)
        self.w2 = pnp.random.randn(hidden_dim, output_dim) * 0.1
        self.b2 = pnp.zeros(output_dim)
        
        self.w1.requires_grad = True
        self.b1.requires_grad = True
        self.w2.requires_grad = True
        self.b2.requires_grad = True
    
    def forward(self, x):
        """Forward pass with tanh activation"""
        h = pnp.tanh(pnp.dot(x, self.w1) + self.b1)
        out = pnp.tanh(pnp.dot(h, self.w2) + self.b2)
        # Scale to quantum range
        return out * np.pi
    
    def parameters(self):
        return [self.w1, self.b1, self.w2, self.b2]


class HybridQuantumPolicy:
    """
    Hybrid quantum-classical policy network
    Uses RealAmplitudes ansatz (proven effective in research)
    """
    def __init__(self, state_dim=8, reduced_dim=2, n_actions=4, n_layers=2):
        self.state_dim = state_dim
        self.reduced_dim = reduced_dim
        self.n_actions = n_actions
        self.n_layers = n_layers
        
        # Classical preprocessor
        self.preprocessor = ClassicalPreprocessor(state_dim, reduced_dim)
        
        # Quantum circuit
        self.dev = qml.device('default.qubit', wires=reduced_dim)
        
        @qml.qnode(self.dev, interface='autograd')
        def quantum_circuit(inputs, weights):
            """
            RealAmplitudes ansatz - rotation + entanglement pattern
            Simpler and more effective than complex circuits
            """
            # Initial state encoding
            for i in range(self.reduced_dim):
                qml.RY(inputs[i], wires=i)
            
            # Variational layers
            for layer in range(self.n_layers):
                # Rotation layer
                for i in range(self.reduced_dim):
                    qml.RY(weights[layer, i, 0], wires=i)
                    qml.RZ(weights[layer, i, 1], wires=i)
                
                # Entanglement layer (linear topology)
                if self.reduced_dim > 1:
                    for i in range(self.reduced_dim - 1):
                        qml.CNOT(wires=[i, i + 1])
            
            # Measure all qubits
            return [qml.expval(qml.PauliZ(i)) for i in range(self.reduced_dim)]
        
        self.circuit = quantum_circuit
        
        # Quantum weights
        self.q_weights = pnp.random.uniform(
            0, 2*np.pi,
            size=(n_layers, reduced_dim, 2),
            requires_grad=True
        )
        
        # Classical post-processing layer
        self.output_weights = pnp.random.randn(n_actions, reduced_dim) * 0.1
        self.output_weights.requires_grad = True
    
    def forward(self, state):
        """Full forward pass: classical -> quantum -> classical"""
        # Step 1: Dimensionality reduction
        reduced_state = self.preprocessor.forward(state)
        
        # Step 2: Quantum processing
        q_output = pnp.array(self.circuit(reduced_state, self.q_weights))
        
        # Step 3: Classical post-processing to get action logits
        logits = pnp.dot(self.output_weights, q_output)
        
        return logits
    
    def get_action_probs(self, state):
        """Get action probability distribution"""
        logits = self.forward(state)
        # Softmax
        exp_logits = pnp.exp(logits - pnp.max(logits))
        probs = exp_logits / pnp.sum(exp_logits)
        return probs
    
    def select_action(self, state):
        """Sample action from policy"""
        probs = self.get_action_probs(state)
        probs_np = np.array(probs)
        action = np.random.choice(self.n_actions, p=probs_np)
        return action
    
    def parameters(self):
        """All trainable parameters"""
        return self.preprocessor.parameters() + [self.q_weights, self.output_weights]


def compute_returns(rewards, gamma=0.99):
    """Compute discounted returns"""
    returns = []
    R = 0
    for r in reversed(rewards):
        R = r + gamma * R
        returns.insert(0, R)
    returns = np.array(returns)
    # Normalize
    if len(returns) > 1:
        returns = (returns - np.mean(returns)) / (np.std(returns) + 1e-8)
    return returns


def main():
    env = gym.make('LunarLander-v3')
    
    # Hybrid model with dimensionality reduction
    agent = HybridQuantumPolicy(
        state_dim=8,      # LunarLander state
        reduced_dim=2,    # Reduced to 2 qubits (research recommendation)
        n_actions=4,
        n_layers=2        # Shallow circuit (proven better)
    )
    
    # Single optimizer for all parameters
    optimizer = qml.AdamOptimizer(stepsize=0.01)
    
    epochs = 50
    episodes_per_epoch = 10  # More episodes for stable learning
    
    best_reward = -float('inf')
    
    for epoch in range(epochs):
        epoch_trajectories = []
        epoch_rewards = []
        
        # Collect trajectories
        for ep in range(episodes_per_epoch):
            state, _ = env.reset()
            trajectory = {'states': [], 'actions': [], 'rewards': []}
            
            done = False
            steps = 0
            
            while not done and steps < 500:
                action = agent.select_action(state)
                next_state, reward, terminated, truncated, _ = env.step(action)
                
                trajectory['states'].append(state)
                trajectory['actions'].append(action)
                trajectory['rewards'].append(reward)
                
                state = next_state
                done = terminated or truncated
                steps += 1
            
            # Compute returns
            returns = compute_returns(trajectory['rewards'])
            trajectory['returns'] = returns
            
            epoch_trajectories.append(trajectory)
            epoch_rewards.append(sum(trajectory['rewards']))
        
        # Policy gradient update
        # FIX: Accept unpacked parameters instead of a single list
        def loss_fn(w1, b1, w2, b2, q_weights, output_weights):
            total_loss = 0
            n_steps = 0
            
            for traj in epoch_trajectories:
                for state, action, ret in zip(
                    traj['states'], 
                    traj['actions'],
                    traj['returns']
                ):
                    # Forward pass
                    # Preprocessor
                    h = pnp.tanh(pnp.dot(state, w1) + b1)
                    reduced = pnp.tanh(pnp.dot(h, w2) + b2) * np.pi
                    
                    # Quantum circuit
                    q_out = pnp.array(agent.circuit(reduced, q_weights))
                    
                    # Output layer
                    logits = pnp.dot(output_weights, q_out)
                    
                    # Softmax
                    exp_logits = pnp.exp(logits - pnp.max(logits))
                    probs = exp_logits / pnp.sum(exp_logits)
                    
                    # Log probability of taken action
                    log_prob = pnp.log(probs[action] + 1e-8)
                    
                    # Policy gradient loss
                    total_loss -= log_prob * ret
                    n_steps += 1
            
            return total_loss / n_steps if n_steps > 0 else 0
        
        # Update all parameters jointly
        params = agent.parameters()
        params = optimizer.step(loss_fn, *params)
        
        # Update agent parameters
        agent.preprocessor.w1 = params[0]
        agent.preprocessor.b1 = params[1]
        agent.preprocessor.w2 = params[2]
        agent.preprocessor.b2 = params[3]
        agent.q_weights = params[4]
        agent.output_weights = params[5]
        
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
        
        # Success threshold
        if avg_reward > 200:
            print(f"\nEnvironment solved at epoch {epoch + 1}!")
            break
    
    env.close()
    
    # Test
    print("\n--- Testing trained agent ---")
    test_env = gym.make('LunarLander-v3')
    test_rewards = []
    
    for test_ep in range(20):
        state, _ = test_env.reset()
        total_reward = 0
        done = False
        steps = 0
        
        while not done and steps < 500:
            probs = agent.get_action_probs(state)
            action = int(np.random.choice(agent.n_actions, p=np.array(probs)))
            state, reward, terminated, truncated, _ = test_env.step(action)
            total_reward += reward
            done = terminated or truncated
            steps += 1
        
        test_rewards.append(total_reward)
    
    print(f"\nTest Results:")
    print(f"  Average: {np.mean(test_rewards):.2f} ± {np.std(test_rewards):.2f}")
    print(f"  Median: {np.median(test_rewards):.2f}")
    print(f"  Best: {np.max(test_rewards):.2f}")
    test_env.close()


if __name__ == '__main__':
    main()