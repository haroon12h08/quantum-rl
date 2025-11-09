import gymnasium as gym
import numpy as np
import pennylane as qml
from pennylane import numpy as pnp


class QuantumActorCritic:
    """
    Improved Quantum Actor-Critic with:
    - More state dimensions
    - Better action space
    - Entropy bonus for exploration
    """
    def __init__(self, n_qubits=8, n_actions=9, n_layers=2):
        self.n_qubits = n_qubits
        self.n_actions = n_actions
        self.n_layers = n_layers
        
        # Separate devices for actor and critic
        self.actor_dev = qml.device('default.qubit', wires=n_qubits)
        self.critic_dev = qml.device('default.qubit', wires=n_qubits)
        
        # Actor circuit (policy)
        @qml.qnode(self.actor_dev, interface='autograd')
        def actor_circuit(inputs, weights, scaling):
            for layer in range(self.n_layers):
                # State encoding with scaling
                for i in range(self.n_qubits):
                    qml.RY(scaling[layer, i] * inputs[i], wires=i)
                
                # Variational layer
                for i in range(self.n_qubits):
                    qml.RY(weights[layer, i, 0], wires=i)
                    qml.RZ(weights[layer, i, 1], wires=i)
                    qml.RX(weights[layer, i, 2], wires=i)
                
                # Stronger entanglement
                for i in range(self.n_qubits):
                    qml.CNOT(wires=[i, (i + 1) % self.n_qubits])
                if layer < self.n_layers - 1:  # Extra entanglement except last layer
                    for i in range(0, self.n_qubits - 1, 2):
                        qml.CNOT(wires=[i, i + 1])
            
            return [qml.expval(qml.PauliZ(i)) for i in range(self.n_qubits)]
        
        # Critic circuit (value function)
        @qml.qnode(self.critic_dev, interface='autograd')
        def critic_circuit(inputs, weights, scaling):
            for layer in range(self.n_layers):
                # State encoding
                for i in range(self.n_qubits):
                    qml.RY(scaling[layer, i] * inputs[i], wires=i)
                
                # Variational layer
                for i in range(self.n_qubits):
                    qml.RY(weights[layer, i, 0], wires=i)
                    qml.RZ(weights[layer, i, 1], wires=i)
                    qml.RX(weights[layer, i, 2], wires=i)
                
                # Entanglement
                for i in range(self.n_qubits):
                    qml.CNOT(wires=[i, (i + 1) % self.n_qubits])
            
            # Average of multiple qubits for more stable value
            return [qml.expval(qml.PauliZ(i)) for i in range(min(3, self.n_qubits))]
        
        self.actor_circuit = actor_circuit
        self.critic_circuit = critic_circuit
        
        # Actor parameters (more gates = more parameters)
        self.actor_weights = pnp.random.uniform(
            -np.pi, np.pi, 
            size=(n_layers, n_qubits, 3),  # Added RX gate
            requires_grad=True
        )
        self.actor_scaling = pnp.ones(
            (n_layers, n_qubits), 
            requires_grad=True
        )
        self.actor_output_weights = pnp.random.uniform(
            -1, 1,
            size=(n_actions, n_qubits),
            requires_grad=True
        )
        
        # Critic parameters
        self.critic_weights = pnp.random.uniform(
            -np.pi, np.pi, 
            size=(n_layers, n_qubits, 3),
            requires_grad=True
        )
        self.critic_scaling = pnp.ones(
            (n_layers, n_qubits), 
            requires_grad=True
        )
        self.critic_output_weights = pnp.ones(3, requires_grad=True)
    
    def get_action_probs(self, state):
        """Get action probabilities from actor"""
        expectations = pnp.array(
            self.actor_circuit(state, self.actor_weights, self.actor_scaling)
        )
        
        logits = pnp.array([
            pnp.sum(self.actor_output_weights[action] * expectations)
            for action in range(self.n_actions)
        ])
        
        # Temperature scaling for exploration
        temperature = 1.5
        logits = logits / temperature
        
        exp_logits = pnp.exp(logits - pnp.max(logits))
        probs = exp_logits / pnp.sum(exp_logits)
        return probs
    
    def get_value(self, state):
        """Get state value from critic"""
        expectations = pnp.array(
            self.critic_circuit(state, self.critic_weights, self.critic_scaling)
        )
        # Average multiple measurements
        value = pnp.sum(self.critic_output_weights * expectations) * 20
        return value
    
    def select_action(self, state):
        """Sample action from policy"""
        probs = self.get_action_probs(state)
        probs_np = np.array(probs)
        action = np.random.choice(self.n_actions, p=probs_np)
        return action


def normalize_state(state):
    """
    BipedalWalker: Use 8 most important dimensions instead of 6
    Includes: hull angle, velocities, hip/knee angles for both legs
    """
    # Select critical features
    important_indices = [0, 1, 2, 3, 4, 5, 8, 9]  # Hull + leg hip joints
    reduced_state = state[important_indices]
    
    # Normalize to [-pi, pi] range
    state_bounds = np.array([3.14, 5.0, 3.14, 5.0, 3.14, 5.0, 3.14, 5.0])
    normalized = np.clip(reduced_state / state_bounds, -1, 1) * np.pi
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
    env = gym.make('BipedalWalker-v3')
    
    # Improved agent with more capacity
    agent = QuantumActorCritic(
        n_qubits=8,  # More state info
        n_actions=9,  # More granular control
        n_layers=2
    )
    
    # Separate optimizers for each parameter
    actor_weights_opt = qml.AdamOptimizer(stepsize=0.005)  # Slower learning
    actor_scaling_opt = qml.AdamOptimizer(stepsize=0.003)
    actor_output_opt = qml.AdamOptimizer(stepsize=0.005)
    
    critic_weights_opt = qml.AdamOptimizer(stepsize=0.01)
    critic_scaling_opt = qml.AdamOptimizer(stepsize=0.007)
    critic_output_opt = qml.AdamOptimizer(stepsize=0.01)
    
    epochs = 40
    episodes_per_epoch = 3
    
    best_reward = -float('inf')
    
    # Improved action mapping with more granular control
    action_map = {
        0: np.array([0.0, 0.0, 0.0, 0.0]),       # No action
        1: np.array([1.0, 0.0, 1.0, 0.0]),       # Forward strong
        2: np.array([0.5, 0.0, 0.5, 0.0]),       # Forward weak
        3: np.array([-1.0, 0.0, -1.0, 0.0]),     # Backward strong
        4: np.array([-0.5, 0.0, -0.5, 0.0]),     # Backward weak
        5: np.array([0.0, 1.0, 0.0, 1.0]),       # Jump
        6: np.array([0.0, 0.5, 0.0, 0.5]),       # Small jump
        7: np.array([1.0, 0.5, 1.0, 0.5]),       # Forward jump
        8: np.array([-1.0, 0.5, -1.0, 0.5])      # Backward jump
    }
    
    # Entropy coefficient for exploration
    entropy_coef = 0.01
    
    for epoch in range(epochs):
        epoch_rewards = []
        epoch_trajectories = []
        
        for ep in range(episodes_per_epoch):
            state, _ = env.reset()
            done = False
            
            trajectory = {
                'states': [],
                'actions': [],
                'rewards': [],
                'values': [],
                'log_probs': []
            }
            
            step_count = 0
            max_steps = 1000  # Longer episodes
            
            while not done and step_count < max_steps:
                norm_state = normalize_state(state)
                
                # Get action and value
                probs = agent.get_action_probs(norm_state)
                action = np.random.choice(agent.n_actions, p=np.array(probs))
                value = agent.get_value(norm_state)
                
                # Calculate log probability for this action
                log_prob = pnp.log(probs[action] + 1e-8)
                
                # Map discrete action to continuous
                continuous_action = action_map[action]
                
                next_state, reward, terminated, truncated, _ = env.step(continuous_action)
                done = terminated or truncated
                
                trajectory['states'].append(norm_state)
                trajectory['actions'].append(action)
                trajectory['rewards'].append(reward)
                trajectory['values'].append(value)
                trajectory['log_probs'].append(log_prob)
                
                state = next_state
                step_count += 1
            
            # Calculate advantages
            returns = discount_rewards(trajectory['rewards'])
            values = np.array([float(v) for v in trajectory['values']])
            advantages = returns - values
            
            # Normalize advantages
            if len(advantages) > 1:
                advantages = (advantages - np.mean(advantages)) / (np.std(advantages) + 1e-8)
            
            trajectory['returns'] = returns
            trajectory['advantages'] = advantages
            
            epoch_trajectories.append(trajectory)
            epoch_rewards.append(sum(trajectory['rewards']))
        
        # Update Actor with entropy bonus
        def actor_loss(actor_params):
            actor_weights, actor_scaling, actor_output = actor_params
            policy_loss = 0
            entropy_loss = 0
            n_steps = 0
            
            for traj in epoch_trajectories:
                for state, action, advantage in zip(
                    traj['states'], 
                    traj['actions'], 
                    traj['advantages']
                ):
                    expectations = pnp.array(
                        agent.actor_circuit(state, actor_weights, actor_scaling)
                    )
                    logits = pnp.array([
                        pnp.sum(actor_output[a] * expectations)
                        for a in range(agent.n_actions)
                    ])
                    
                    temperature = 1.5
                    logits = logits / temperature
                    
                    exp_logits = pnp.exp(logits - pnp.max(logits))
                    probs = exp_logits / pnp.sum(exp_logits)
                    
                    log_prob = pnp.log(probs[action] + 1e-8)
                    policy_loss -= log_prob * advantage
                    
                    # Entropy bonus for exploration
                    entropy = -pnp.sum(probs * pnp.log(probs + 1e-8))
                    entropy_loss -= entropy
                    
                    n_steps += 1
            
            total_loss = policy_loss + entropy_coef * entropy_loss
            return total_loss / n_steps if n_steps > 0 else 0
        
        # Update Critic
        def critic_loss(critic_params):
            critic_weights, critic_scaling, critic_output = critic_params
            loss = 0
            n_steps = 0
            
            for traj in epoch_trajectories:
                for state, ret in zip(traj['states'], traj['returns']):
                    expectations = pnp.array(
                        agent.critic_circuit(state, critic_weights, critic_scaling)
                    )
                    predicted_value = pnp.sum(critic_output * expectations) * 20
                    
                    # Huber loss (more robust than MSE)
                    error = predicted_value - ret
                    huber_delta = 10.0
                    if pnp.abs(error) <= huber_delta:
                        loss += 0.5 * error ** 2
                    else:
                        loss += huber_delta * (pnp.abs(error) - 0.5 * huber_delta)
                    
                    n_steps += 1
            
            return loss / n_steps if n_steps > 0 else 0
        
        # Update actor parameters
        def actor_weights_cost(w):
            return actor_loss([w, agent.actor_scaling, agent.actor_output_weights])
        agent.actor_weights = actor_weights_opt.step(actor_weights_cost, agent.actor_weights)
        
        def actor_scaling_cost(s):
            return actor_loss([agent.actor_weights, s, agent.actor_output_weights])
        agent.actor_scaling = actor_scaling_opt.step(actor_scaling_cost, agent.actor_scaling)
        
        def actor_output_cost(o):
            return actor_loss([agent.actor_weights, agent.actor_scaling, o])
        agent.actor_output_weights = actor_output_opt.step(actor_output_cost, agent.actor_output_weights)
        
        # Update critic parameters
        def critic_weights_cost(w):
            return critic_loss([w, agent.critic_scaling, agent.critic_output_weights])
        agent.critic_weights = critic_weights_opt.step(critic_weights_cost, agent.critic_weights)
        
        def critic_scaling_cost(s):
            return critic_loss([agent.critic_weights, s, agent.critic_output_weights])
        agent.critic_scaling = critic_scaling_opt.step(critic_scaling_cost, agent.critic_scaling)
        
        def critic_output_cost(o):
            return critic_loss([agent.critic_weights, agent.critic_scaling, o])
        agent.critic_output_weights = critic_output_opt.step(critic_output_cost, agent.critic_output_weights)
        
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
        if avg_reward > 250:
            print(f"\nEnvironment performing well at epoch {epoch + 1}!")
            break
    
    env.close()
    
    # Test the trained agent
    print("\n--- Testing trained agent ---")
    test_env = gym.make('BipedalWalker-v3', render_mode='rgb_array')
    test_rewards = []
    
    for test_ep in range(5):
        state, _ = test_env.reset()
        total_reward = 0
        done = False
        steps = 0
        
        while not done and steps < 1000:
            norm_state = normalize_state(state)
            probs = agent.get_action_probs(norm_state)
            action = np.random.choice(agent.n_actions, p=np.array(probs))
            continuous_action = action_map[action]
            
            state, reward, terminated, truncated, _ = test_env.step(continuous_action)
            total_reward += reward
            done = terminated or truncated
            steps += 1
        
        test_rewards.append(total_reward)
        print(f"Test episode {test_ep + 1}: {total_reward:.2f}")
    
    print(f"\nTest average: {np.mean(test_rewards):.2f} ± {np.std(test_rewards):.2f}")
    test_env.close()


if __name__ == '__main__':
    main()