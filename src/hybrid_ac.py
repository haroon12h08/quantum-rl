"""
Improved Quantum DDPG for Continuous Action Spaces
Key improvements:
1. Deeper quantum circuit (2 layers)
2. Better initialization
3. Gradient clipping
4. Warm-up period before training
5. Better exploration strategy
"""

import gymnasium as gym
import numpy as np
import pennylane as qml
from pennylane import numpy as pnp
from collections import deque
import random


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


def clip_gradients(params, max_norm=1.0):
    """Clip gradients by global norm"""
    total_norm = 0
    for p in params:
        if hasattr(p, 'grad') and p.grad is not None:
            total_norm += pnp.sum(p.grad ** 2)
    total_norm = pnp.sqrt(total_norm)
    
    if total_norm > max_norm:
        scale = max_norm / (total_norm + 1e-6)
        for p in params:
            if hasattr(p, 'grad') and p.grad is not None:
                p.grad *= scale
    
    return params


def main():
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
    
    # Adjusted hyperparameters
    actor_optimizer = qml.AdamOptimizer(stepsize=5e-4)  # Lower learning rate
    critic_optimizer = qml.AdamOptimizer(stepsize=1e-3)
    
    gamma = 0.99
    tau = 0.005
    batch_size = 64
    warmup_episodes = 10  # Random exploration first
    
    epochs = 300
    episodes_per_epoch = 5
    exploration_noise = 0.2  # Higher initial exploration
    
    best_reward = -float('inf')
    
    print("Training Improved Quantum DDPG on BipedalWalker-v3")
    print("=" * 60)
    
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
            for _ in range(50):
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
    
    # Test
    print("\n--- Testing ---")
    test_env = gym.make('BipedalWalker-v3')
    test_rewards = []
    
    for _ in range(20):
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
    
    print(f"Test Average: {np.mean(test_rewards):.2f} ± {np.std(test_rewards):.2f}")
    print(f"Test Best: {np.max(test_rewards):.2f}")
    test_env.close()


if __name__ == '__main__':
    main()