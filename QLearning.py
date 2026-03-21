# ===============================
# 1. Import Libraries
# ===============================
import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt

# ===============================
# 2. Create Environments
# ===============================
# Fast environment for training (no rendering)
env = gym.make("FrozenLake-v1")

# Separate environment for visualization (rendering)
visual_env = gym.make("FrozenLake-v1", render_mode="human")

# ===============================
# 3. Q-Table Initialization
# ===============================
num_states = env.observation_space.n
num_actions = env.action_space.n
q_table = np.zeros((num_states, num_actions))

# ===============================
# 4. Hyperparameters
# ===============================
learning_rate = 0.1  # alpha
discount_factor = 0.99  # gamma
episodes = 2000
max_steps = 100
epsilon = 0.1  # small randomness for training

# For plotting learning progress
rewards_per_episode = []

# ===============================
# 5. Training Loop
# ===============================
for episode in range(episodes):
    state, _ = env.reset()
    total_rewards = 0

    for step in range(max_steps):
        # Epsilon-greedy: mostly best action, sometimes random
        if np.random.uniform(0, 1) < epsilon:
            action = env.action_space.sample()  # explore
        else:
            action = np.argmax(q_table[state])  # exploit

        new_state, reward, terminated, truncated, _ = env.step(action)

        # Q-learning update (Bellman equation)
        best_future_q = np.max(q_table[new_state])
        current_q = q_table[state, action]
        q_table[state, action] = current_q + learning_rate * (
                reward + discount_factor * best_future_q - current_q
        )

        state = new_state
        total_rewards += reward

        if terminated or truncated:
            break

    rewards_per_episode.append(total_rewards)

# ===============================
# 6. Plot Learning Curve
# ===============================
# Raw plot
plt.figure()
plt.plot(rewards_per_episode, alpha=0.5, label='Reward per episode')

# Smoothed plot
window = 50
smoothed = np.convolve(rewards_per_episode, np.ones(window) / window, mode='valid')
plt.plot(range(window - 1, episodes), smoothed, color='red', label='Smoothed Reward')

plt.xlabel("Episode")
plt.ylabel("Reward")
plt.title("FrozenLake Learning Progress")
plt.legend()
plt.show()

# ===============================
# 7. Save Q-Table
# ===============================
np.save("q_table.npy", q_table)


# ===============================
# 8. Display Learned Policy
# ===============================
def display_policy(q_table):
    arrows = ['←', '↓', '→', '↑']
    policy = np.argmax(q_table, axis=1)
    grid = policy.reshape((4, 4))

    print("\nLearned Policy:")
    for row in grid:
        print(" ".join(arrows[a] for a in row))


display_policy(q_table)

# ===============================
# 9. Test the Trained Agent (100 episodes)
# ===============================
test_episodes = 100
successes = 0

for episode in range(test_episodes):
    state, _ = env.reset()
    for step in range(max_steps):
        action = np.argmax(q_table[state])  # exploit learned policy
        state, reward, terminated, truncated, _ = env.step(action)

        if terminated or truncated:
            if reward == 1:
                successes += 1
            break

success_rate = (successes / test_episodes) * 100
print(f"\nSuccess Rate over {test_episodes} test episodes: {success_rate:.2f}%")

# ===============================
# 10. Watch the Agent Navigate
# ===============================
print("\nWatching the trained agent...\n")
for episode in range(3):  # watch 3 episodes
    state, _ = visual_env.reset()
    for step in range(max_steps):
        action = np.argmax(q_table[state])
        state, reward, terminated, truncated, _ = visual_env.step(action)
        if terminated or truncated:
            break