import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
import matplotlib.pyplot as plt
 
# -------------------------
# 1. TicTacToe Environment
# -------------------------
class TicTacToeEnv:
    """
    Board representation:
      We store the board as a 1D array of length 9:
        0 = empty
        1 = agent 'X'
        2 = opponent 'O'
    The environment is turn-based, with current_player in {1, 2}.
    After a move:
      - Check if there's a winner or if the board is full (draw).
      - Return a reward (+1 if agent wins, -1 if opponent wins, 0 if draw).
    """
    def __init__(self):
        self.reset()
 
    def reset(self):
        self.board = np.zeros(9, dtype=int)  # 9 cells for 3x3
        self.done = False
        self.winner = None
        self.current_player = 1  # Start with player 1 (the agent)
        return self.get_observation()
 
    def get_observation(self):
        # Simply return a copy of the board (state is the 1D board)
        return self.board.copy()
 
    def step(self, action):
        """
        action: an integer in [0..8] indicating where to place the current player's mark.
       
        Returns:
          next_state (np array) - the updated board
          reward (float) - the reward for this step
          done (bool) - whether the game ended
        """
        # Check if the action is valid (the cell is empty):
        if self.board[action] != 0:
            # Invalid move: penalize heavily and end the game
            return self.get_observation(), -1.0, True
 
        # Place the mark
        self.board[action] = self.current_player
 
        # Check for win/draw
        done, winner = self.check_game_status()
        if done:
            self.done = True
            self.winner = winner
            reward = self.get_reward(winner)
            return self.get_observation(), reward, True
        else:
            # Switch turn to the other player
            self.current_player = 1 if self.current_player == 2 else 2
            return self.get_observation(), 0.0, False
 
    def check_game_status(self):
        # Rows
        for i in range(3):
            if (self.board[3*i] != 0 and
                self.board[3*i] == self.board[3*i+1] == self.board[3*i+2]):
                return True, self.board[3*i]
       
        # Columns
        for i in range(3):
            if (self.board[i] != 0 and
                self.board[i] == self.board[i+3] == self.board[i+6]):
                return True, self.board[i]
       
        # Diagonals
        if self.board[0] != 0 and self.board[0] == self.board[4] == self.board[8]:
            return True, self.board[0]
        if self.board[2] != 0 and self.board[2] == self.board[4] == self.board[6]:
            return True, self.board[2]
       
        # Check draw (no empty cells)
        if 0 not in self.board:
            return True, 0  # 0 => draw
        return False, None
 
    def get_reward(self, winner):
        """
        +1 if agent (player 1) wins,
        -1 if player 2 wins,
         0 if draw or no winner.
        """
        if winner == 1:
            return 1.0
        elif winner == 2:
            return -1.0
        else:
            return 0.0
 
 
# -------------------------
# 2. DQN Model Definition
# -------------------------
class DQN(nn.Module):
    """
    A simple 3-layer feed-forward network to estimate Q-values.
   
    Input:  9-dimensional board state (values in {0,1,2})
    Output: 9 Q-values (one for each possible action).
    """
    def __init__(self, state_size=9, action_size=9, hidden_size=64):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, action_size)
 
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x
 
 
# -------------------------
# 3. Replay Buffer
# -------------------------
class ReplayBuffer:
    """
    Stores experiences (state, action, reward, next_state, done) for training.
    We then sample mini-batches from this buffer to train the DQN.
    """
    def __init__(self, capacity=10000):
        self.capacity = capacity
        self.memory = []
        self.position = 0
 
    def push(self, state, action, reward, next_state, done):
        # If we haven't filled the buffer yet, append
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        # Replace the oldest experience
        self.memory[self.position] = (state, action, reward, next_state, done)
        self.position = (self.position + 1) % self.capacity
 
    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)
 
    def __len__(self):
        return len(self.memory)
 
 
# -------------------------
# 4. Training Loop
# -------------------------
def train_dqn_tictactoe(num_episodes=1000,
                        batch_size=32,
                        gamma=0.99,
                        lr=1e-3,
                        epsilon_start=1.0,
                        epsilon_end=0.1,
                        epsilon_decay=0.998):
    """
    Train a DQN agent on the TicTacToeEnv for a given number of episodes.
    Returns the trained policy network and the reward history for plotting.
    """
    env = TicTacToeEnv()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
 
    # Create the policy network and the target network
    policy_net = DQN().to(device)
    target_net = DQN().to(device)
   
    # Initialize the target network to match the policy network
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()
 
    optimizer = optim.Adam(policy_net.parameters(), lr=lr)
    replay_buffer = ReplayBuffer(capacity=10000)
 
    epsilon = epsilon_start
    all_rewards = []
 
    # Helper function for selecting actions
    def select_action(state):
        """
        Epsilon-greedy action selection:
         - With probability epsilon, pick a random action (exploration).
         - Otherwise, pick the action with the highest Q-value (exploitation).
        """
        if random.random() < epsilon:
            return random.randint(0, 8)  # 9 possible moves
        else:
            # Evaluate Q-values using the policy network
            with torch.no_grad():
                q_values = policy_net(torch.FloatTensor(state).to(device))
            q_values = q_values.cpu().numpy()
           
            # Mask invalid actions (cells already occupied)
            for idx in range(9):
                if state[idx] != 0:
                    q_values[idx] = -9999  # Make it very negative so it won't get chosen
               
            return int(np.argmax(q_values))
 
    update_target_frequency = 50  # Frequency in episodes to update target_net
 
    for episode in range(num_episodes):
        state = env.reset()
        done = False
        episode_reward = 0
       
        while not done:
            action = select_action(state)
            next_state, reward, done = env.step(action)
           
            # Store transition in the replay buffer
            replay_buffer.push(state, action, reward, next_state, done)
 
            state = next_state
            episode_reward += reward
 
            # If we have enough samples, start training
            if len(replay_buffer) >= batch_size:
                transitions = replay_buffer.sample(batch_size)
               
                # Each transition is (state, action, reward, next_state, done)
                state_batch = torch.FloatTensor([t[0] for t in transitions]).to(device)
                action_batch = torch.LongTensor([t[1] for t in transitions]).unsqueeze(1).to(device)
                reward_batch = torch.FloatTensor([t[2] for t in transitions]).to(device)
                next_state_batch = torch.FloatTensor([t[3] for t in transitions]).to(device)
                done_batch = torch.FloatTensor([t[4] for t in transitions]).to(device)
 
                # Compute current Q-values for the taken actions
                current_q_values = policy_net(state_batch).gather(1, action_batch).squeeze(1)
 
                # Compute next Q-values using the target network
                with torch.no_grad():
                    max_next_q_values = target_net(next_state_batch).max(1)[0]
                    target_q_values = reward_batch + (1 - done_batch) * gamma * max_next_q_values
 
                # Loss is MSE between current and target Q-values
                loss = nn.MSELoss()(current_q_values, target_q_values)
 
                # Gradient descent
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
 
        # Update epsilon (exploration rate) after each episode
        if epsilon > epsilon_end:
            epsilon *= epsilon_decay
 
        # Update target network periodically
        if episode % update_target_frequency == 0:
            target_net.load_state_dict(policy_net.state_dict())
 
        all_rewards.append(episode_reward)
 
        # Print progress occasionally
        if (episode+1) % 100 == 0:
            avg_last_100 = np.mean(all_rewards[-100:])
            print(f"Episode {episode+1}/{num_episodes}, Epsilon: {epsilon:.3f}, Avg Reward (last 100): {avg_last_100:.2f}")
 
    # Training done
    print("\nTraining complete!")
   
    # Plot the reward history
    plt.plot(all_rewards)
    plt.title("Tic-Tac-Toe DQN Training Rewards")
    plt.xlabel("Episode")
    plt.ylabel("Total Episode Reward")
    plt.show()
 
    return policy_net, all_rewards
 
 
# -------------------------
# 5. Human vs Agent
# -------------------------
def play_against_agent(policy_net):
    """
    Let a human play Tic-Tac-Toe against the trained agent (policy_net).
    The agent is always Player 1 (X).
    """
    env = TicTacToeEnv()
    state = env.reset()
    done = False
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
 
    print("Let's play Tic-Tac-Toe against the trained DQN agent!")
    print_board(state)
 
    while not done:
        if env.current_player == 1:
            # Agent's turn (Player 1)
            with torch.no_grad():
                q_values = policy_net(torch.FloatTensor(state).to(device)).cpu().numpy()
            # Mask invalid actions
            for idx in range(9):
                if state[idx] != 0:
                    q_values[idx] = -9999
            action = int(np.argmax(q_values))
            print(f"Agent chooses cell {action}.")
            next_state, reward, done = env.step(action)
            state = next_state
            print_board(state)
            if done:
                if reward == 1:
                    print("Agent (X) wins!")
                elif reward == 0:
                    print("It's a draw!")
                else:
                    print("Human (O) wins!")
        else:
            # Human's turn (Player 2)
            valid_move = False
            while not valid_move:
                human_action = input("Enter your move (0-8): ")
                try:
                    human_action = int(human_action)
                    if 0 <= human_action < 9 and state[human_action] == 0:
                        valid_move = True
                    else:
                        print("Invalid move. Please try again.")
                except ValueError:
                    print("Please enter a valid number between 0-8.")
 
            next_state, reward, done = env.step(human_action)
            state = next_state
            print_board(state)
            if done:
                if reward == -1:
                    print("Human (O) wins!")
                elif reward == 0:
                    print("It's a draw!")
                else:
                    print("Agent (X) wins!")
 
 
def print_board(state):
    """
    Print the board in a human-readable 3x3 grid.
    0 -> ' '
    1 -> 'X'
    2 -> 'O'
    """
    symbols = {0: ' ', 1: 'X', 2: 'O'}
    for i in range(9):
        if i % 3 != 2:
            print(f" {symbols[state[i]]} |", end="")
        else:
            print(f" {symbols[state[i]]} ")
            if i != 8:
                print("---+---+---")
    print()
import os
 
if __name__ == "__main__":
    # Choose whether to train new or load existing
    TRAIN_NEW_MODEL = True  # Set to True to train a new model
   
    if TRAIN_NEW_MODEL:
        # Train and save
        policy_net, rewards = train_dqn_tictactoe(num_episodes=5000)
       
        # Save the model state_dict
        torch.save(policy_net.state_dict(), 'tictactoe_dqn.pth')
        print("Trained model saved to tictactoe_dqn.pth")
    else:
        # Load the trained model
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        policy_net = DQN().to(device)
       
        if os.path.exists('tictactoe_dqn.pth'):
            policy_net.load_state_dict(torch.load('tictactoe_dqn.pth', map_location=device))
            policy_net.eval()
            print("Loaded trained model from tictactoe_dqn.pth")
        else:
            raise FileNotFoundError("No saved model found. Set TRAIN_NEW_MODEL=True to train first.")
 
    # Play against the agent
    play_against_agent(policy_net)