import torch
import torch.nn as nn
import random
from tqdm import tqdm
import pickle

from dqn_memory import initial_solution, permute_solution
from plot import make_plot


class DQNSolver(nn.Module):
    def __init__(self, input_shape, num_actions):
        super().__init__()

        self.input_shape = input_shape
        self.num_actions = num_actions

        self.net = nn.Sequential(
            nn.Linear(self.input_shape, 124),
            nn.ReLU(),
            nn.Linear(124, self.num_actions))

    def forward(self, x):
        return self.net(x)


class DQNAgent:

    def __init__(self, state_space, action_space, max_memory_size, batch_size, gamma, lr,
                 exploration_max, exploration_min, exploration_decay, pretrained, target_update_freq):

        # Define DQN Layers
        self.state_space = state_space
        self.action_space = action_space
        self.pretrained = pretrained
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        # DQN network (target net) and online net
        self.dqn = DQNSolver(state_space, action_space).to(self.device)
        self.online_net = DQNSolver(state_space, action_space).to(self.device)
        self.dqn.load_state_dict(self.online_net.state_dict())

        if self.pretrained:
            self.dqn.load_state_dict(torch.load("DQN.pt", map_location=torch.device(self.device)))
        self.optimizer = torch.optim.Adam(self.dqn.parameters(), lr=lr)

        # Create memory
        self.max_memory_size = max_memory_size
        if self.pretrained:
            self.STATE_MEM = torch.load("STATE_MEM.pt")
            self.ACTION_MEM = torch.load("ACTION_MEM.pt")
            self.REWARD_MEM = torch.load("REWARD_MEM.pt")
            self.STATE2_MEM = torch.load("STATE2_MEM.pt")
            with open("ending_position.pkl", 'rb') as f:
                self.ending_position = pickle.load(f)
            with open("num_in_queue.pkl", 'rb') as f:
                self.num_in_queue = pickle.load(f)
        else:
            self.STATE_MEM = torch.zeros(max_memory_size, self.state_space)
            self.ACTION_MEM = torch.zeros(max_memory_size, 1)
            self.REWARD_MEM = torch.zeros(max_memory_size, 1)
            self.STATE2_MEM = torch.zeros(max_memory_size, self.state_space)
            self.ending_position = 0
            self.num_in_queue = 0

        self.memory_sample_size = batch_size

        # Learning parameters
        self.gamma = gamma
        self.lr = lr
        self.l1 = nn.SmoothL1Loss().to(self.device)  # Also known as Huber loss
        self.exploration_max = exploration_max
        self.exploration_rate = exploration_max
        self.exploration_min = exploration_min
        self.exploration_decay = exploration_decay
        self.target_update_freq = target_update_freq
        self.loss_result = []

    def remember(self, state, action, reward, state2):
        """Store the experiences in a buffer to use later"""
        self.STATE_MEM[self.ending_position] = state.float()
        self.ACTION_MEM[self.ending_position] = action.float()
        self.REWARD_MEM[self.ending_position] = reward.float()
        self.STATE2_MEM[self.ending_position] = state2.float()
        self.ending_position = (self.ending_position + 1) % self.max_memory_size  # FIFO tensor
        self.num_in_queue = min(self.num_in_queue + 1, self.max_memory_size)

    def batch_experiences(self):
        """Randomly sample 'batch size' experiences"""
        idx = random.choices(range(self.num_in_queue), k=self.memory_sample_size)
        STATE = self.STATE_MEM[idx]
        ACTION = self.ACTION_MEM[idx]
        REWARD = self.REWARD_MEM[idx]
        STATE2 = self.STATE2_MEM[idx]
        return STATE, ACTION, REWARD, STATE2

    def act(self, state):
        """Epsilon-greedy action"""
        if random.random() < self.exploration_rate:
            return torch.tensor([[random.randrange(self.action_space)]])
        else:
            return torch.argmax(self.dqn(state.to(self.device))).unsqueeze(0).unsqueeze(0).cpu()

    def experience_replay(self, iteration):
        if self.memory_sample_size > self.num_in_queue:
            return

        # Sample a batch of experiences
        STATE, ACTION, REWARD, STATE2 = self.batch_experiences()
        STATE = STATE.to(self.device)
        ACTION = ACTION.to(self.device)
        REWARD = REWARD.to(self.device)
        STATE2 = STATE2.to(self.device)

        self.optimizer.zero_grad()
        # Q-Learning target is Q*(S, A) <- r + γ max_a Q(S', a)
        target = REWARD + self.gamma * self.dqn(STATE2).max(1).values.unsqueeze(1)
        current = self.online_net(STATE).gather(1, ACTION.long())

        loss = self.l1(current, target)
        self.loss_result.append(loss.item())

        loss.backward()  # Compute gradients
        self.optimizer.step()  # Back propagate error

        self.exploration_rate *= self.exploration_decay

        # Makes sure that exploration rate is always at least 'exploration min'
        self.exploration_rate = max(self.exploration_rate, self.exploration_min)

        # updating the target network
        if iteration % self.target_update_freq == 0:
            self.dqn.load_state_dict(self.online_net.state_dict())


def run(training_mode, pretrained, episode_size=1000, num_episodes=10000, exploration_max=1, data='taillard', instance_number=0):
    observation_space = 2
    action_space = 3
    agent = DQNAgent(state_space=observation_space,
                     action_space=action_space,
                     max_memory_size=30000,
                     batch_size=32,
                     gamma=0.999,
                     lr=0.00001,
                     exploration_max=1.0,
                     exploration_min=0.02,
                     exploration_decay=0.996,
                     pretrained=pretrained,
                     target_update_freq=500)

    num_episodes = num_episodes
    makespan_results = []
    rewards = []

    state = [1, 0]
    state = torch.Tensor([state])
    current_solution, best_makespan = initial_solution(dataset=data, inst_number=instance_number)
    makespan_results.append(best_makespan)
    best_solution = current_solution

    for i in range(num_episodes):
        print("Initialize Episode: {}".format(i))
        sum_reward = 0

        for ep_num in tqdm(range(episode_size)):

            action = agent.act(state)
            state_next, reward, next_solution, next_makespan = permute_solution(int(action[0]), current_solution)
            makespan_results.append(next_makespan)
            sum_reward += reward

            if next_makespan <= best_makespan:
                best_makespan = next_makespan
                best_solution = next_solution

            state_next = torch.Tensor([state_next])
            reward = torch.tensor([reward]).unsqueeze(0)

            if training_mode:
                agent.remember(state, action, reward, state_next)
                agent.experience_replay(ep_num)

            state = state_next
            current_solution = next_solution

            if ep_num != 0 and ep_num % 100 == 0:
                print("In Episode {} Iteration {} best make-span found is = {}".format(i, ep_num + 1, best_makespan))

        rewards.append(sum_reward)
        current_solution = best_solution
        state = [0,1]
        state = torch.Tensor([state])

        if i != 0 and i % 1000 == 0:

            print(" Plot makespan, reward, and loss each 100 episodes")
            print(" Episode {} best make-span found is = {}".format(i, min(makespan_results)))
            make_plot(makespan_results, 'makespan')
            make_plot(rewards, 'rewards')
            make_plot(agent.loss_result, 'loss')




        print("Episode {} best make-span found is = {}".format(i, best_makespan))
    print("Best make-span found in all episodes is = {}".format(min(makespan_results)))

    # plot the results
    print(" Plot makespan, reward, and loss for all episodes")
    make_plot(makespan_results, 'makespan')
    make_plot(rewards, 'rewards')
    make_plot(agent.loss_result, 'loss')



if __name__ == "__main__":
    run(training_mode=True, pretrained=False, episode_size=1000, num_episodes=10000, exploration_max=1, data='taillard', instance_number=46)

