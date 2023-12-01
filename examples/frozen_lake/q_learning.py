import numpy as np
from tqdm import tqdm

#  Following Aleksandar Haber from: https://www.youtube.com/watch?v=ZXEVznd5kaM&t=881s

#  alpha - learning rate / step size
#  gamma - discount factor
#  epsilon - probability to take random action (following epsilon-greedy approach)
class QL():
    def __init__(self, env, alpha, gamma, epsilon=None):
        self.env = env
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon

        self.policy = np.zeros(env.observation_space.n, dtype=int)
        self.Q = np.zeros((env.observation_space.n, env.action_space.n))

    
    #  Select action following epsilon-greedy policy
    def _select_action(self, state, k):
        if self.epsilon is None:
            if k == 0:
                k = 1
            epsilon =  1/k
        else:
            epsilon = self.epsilon
        rand = np.random.uniform(size=1)

        if rand <= epsilon:
            return np.random.randint(0, self.env.action_space.n)
        
        return self.Q[state].argmax()


    #  Calculates the policy by a greedy policy
    def _determine_policy(self):
        for state in range(self.env.observation_space.n):
            self.policy[state] = self.Q[state].argmax()


    #  S - state, A - action, R - reward, S - state_1, A - action_1
    def train(self, num_episodes):
        loop = tqdm(range(num_episodes), total=num_episodes, position=0, leave=False)
        for k in loop:
            loop.set_description(f"Episode {k+1}/{num_episodes}")
            state = self.env.reset()[0]
            action = self._select_action(state, k)

            terminated, truncated = False, False
            while not terminated and not truncated:
                (state_1, reward, terminated, truncated, _) = self.env.step(action)
                reward -= 0.01
                if terminated:
                    reward -= 0.2

                action_1 = self._select_action(state_1, k)
                #  Odd implementation of SARSA update formula
                #  This is more indicative of an error
                err = self.Q[state, action] - reward
                #  Due to implementation of envs, we must make the condition that 'rewards after done = 0' explicit
                if not terminated:
                    err -= self.gamma * self.Q[state_1,:].max()
                self.Q[state, action] -= self.alpha*err

                state = state_1
                action = action_1
        
        self._determine_policy()


    def act(self, state):
        return self.policy[state]

        