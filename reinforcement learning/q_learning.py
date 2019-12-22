from environment import MountainCar
import sys
import random
import numpy as np
import copy
import matplotlib.pyplot as pl
class LinearQNetwork(object):
    # Linear approximation of Q network
    # (a) initialize a DQN object that contains all necessary parameters needed to calculate q()
    # (b) a class method that calculates q values given input state
    # (c) a class method that updates the paramenters of DQN
    def __init__(self, state, state_space):
        #initialize Q-learning network
        self.state_space = state_space
        self.w = np.zeros((self.state_space,3)) # |S| X |A|
        self.b = 0.0
        return

    def evaluate(self, s, a):
        # This function should calculate q(s, a; w)
        # The extra input(s) depend on design choice, should return q(s,a;w) for a given state s and all actions a
        res = 0.0
        for key,val in s.items():
            res += self.w[key,a] * val
        return res + self.b

    def update(self, learning_rate, cur_s, action, reward, gamma,max_q):
        # update the network parameters
        # What are the inputs? Those should be the quantities that are not included in the class attribute
        self.TDtarget = reward + gamma * max_q
        self.TDerror = self.evaluate(cur_s,action) - self.TDtarget
        for key,val in cur_s.items():
            self.w[key,action] -= learning_rate * self.TDerror * val
        self.b -= learning_rate * self.TDerror
        return

class Agent(object):
    # Actual Q learning agent
    # (a) implement the epsilon-greedy policy
    # (b) train the DQN network and the agent
    def __init__(self, mode,epsilon,max_iter,gamma,learning_rate, episodes):
        self.mode = mode
        self.epsilon = epsilon
        self.gamma = gamma
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.episodes = episodes
        self.car = MountainCar(mode)
        self.network = LinearQNetwork(self.car.state,self.car.state_space)
        self.out = []
        pass

    def policy(self, q_value_array):
        # Given an input array of q values, return the action selected by epsilon-greedy policy
        prob = random.uniform(0,1)
        action = 0
        if prob < self.epsilon:
            action = random.choice([0,1,2])
        else:
            action = np.argmax(q_value_array)
        return action

    def train(self):
        # train agent and the linear Q network and return the returns from all training episodes
        for ep in range(self.episodes):
            self.one_out = 0.0
            self.next_state = copy.deepcopy(self.car.reset())
            for i in range(self.max_iter):
                q_value_array = []
                self.cur_state = copy.deepcopy(self.next_state)
                for a in [0,1,2]:
                    q_value_array.append(self.network.evaluate(self.cur_state,a))

                # choose action based on current state and 3 options
                action = self.policy(q_value_array)
                # make a step
                self.step_res = self.car.step(action)
                self.next_state = copy.deepcopy(self.step_res[0])
                reward = self.step_res[1]
                done = self.step_res[2]
                self.one_out += reward

                max_q = []
                for a_prime in [0, 1, 2]:
                    # s_prime = self.car.state after a step
                    q_sp_ap = self.network.evaluate(self.next_state,a_prime)
                    max_q.append(q_sp_ap)
                self.network.update(self.learning_rate, self.cur_state, action, reward,
                                    self.gamma, max(max_q))
                if done: # terminal state, break current episode
                    break


            self.out.append(self.one_out)
        return self.out

def main(args):
    # parse input
    mode = args[1]
    weight_out = args[2]
    returns_out = args[3]
    episodes = int(args[4])#one episode is a sequence of [states,actions,rewards]
    max_iter = int(args[5])
    epsilon = float(args[6])
    gamma = float(args[7])
    learning_rate = float(args[8])
    #instantiate environment, qnetwork, agent
    test = Agent(mode,epsilon,max_iter,gamma,learning_rate, episodes)
    #train model
    output = test.train()
    #generate output files
    with open(weight_out,'w') as f:
        f.write(str(test.network.b)+'\n')
        for r in test.network.w:
            for c in r:
                f.write(str(c) + '\n')

    with open(returns_out,'w') as f:
        for r in output:
                f.write(str(r) + '\n')

    test = Agent('tile', epsilon, max_iter, gamma, learning_rate, episodes)
    # train model
    output2 = test.train()

    pl.plot(range(episodes), output, label="raw")
    pl.plot(range(episodes), output2, label= 'tiles')
    pl.xlabel("#episode")
    pl.ylabel("sum of all rewards in an episode")
    pl.legend(loc='upper right')
    pl.show()
if __name__ == "__main__":
    print('hello world')
    main(sys.argv)