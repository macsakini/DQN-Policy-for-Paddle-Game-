from collections import deque
import numpy as np
import torch
import random
from model import DQNModel
from environment import PingPong

game = PingPong()


class DQNAgent():
    def __init__(self):
        self.action_space = 3
        self.state_space = 5
        self.batch_size = 1
        self.epsilon = 1
        self.epsilon_min = .01
        self.epsilon_decay = .995
        self.gamma = .95
        self.model = DQNModel()
        self.memory = deque(maxlen=100000)

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_space)
        act_value = self.model(state)
        return np.argmax(act_value.detach().numpy())

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def replay(self):
        if len(self.memory) < self.batch_size:
            return

        minibatch = random.sample(self.memory, self.batch_size)
        states = np.array([i[0] for i in minibatch]).astype(np.float32)
        actions = np.array([i[1] for i in minibatch])
        rewards = np.array([i[2] for i in minibatch]).astype(np.float32)
        next_states = np.array([i[3] for i in minibatch]).astype(np.float32)
        dones = np.array([i[4] for i in minibatch])

        states = torch.tensor(np.squeeze(states))
        actions = torch.tensor(np.squeeze(actions))
        rewards = torch.tensor(np.squeeze(rewards))
        next_states = torch.tensor(np.squeeze(next_states))

        q_pred = self.model.forward(states)[actions]
        # print(q_pred.reshape([1]))
        # print(q_pred)

        q_next = self.model.forward(next_states).max()
        # print(q_next)

        q_target = rewards + self.gamma * q_next
        q_target = q_target.reshape([1])
        # print(q_target)

        loss = self.model.loss(q_target, q_pred)

        loss.backward()

        self.model.optimizer.step()

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay


def train(episodes):
    agent = DQNAgent()
    for ep in range(episodes):
        state = torch.tensor(game.reset())
        max_steps = 1000
        score = 0
        for i in range(0, max_steps):
            action = agent.act(state)
            reward, next_state, done = game.step(action)
            score += reward
            agent.remember(state.tolist(), action, reward, next_state, done)
            state = torch.tensor(next_state)
            agent.replay()
            if done:
                print("The model scored: ", score)
                break


train(100)
