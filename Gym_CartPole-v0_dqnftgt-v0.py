#!/home/aboul/anaconda3/envs/openai/bin/python

'''
aboul@free.fr
starter from Keon Kim blog: Deep Q-Learning with Keras and Gym CartPole

prerequisites
create /openai/save  in HOME directory
'''

import os.path
import time
import random
import gym
import numpy as np
import collections
import keras
import matplotlib.pyplot as plt
import sys

class DQNAgent(object):
    def __init__(self, state_size, action_size,
                 memory_replay_size, batch_size,
                 exploration_rate, exploration_rate_decay, exploration_rate_min,
                 discount_rate,
                 learning_rate):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = collections.deque(maxlen=memory_replay_size)  # 2000
        self.gamma = discount_rate  # 0.95
        self.epsilon = exploration_rate  # 1 exploration rate (epsilon greedy)
        self.epsilon_min = exploration_rate_min
        self.epsilon_decay = exploration_rate_decay  # 0.995
        self.alpha = learning_rate  # 0.001
        self.batch_size = batch_size  # 32
        self.model = self.build_model()
        self.update_model_tgt()

    def build_model(self):
        # Neural Net for Deep-Q learning Model
        model = keras.models.Sequential()
        model.add(keras.layers.Dense(24, input_dim=self.state_size, activation='relu'))
        model.add(keras.layers.Dense(24, activation='relu'))
        model.add(keras.layers.Dense(self.action_size, activation='linear'))
        model.compile(loss='mse',
                      optimizer=keras.optimizers.Adam(lr=self.alpha))
        return model

    def record_in_replay_memory(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def do_act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(start=0 ,
                                    stop=self.action_size,
                                    step=1)  # just to properly break down ;-)
        act_values = self.model.predict(state)
        return np.argmax(act_values[0])  # returns action indice

    def replay(self):
        mini_batch = random.sample(self.memory, self.batch_size)
        states, targets_full = [], []  # init arrays
        for state, action, reward, next_state, done in mini_batch:
            target = reward  # if finished
            if not done:
                target = (reward + self.gamma *
                          np.amax(self.model.predict(next_state)[0]))  # return max value
            target_full = self.model_tgt.predict(state)
            target_full[0][action] = target  # no loss on other actions ;-)
            # Filtering out states and targets for training
            states.append(state[0])
            targets_full.append(target_full[0])
        history = self.model.fit(np.array(states), np.array(targets_full),
                                 epochs=1, verbose=0)
        # Keeping track of loss
        loss = np.mean(history.history['loss'][0:self.batch_size])
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        return loss

    def load(self, name):
        self.model.load_weights(name)
        self.update_model_tgt()

    def save(self, name):
        self.model.save_weights(name)

    def update_model_tgt(self):
        self.model_tgt = self.model

if __name__ == "__main__":
    ENV = 'CartPole-v0'
    rewards = []
    random.seed(72)
    DIR_FULL = os.getenv('HOME') + '/openai/save/'
    env = gym.make(ENV)
    MODEL_NM = 'cartpole-dqn'
    MODEL_DATE_REF = time.strftime("%Y%m%d-%H%M%S")
    MODE_CONTINUE = os.path.isfile(DIR_FULL + MODEL_NM + '.h5')
    MODE_TRAIN = 1
    MODE_RENDER = 0
    MODE_VERBOZE = 1  # 0 off 1 base 2 detailed
    EPISODES_NB = 150  # was 1000
    EPISODES_TIME = 200
    STEPS_TO_UPDT_MOD_TGT = 1000
    STATE_SIZE = env.observation_space.shape[0]
    ACTION_SIZE = env.action_space.n
    memory_replay_size = 2000
    batch_size = 2 ** 5
    exploration_rate = 1.0
    exploration_rate_decay = 0.995  # nb: 1/(1-B)
    exploration_rate_min = 0.01
    discount_rate = 0.90  # nb: 1/(1-B)
    learning_rate = 0.001 # was 0.001
    agent = DQNAgent(state_size=STATE_SIZE, action_size=ACTION_SIZE,
         memory_replay_size=memory_replay_size, batch_size=batch_size,
         exploration_rate=exploration_rate, exploration_rate_decay=exploration_rate_decay,
         exploration_rate_min=exploration_rate_min,
         discount_rate=discount_rate,
         learning_rate=learning_rate)
    GRAPH_Y_ORIGIN = 10
    GRAPH_X_ORIGIN = 70
    def academy_reward(state, reward, done, step, max_dur):
        reward = -10 if done and step < (max_dur - 1) else reward
        return reward
    def academy_score(state, reward, done, step, max_dur):
        score = step
        return score

    if MODE_CONTINUE: agent.load(DIR_FULL + MODEL_NM + '.h5')
    done = False

    steps_ref = 0
    for episode in range(EPISODES_NB):
        state = env.reset()
        state = np.reshape(state, [1, STATE_SIZE])
        for step in range(EPISODES_TIME):
            if MODE_RENDER: env.render()
            action = agent.do_act(state)
            steps_ref += 1
            next_state, reward, done, _ = env.step(action)
            next_state = np.reshape(next_state, [1, STATE_SIZE])
            if MODE_TRAIN:
                reward = academy_reward(state=next_state, reward=reward,
                                        done=done, step=step,
                                        max_dur=EPISODES_TIME)
                agent.record_in_replay_memory(state, action, reward, next_state, done)
                if steps_ref % STEPS_TO_UPDT_MOD_TGT == 0:
                    agent.update_model_tgt()
                    if MODE_VERBOZE >= 2: print('#tgt mdl updted ;-)')
            state = next_state
            if done:
                score = academy_score(state=next_state, reward=reward,
                                      done=done, step=step,
                                      max_dur=EPISODES_TIME)
                rewards.append((episode, score))
                if MODE_VERBOZE >= 1:
                    print("END episode: {}/{}, score: {}, epsil_greedy: {:.2}, total steps:{}, rew:{}"\
                        .format(episode, EPISODES_NB, score, agent.epsilon, steps_ref, reward))
                break
            if len(agent.memory) > batch_size and MODE_TRAIN:
                loss = agent.replay()
                # Logging training loss every 10 timesteps
                if step % 10 == 0 and MODE_VERBOZE >= 2:
                    print("episode: {}/{}, step: {}, mean loss mini batch: {:.4f}"\
                          .format(episode, EPISODES_NB, step, loss))
        if episode % 10 == 0 and MODE_TRAIN:
            agent.save(DIR_FULL + MODEL_NM + '-' + MODEL_DATE_REF + '.h5')
            file_object = open(DIR_FULL + MODEL_NM + '-' + MODEL_DATE_REF + '.txt', 'w')
            file_object.write('ENV: ' + ENV + '\n')
            file_object.write('MODEL_NM: ' + MODEL_NM + '\n')
            file_object.write('MODE_TRAIN: ' + str(MODE_TRAIN) + '\n')
            file_object.write('MODE_RENDER: ' + str(MODE_RENDER) + '\n')
            file_object.write('MODE_VERBOZE: ' + str(MODE_VERBOZE) + '\n')
            file_object.write('EPISODES_NB: ' + str(EPISODES_NB) + '\n')
            file_object.write('EPISODES_TIME: ' + str(EPISODES_TIME) + '\n')
            file_object.write('STEPS_TO_UPDT_MOD_TGT: ' + str(STEPS_TO_UPDT_MOD_TGT) + '\n')
            file_object.write('STATE_SIZE: ' + str(STATE_SIZE) + '\n')
            file_object.write('ACTION_SIZE: ' + str(ACTION_SIZE) + '\n')
            file_object.write('memory_replay_size: ' + str(memory_replay_size) + '\n')
            file_object.write('batch_size: ' + str(batch_size) + '\n')
            file_object.write('epsil_greedy: ' + str(agent.epsilon) + '\n')
            file_object.write('exploration_rate_decay: ' + str(exploration_rate_decay) + '\n')
            file_object.write('exploration_rate_min: ' + str(exploration_rate_min) + '\n')
            file_object.write('discount_rate: ' + str(discount_rate) + '\n')
            file_object.write('learning_rate: ' + str(learning_rate) + '\n')
            file_object.close()

    env.close()
    episode_plot, score_plot = np.array(rewards).T
    # plot reward v.s. episode
    plt.plot(episode_plot, score_plot)
    plt.xlabel('episode')
    plt.ylabel('score')
    plt.text(GRAPH_X_ORIGIN, GRAPH_Y_ORIGIN + 80, s='ENV: ' + str(ENV), fontsize=8)
    plt.text(GRAPH_X_ORIGIN, GRAPH_Y_ORIGIN + 70, s='MODEL_NM: ' + str(MODEL_NM), fontsize=8)
    plt.text(GRAPH_X_ORIGIN, GRAPH_Y_ORIGIN + 60, s='STEPS_TO_UPDT_MOD_TGT: ' + str(STEPS_TO_UPDT_MOD_TGT), fontsize=8)
    plt.text(GRAPH_X_ORIGIN, GRAPH_Y_ORIGIN + 50, s='memory_replay_size: ' + str(memory_replay_size), fontsize=8)
    plt.text(GRAPH_X_ORIGIN, GRAPH_Y_ORIGIN + 40, s='batch_size: ' + str(batch_size), fontsize=8)
    plt.text(GRAPH_X_ORIGIN, GRAPH_Y_ORIGIN + 30, s='epsil_greedy: ' + str(agent.epsilon), fontsize=8)
    plt.text(GRAPH_X_ORIGIN, GRAPH_Y_ORIGIN + 20, s='discount_rate: ' + str(discount_rate), fontsize=8)
    plt.text(GRAPH_X_ORIGIN, GRAPH_Y_ORIGIN + 10, s='learning_rate: ' + str(learning_rate), fontsize=8)
    plt.savefig(fname=DIR_FULL + MODEL_NM + '-' + MODEL_DATE_REF + '.png')
    plt.show()
