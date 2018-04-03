import keras
from keras import optimizers
from keras.layers import Activation
from keras import losses
from keras import metrics
from keras import backend as k
import sys
import argparse
import numpy as np
import tensorflow as tf
import gym
import pdb

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


class Reinforce(object):
    # Implementation of the policy gradient method REINFORCE.

    def __init__(self, model, env, lr):
        self.model = model
        self.env = env
        self.num_obs = self.env.observation_space.shape[0]
        self.num_acts = self.env.action_space.n
        self.lr = lr
        self.layers = [self.num_obs, 16, 16, self.num_acts]
        self.buildModel()
        
        
    def buildModel(self):
        self.input_state = tf.placeholder(tf.float32, [None, self.num_obs],
                                          name='input_state')

        layer = self.input_state
        for i in range(len(self.layers) - 1):
            layer = tf.layers.dense(layer, self.layers[i], tf.nn.relu, name = 'FC_Layer_' + str(i))

        self.output_prob = tf.layers.dense(layer, self.layers[-1], tf.nn.softmax, name='Softmax_Layer')
        tf.summary.histogram('Action Probabilities', self.output_prob)

        self.rewards = tf.placeholder(tf.float32, shape=(None))
        self.actions = tf.placeholder(tf.float32, shape=(None, self.num_acts))
        self.loss = tf.reduce_mean(self.rewards * -tf.log(tf.reduce_sum(self.actions * self.output_prob, axis = 1)))
        self.train_op = tf.train.AdamOptimizer(self.lr).minimize(self.loss)

        self.writer = tf.summary.FileWriter("logs2", graph=tf.get_default_graph())
        self.sess = tf.InteractiveSession()
        self.saver = tf.train.Saver()
        self.sess.run(tf.initialize_all_variables())
        self.merge_op = tf.summary.merge_all()

    def trainModel(self, input, actions, rewards):
        _, loss, merge = self.sess.run([self.train_op, self.loss, self.merge_op],
                      feed_dict={self.input_state: input, self.actions: actions, self.rewards: rewards})

        return loss, merge

    def predict(self, input):
        return self.sess.run(self.output_prob,
                      feed_dict={self.input_state: input})

    def train(self, num_episodes, render, gamma=1.0):
        for i in range(num_episodes):
            states, act_probs, act_OH, rewards = self.generate_episode(render)
            c_rewards = np.cumsum(rewards[::-1])[::-1]
            act_OH = np.asarray(act_OH)

            loss, merge = self.trainModel(states, act_OH, c_rewards)

            self.writer.add_summary(merge, i)
            summary = tf.Summary(value=[
                        tf.Summary.Value(tag="Loss", simple_value=loss),
                    ])
            self.writer.add_summary(summary, i)
            #
            summary = tf.Summary(value=[
                        tf.Summary.Value(tag="Training reward", simple_value=c_rewards[0]),
                    ])
            self.writer.add_summary(summary, i)

            if i % 1000:
                self.saver.save(self.sess, "save/REINFORCE_" + str(i))

    def one_hot(self, data, num_c):
        targets = data.reshape(-1)
        return np.eye(num_c)[targets]

    def generate_episode(self, render=False):
        states = []
        actions_OH = []
        actions_prob = []
        rewards = []

        done = False
        obs = self.env.reset()

        while not done:
            if render:
                self.env.render()
            acts = self.predict(np.reshape(obs, (1,self.num_obs)))  # 4 float out for action
            action_chosen = np.random.choice(self.num_acts, 1, p = np.squeeze(acts))
            oh_vec = self.one_hot(action_chosen, self.num_acts)  # returns a np array in a list
            next_obs, reward, done, _ = self.env.step(np.squeeze(action_chosen))

            states.append(obs)
            actions_prob.append(acts)
            actions_OH.append(oh_vec[0].astype(int))
            rewards.append(reward)

            obs = next_obs

        return states, actions_prob, actions_OH, rewards


def parse_arguments():
    # Command-line flags are defined here.
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-config-path', dest='model_config_path',
                        type=str, default='LunarLander-v2-config.json',
                        help="Path to the model config file.")
    parser.add_argument('--num-episodes', dest='num_episodes', type=int,
                        default=50000, help="Number of episodes to train on.")
    parser.add_argument('--lr', dest='lr', type=float,
                        default=5e-4, help="The learning rate.")

    # https://stackoverflow.com/questions/15008758/parsing-boolean-values-with-argparse
    parser_group = parser.add_mutually_exclusive_group(required=False)
    parser_group.add_argument('--render', dest='render',
                              action='store_true',
                              help="Whether to render the environment.")
    parser_group.add_argument('--no-render', dest='render',
                              action='store_false',
                              help="Whether to render the environment.")
    parser.set_defaults(render=False)

    return parser.parse_args()


def main(args):
    # Parse command-line arguments.
    args = parse_arguments()
    model_config_path = args.model_config_path
    num_episodes = args.num_episodes
    lr = args.lr
    render = args.render

    # Create the environment.
    env = gym.make('LunarLander-v2')
    #env = gym.make('CartPole-v0')
    # Load the policy model from file.
    # with open(model_config_path, 'r') as f:
    #     model = keras.models.model_from_json(f.read())

    re = Reinforce(None, env, 0.001)
    re.train(num_episodes, render)

    # TODO: Train the model using REINFORCE and plot the learning curve.


if __name__ == '__main__':
    main(sys.argv)
