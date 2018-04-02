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

    def __init__(self, model, env, lr=0.0001):
        self.model = model
        self.env = env
        self.num_obs = self.env.observation_space.shape[0]
        self.num_acts = 4
        self.lr = lr
        self.layers = [8, 16, 16, 4]

        self.buildModel()


    def buildModel(self):
        self.input_state = tf.placeholder(tf.float32, [None, self.num_obs],
                                          name='input_state')

        layer = self.input_state
        for i in range(len(self.layers) - 1):
            layer = tf.layers.dense(layer, self.layers[i], tf.nn.relu, name = 'FC_Layer_' + str(i))

        self.output_prob = tf.layers.dense(layer, self.layers[-1], tf.nn.softmax, name='Softmax_Layer')
        self.rewards = tf.placeholder(tf.float32, shape=(None))
        self.actions = tf.placeholder(tf.float32, shape=(None, self.num_acts))
        self.loss = tf.reduce_mean(self.rewards * -tf.log(tf.reduce_sum(self.actions * self.output_prob, axis = 1)))
        self.train_op = tf.train.AdamOptimizer(self.lr).minimize(self.loss)

        self.writer = tf.summary.FileWriter("logs", graph=tf.get_default_graph())
        self.sess = tf.InteractiveSession()
        self.saver = tf.train.Saver()
        self.sess.run(tf.initialize_all_variables())

    def trainModel(self, input, actions, rewards):
        _, loss = self.sess.run([self.train_op, self.loss],
                      feed_dict={self.input_state: input, self.actions: actions, self.rewards: rewards})

        return loss

    def predict(self, input):
        return self.sess.run(self.output_prob,
                      feed_dict={self.input_state: input})

    def train(self, env, gamma=1.0):
        for i in range(200000):
            states, act_probs, act_OH, rewards = self.generate_episode(env)
            act_OH = np.asarray(act_OH)

            loss = self.trainModel(states, act_OH, rewards)
            summary = tf.Summary(value=[
                        tf.Summary.Value(tag="Loss", simple_value=loss),
                    ])
            self.writer.add_summary(summary, i)
            #
            summary = tf.Summary(value=[
                        tf.Summary.Value(tag="Training reward", simple_value=rewards[0]),
                    ])
            self.writer.add_summary(summary, i)


    def one_hot(self, data, num_c):
        targets = data.reshape(-1)
        return np.eye(num_c)[targets]

    def generate_episode(self, env, render=False):
        states = []
        actions_OH = []
        actions_prob = []
        rewards = []

        done = False
        obs = env.reset()

        while not done:

            acts = self.predict(np.reshape(obs, (1,8)))  # 4 float out for action
            action_chosen = np.random.choice(4, 1, p = np.squeeze(acts))
            oh_vec = self.one_hot(action_chosen, self.num_acts)  # returns a np array in a list
            next_obs, reward, done, _ = env.step(np.squeeze(action_chosen))

            states.append(obs)
            actions_prob.append(acts)
            actions_OH.append(oh_vec[0].astype(int))
            rewards.append(reward)

            obs = next_obs

        return states, actions_prob, actions_OH, np.cumsum(rewards[::-1])[::-1]


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
    
    # Load the policy model from file.
    # with open(model_config_path, 'r') as f:
    #     model = keras.models.model_from_json(f.read())

    re = Reinforce(None, env=env, lr=0.0005)
    re.train(env)

    # TODO: Train the model using REINFORCE and plot the learning curve.


if __name__ == '__main__':
    main(sys.argv)
