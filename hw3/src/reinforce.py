import keras
from keras import optimizers
from keras.layers import Activation
from keras import losses
from keras import metrics

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

        # TODO: Define any training operations and optimizers here, initialize
        #       your variables, or alternately compile your model here. 
        adam = optimizers.Adam(lr = 0.0001, beta_1 = 0.9, beta_2 = 0.999, epsilon = None, decay = 0.0, amsgrad = False)
        self.model.compile(loss = self.PG_loss,  # gradient loss is just XE
                           optimizer = adam,
                           metrics = [metrics.mae, metrics.categorical_accuracy]) 

    def train(self, env, gamma=1.0):
        # Trains the model on a single episode using REINFORCE.
        # TODO: Implement this method. It may be helpful to call the class
        #       method generate_episode() to generate training data.
        
        states, act_probs, act_OH, rewards = self.generate_episode(env)
        act_probs = np.squeeze(np.asarray(act_probs), axis=1)
        act_OH = np.asarray(act_OH)
        container = np.empty((0, 4))

        # iterate through to get Gt at each time step
        Gt = np.asarray([sum(rewards[i:]) for i in range (len(rewards))])

        # [n, 1] vector, the probability of taking the ideal action
        temp_sum = np.log(np.sum(act_probs*act_OH, axis=1))
        temp = Gt*temp_sum
        loss = 1/len(states)*np.sum(temp, axis=0)


        pdb.set_trace()

        # summation process
        summation = np.sum(container, axis=0)  # sum along the vertical
        loss = 1/len(rewards)*summation
        loss = tf.convert_to_tensor(loss)

        # dont know whats happening here ...
        print ('loss')
        self.model.fit(verbose=0)


    def one_hot(self, data, num_c):
        targets = data.reshape(-1)
        return np.eye(num_c)[targets]


    def PG_loss (self, input):
        return input


    def generate_episode(self, env, render=False):
        # Generates an episode by executing the current policy in the given env.
        # Returns:
        # - a list of states, indexed by time step
        # - a list of actions, indexed by time step
        # - a list of rewards, indexed by time step
        # TODO: Implement this method.
        states = []
        actions_OH = []
        actions_prob = []
        rewards = []

        done = False
        obs = env.reset()

        while (not done):

            acts = self.model.predict(np.reshape(obs, (1,8)))  # 4 float out for action
            action_chosen = np.argmax(acts)
            oh_vec = self.one_hot(action_chosen, self.num_acts)  # returns a np array in a list
            next_obs, reward, done, _ = env.step(action_chosen)

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
    
    # Load the policy model from file.
    with open(model_config_path, 'r') as f:
        model = keras.models.model_from_json(f.read())

    re = Reinforce(model, env=env, lr=0.0001)
    re.train(env)

    # TODO: Train the model using REINFORCE and plot the learning curve.


if __name__ == '__main__':
    main(sys.argv)
