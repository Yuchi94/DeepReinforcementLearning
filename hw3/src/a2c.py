import sys
import argparse
import numpy as np
import tensorflow as tf
import keras
import gym

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from reinforce import Reinforce


class A2C(Reinforce):
    # Implementation of N-step Advantage Actor Critic.
    # This class inherits the Reinforce class, so for example, you can reuse
    # generate_episode() here.

    def __init__(self, env, actor_lr, critic_lr, n):

        self.env = env
        self.actor_lr = actor_lr
        self.critic_lr = critic_lr
        self.n = n
        self.num_obs = self.env.observation_space.shape[0]
        self.num_acts = self.env.action_space.n
        self.a_layers = [self.num_obs, 16, 16, self.num_acts]
        self.c_layers = [self.num_obs, 100, 100, 50, 1]

        self.buildActorModel()
        self.buildCriticModel()

        self.writer = tf.summary.FileWriter("logs", graph=tf.get_default_graph())
        self.sess = tf.InteractiveSession()
        self.saver = tf.train.Saver()
        self.sess.run(tf.initialize_all_variables())
        self.merge_op = tf.summary.merge_all()

    def buildCriticModel(self):
        with tf.variable_scope("Critic"):
            self.critic_input = tf.placeholder(tf.float32, [None, self.num_obs],
                                              name='input_state')
            regularizer = tf.contrib.layers.l2_regularizer(scale=0.001)

            layer = self.critic_input
            for i in range(len(self.c_layers) - 1):
                layer = tf.layers.dense(layer, self.c_layers[i], tf.nn.relu, kernel_regularizer=regularizer, name = 'FC_Layer_' + str(i))

            self.state_values = tf.layers.dense(layer, self.c_layers[-1], kernel_regularizer=regularizer, name='Output_Layer')
            self.state_summary = tf.summary.histogram('State Values', self.state_values)

            self.critic_rewards = tf.placeholder(tf.float32, shape=(None))
            self.critic_loss = tf.reduce_mean(tf.square(self.state_values - self.critic_rewards))
            #self.critic_loss = tf.losses.mean_squared_error(self.state_values, self.critic_rewards)

        self.critic_train_op = tf.train.AdamOptimizer(self.critic_lr).minimize(self.critic_loss,
                                                                   var_list = tf.trainable_variables('Critic'))

    def buildActorModel(self):
        with tf.variable_scope("Actor"):
            self.actor_input = tf.placeholder(tf.float32, [None, self.num_obs],
                                              name='input_state')

            layer = self.actor_input
            for i in range(len(self.a_layers) - 1):
                layer = tf.layers.dense(layer, self.a_layers[i], tf.nn.relu, name = 'FC_Layer_' + str(i))

            self.output_prob = tf.layers.dense(layer, self.a_layers[-1], tf.nn.softmax, name='Softmax_Layer')
            self.action_summary = tf.summary.histogram('Action Probabilities', self.output_prob)

            self.actor_rewards = tf.placeholder(tf.float32, shape=(None))
            self.actor_actions = tf.placeholder(tf.float32, shape=(None, self.num_acts))
            self.actor_state_values = tf.placeholder(tf.float32, shape=(None))
            self.actor_loss = tf.reduce_mean((self.actor_rewards - self.actor_state_values) *
                                             -tf.log(tf.reduce_sum(self.actor_actions * self.output_prob, axis = 1)))

        self.actor_train_op = tf.train.AdamOptimizer(self.actor_lr).minimize(self.actor_loss,
                                                                     var_list = tf.trainable_variables('Actor'))

    def getStateValues(self, input):
        return self.sess.run(self.state_values,
                      feed_dict={self.critic_input: input})

    def getActionProb(self, input):
        return self.sess.run(self.output_prob,
                      feed_dict={self.actor_input: input})

    def predict(self, input):
        #wrapper for getActionProb
        return self.getActionProb(input)

    def getCummRewards(self, values, rewards, gamma):
        v_end = np.zeros_like(values)
        rt = np.zeros_like(rewards)
        T = len(rewards)
        for t in range(T)[::-1]:
            v_end = 0 if t + self.n >= T else values[t + self.n]
            r_sum = 0
            for k in range(self.n):
                if t + k < T:
                    r_sum += np.power(gamma, k) * rewards[t+k]
                else:
                    break
            rt[t] = np.power(gamma, self.n) * v_end + r_sum
            #rt[t] = np.power(gamma, self.n) * v_end + \
            #np.sum([np.power(gamma, k) * (rewards[t + k] if t + k < T else 0) 
            #for k in range(self.n)])

        return rt

        #
        # #TODO: gamma not used. Implement if required
        # v_end = np.pad(np.squeeze(values), ((0, self.n)), mode = 'constant')[self.n:]
        # # v_end[-self.n:] = 0
        #
        # c_rewards = np.cumsum(rewards)
        # shift_reward = np.pad(c_rewards, ((0, self.n)), mode = 'edge')[self.n:]
        # rt = v_end + shift_reward - c_rewards + rewards
        #
        # return rt
        # 1111111
        # 2345677
        # 1234567

    def trainActor(self, input, actions, rewards, state_values):
        _, loss, summary = self.sess.run([self.actor_train_op, self.actor_loss, self.action_summary],
        feed_dict={self.actor_input: input, self.actor_actions: actions, self.actor_rewards: rewards, self.actor_state_values : state_values})

        return loss, summary

    def trainCritic(self, input, rewards):
        _, loss, summary = self.sess.run([self.critic_train_op, self.critic_loss, self.state_summary],
        feed_dict={self.critic_input: input, self.critic_rewards: rewards})

        return loss, summary


    def train(self, num_episodes, gamma=1.0):
        for i in range(num_episodes):
            states, act_probs, act_OH, rewards = self.generate_episode(False)
            values = self.getStateValues(states)
            c_rewards = self.getCummRewards(values, rewards, gamma) / 100

            act_OH = np.asarray(act_OH)

            loss, summary = self.trainActor(states, act_OH, c_rewards, values)
            self.writer.add_summary(summary, i)

            summary = tf.Summary(value=[tf.Summary.Value(tag="Actor Loss", simple_value=loss),])
            self.writer.add_summary(summary, i)

            loss, summary = self.trainCritic(states, c_rewards)
            self.writer.add_summary(summary, i)

            summary = tf.Summary(value=[tf.Summary.Value(tag="Critic Loss", simple_value=loss),])
            self.writer.add_summary(summary, i)

            summary = tf.Summary(value=[tf.Summary.Value(tag="Training reward", simple_value=np.sum(rewards)),])
            self.writer.add_summary(summary, i)

            if i % 1000 == 0:
                self.saver.save(self.sess, "save/A2C_" + str(i))




def parse_arguments():
    # Command-line flags are defined here.
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-config-path', dest='model_config_path',
                        type=str, default='LunarLander-v2-config.json',
                        help="Path to the actor model config file.")
    parser.add_argument('--num-episodes', dest='num_episodes', type=int,
                        default=50000, help="Number of episodes to train on.")
    parser.add_argument('--lr', dest='lr', type=float,
                        default=5e-4, help="The actor's learning rate.")
    parser.add_argument('--critic-lr', dest='critic_lr', type=float,
                        default=1e-4, help="The critic's learning rate.")
    parser.add_argument('--n', dest='n', type=int,
                        default=200000000000, help="The value of N in N-step A2C.")

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
    critic_lr = args.critic_lr
    n = args.n
    render = args.render

    # Create the environment.
    env = gym.make('LunarLander-v2')
   # env = gym.make('CartPole-v0')
    a2c = A2C(env, lr, critic_lr, n)
    a2c.train(200000)

    # TODO: Train the model using A2C and plot the learning curves.


if __name__ == '__main__':
    main(sys.argv)
