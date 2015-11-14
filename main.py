from term2048 import ui
from term2048 import keypress
import tensorflow as tf
import numpy as np
from random import randint


# Assign a normal distribtion to the given shape
def weight_variable(shape):
  initial = tf.truncated_normal(shape)
  return tf.Variable(initial)


# Assign a constant 0.1 to the given shape
def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)


class Learner(object):
  KEY_TO_INDEX = {
      keypress.UP: 0,
      keypress.DOWN: 1,
      keypress.LEFT: 2,
      keypress.RIGHT: 3
  }
  INDEX_TO_KEY = dict((b, a) for (a, b) in KEY_TO_INDEX.iteritems())

  def __init__(self):
    self.session = tf.InteractiveSession()
    self.x = tf.placeholder("float", shape=[None, 16])
    self.y_ = tf.placeholder("float", shape=[None, 4])

    # hidden layer
    self.W_fc1 = weight_variable([16, 20])
    self.b_fc1 = bias_variable([20])
    self.h_fc1 = tf.nn.sigmoid(tf.matmul(self.x, self.W_fc1) + self.b_fc1)

    # hidden layer
    self.W_fc2 = weight_variable([20, 10])
    self.b_fc2 = bias_variable([10])
    self.h_fc2 = tf.nn.sigmoid(tf.matmul(self.h_fc1, self.W_fc2) + self.b_fc2)

    # dropout
    self.keep_prob = tf.placeholder("float")
    self.h_fc1_drop = tf.nn.dropout(self.h_fc1, self.keep_prob)

    # output layer
    self.W_fc3 = weight_variable([10, 4])
    self.b_fc3 = bias_variable([4])
    self.y = tf.nn.softmax(tf.matmul(self.h_fc2, self.W_fc3) + self.b_fc3)

    # training
    self.cross_entropy = -tf.reduce_sum(self.y_ * tf.log(self.y))
    self.train_step = tf.train.GradientDescentOptimizer(0.01).minimize(
        self.cross_entropy)

    self.q = tf.reduce_max(self.y)
    self.prediction = tf.argmax(self.y, 1)

    self.session.run(tf.initialize_all_variables())

    self.discount_factor = 0.5
    self.learning_rate = 0.5
    self.counter = 10

  def add_example(self, example_input, example_action, example_reward,
                  next_state):
    example_action = self.KEY_TO_INDEX[example_action]
    old_value = self.q.eval(feed_dict={self.x: np.matrix(example_input),
                                       self.keep_prob: 1.0})
    if example_reward == 0:
      example_reward = -10
    print "old value:", old_value
    print "reward:", example_reward
    update = self.y.eval(feed_dict={self.x: np.matrix(next_state),
                                    self.keep_prob: 1.0})
    bias = np.zeros(4)
    bias[example_action] += example_reward
    update += bias
    print "update:", update
    train_value = old_value + self.learning_rate * (self.discount_factor *
                                                    update - old_value)
    self.train_step.run(feed_dict={self.x: np.matrix(example_input),
                                   self.y_: np.matrix(train_value),
                                   self.keep_prob: 0.5})
    print "example added: %s -> %s" % (example_input, train_value)

  def get_next_move(self, state):
    weights = self.y.eval(feed_dict={self.x: np.matrix(state)})[0]
    print "weights:", weights
    best_action = self.prediction.eval(feed_dict={self.x: np.matrix(state),
                                                  self.keep_prob: 1.0})[0]
    print "predicted action:", best_action
    self.counter += 1
    random_chance = 1 / (self.counter / self.counter**0.5)
    print "random chance:", random_chance
    if randint(1, self.counter) < self.counter**0.5:
      pass
      #best_action = randint(0, 3)
    print "taken action:", best_action
    return self.INDEX_TO_KEY[best_action]


if __name__ == "__main__":
  learner = Learner()
  while 1:
    ui.start_game(learner)
