import prettytensor as pt
import tensorflow as tf
import numpy as np
import random
import json
import sys
import os

from itertools import cycle

from hex import Hex

topp = len(sys.argv) > 1

n = 5
hex = Hex(n)

BATCH_SIZE = 100

x = tf.placeholder(tf.float64, shape=(None, n**2+1), name="x")
y = tf.placeholder(tf.float64, shape=(None, n**2), name="y")


def tprint(msg):
    def inner(activation):
        return tf.Print(activation, [activation], msg)

    return inner


y_ = (pt.wrap(x)
      .fully_connected(64, activation_fn=tf.nn.relu)
      #   .fully_connected(1024, activation_fn=tf.nn.relu)
      #   .dropout(0.7)
      #   .fully_connected(1024, activation_fn=tf.nn.relu)
      #   .dropout(0.7)
      #   .fully_connected(1024, activation_fn=tf.nn.relu)
      #   .dropout(0.7)
      #   .fully_connected(512, activation_fn=tf.nn.relu)
      #   .dropout(0.7)
      #   .fully_connected(256, activation_fn=tf.nn.relu)
      .fully_connected(64, activation_fn=tf.nn.relu)
      .fully_connected(n**2, activation_fn=tf.nn.softmax, name="y_")
      )


loss = tf.losses.mean_squared_error(y, y_)

optimizer = tf.train.GradientDescentOptimizer(0.001)

train_op = pt.apply_optimizer(optimizer, losses=[loss])

init_op = tf.global_variables_initializer()

if __name__ == "__main__" and not topp:
    with open('examples/net/variables.json', 'w') as outfile:
        json.dump({
            "x": x.name,
            "y": y.name,
            "y_": y_.name,
            "train": train_op.name,
            "variables": list(map(lambda v: v.name, tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES))),
        }, outfile)

    with tf.Session() as session:
        tf.train.write_graph(session.graph_def, 'examples/net',
                             'model.pb', as_text=False)

        writer = tf.summary.FileWriter(
            "examples/net/output/tensorboard", session.graph)
        writer.close()


class HexPlayer:
    def __init__(self, pyz):
        self.variables = np.load(pyz)
        self.session = tf.Session()

        tf_vars = {v.name: v for v in tf.get_collection(
            tf.GraphKeys.TRAINABLE_VARIABLES)}

        for name, array in self.variables.items():
            self.session.run(
                tf.assign(tf_vars[name], np.reshape(array, tf_vars[name].shape)))

    def get_distribution(self, state):
        (player, board, _) = state

        return self.session.run(y_, {
            x: [[player, *board.flatten()]],
        })[0]


def play(m, player1, player2):
    wins = {
        player1: 0,
        player2: 0,
    }

    for _ in range(m):
        to_play = [player1, player2]
        random.shuffle(to_play)
        players = cycle(to_play)

        game = hex.initial_state()
        result = hex.result(game)

        while result == 0:
            player = next(players)

            action = hex.action_from_distribution(
                game, player.get_distribution(game))

            game = hex.take_action(game, action)

            result = hex.result(game)

        wins[player] += 1

    return wins


if __name__ == "__main__" and topp:
    # dir = "examples/net/output/saved/2018-11-22T18:54:13.833627952+01:00"
    # dir = "examples/net/output/saved/2018-11-22T21:56:55.446493267+01:00"
    # dir = "examples/net/output/saved/2018-11-22T23:36:46.924831559+01:00"
    dir = sys.argv[1]

    files = os.listdir(dir)
    players = {file.split(".")[0]: HexPlayer(
        os.path.join(dir, file)) for file in files}

    for p1 in players:
        for p2 in players:
            if p1 != p2:
                wins = play(100, players[p1], players[p2])

                wins = {
                    p1: wins[players[p1]],
                    p2: wins[players[p2]],
                }

                print(wins)
