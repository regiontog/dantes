import prettytensor as pt
import tensorflow as tf
import json

n = 5
BATCH_SIZE = 100

x = tf.placeholder(tf.float64, shape=(None, n**2+1), name="x")
y = tf.placeholder(tf.float64, shape=(None, n**2), name="y")

y_ = (pt.wrap(x)
      .fully_connected(64, activation_fn=tf.nn.relu)
      .fully_connected(256, activation_fn=tf.nn.relu)
      .fully_connected(256, activation_fn=tf.nn.relu)
      .fully_connected(64, activation_fn=tf.nn.relu)
      .fully_connected(n**2, activation_fn=tf.nn.softmax, name="y_"))


loss = tf.losses.mean_squared_error(y, y_)

optimizer = tf.train.AdamOptimizer(0.00001)
train_op = pt.apply_optimizer(
    optimizer, losses=[loss])

with open('examples/net/variables.json', 'w') as outfile:
    json.dump({
        "x": x.name,
        "y": y.name,
        "y_": y_.name,
        "train": train_op.name,
        "variables": list(map(lambda v: v.name, tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES))),
    }, outfile)

init_op = tf.global_variables_initializer()

with tf.Session() as session:
    tf.train.write_graph(session.graph_def, 'examples/net',
                         'model.pb', as_text=False)

    writer = tf.summary.FileWriter("examples/net/output", session.graph)
    writer.close()
