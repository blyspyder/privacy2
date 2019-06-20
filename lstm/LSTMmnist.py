# encoding:utf-8
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import os

# 导入数据
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

# hyperparameters 超参数
lr = 0.001
training_iters = 1000000
batch_size = 128

n_inputs = 28  # (img shape:28 * 28)
n_steps = 28  # time steps
n_hidden_units = 128  # 隐藏层神经元数目
n_classes = 10  # classes(0-9 digits)

noise = 1  # 0表示不加早  1,表示先加噪后裁剪  2表示先裁剪后加噪
C=0.8
stddev=0.5
filename = 'a0.5c0.8.txt'
model_dir = './Adamcheckpoints/a0.5c0.8'
if not model_dir:
    os.mkdir(model_dir)
# 输入
x = tf.placeholder(tf.float32, [None, n_steps, n_inputs])
y = tf.placeholder(tf.float32, [None, n_classes])

# 定义权值

weights = {
    # (28, 128)
    'in': tf.Variable(tf.random_normal([n_inputs, n_hidden_units])),
    # (128, 10)
    'out': tf.Variable(tf.random_normal([n_hidden_units, n_classes]))
}
biases = {
    # (128, )
    'in': tf.Variable(tf.constant(0.1, shape=[n_hidden_units, ])),
    # (10, )
    'out': tf.Variable(tf.constant(0.1, shape=[n_classes]))
}

def RNN(X, weights, biases):
    # 隐藏层输入到cell
    # X(128 batch, 28 steps, 28 inputs)
    #  ==>(128*28,28 inputs)
    X = tf.reshape(X, [-1, n_inputs])
    # X_in==>(128batch*28steps, 128 hidden)
    X_in = tf.matmul(X, weights['in'] + biases['in'])
    # X_in==>(128batch, 28steps, 128 hidden)
    X_in = tf.reshape(X_in, [-1, n_steps, n_hidden_units])
    # cell
    lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(n_hidden_units, forget_bias=1.0, state_is_tuple=True)
    # lstm cell 被分成两部分，(c_state, m_state)
    _init_state = lstm_cell.zero_state(batch_size, dtype=tf.float32)
    outputs, states = tf.nn.dynamic_rnn(lstm_cell, X_in, initial_state=_init_state, time_major=False)
    # 隐藏层输出
    # 第一种方式
    results = tf.matmul(states[1], weights['out']) + biases['out']
    # 第二种方式
    # outputs = tf.unstack(tf.transpose(outputs, [1, 0, 2]))  # state is the last outputs
    # results = tf.matmul(outputs[-1], weights['out']) + biases['out']
    return results

def add_noise(v):
    v=v + tf.random_normal(tf.shape(v), stddev=stddev)
    return v

pred = RNN(x, weights, biases)
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
optimizer = tf.train.AdamOptimizer(lr)
if noise==0:
    train_op = optimizer.minimize(cost)
elif noise ==1:
    gradients, variables = zip(*optimizer.compute_gradients(cost))
    gradients = [None if gradients is None else add_noise(gradient) for gradient in gradients]
    gradients = [None if gradient is None else tf.clip_by_norm(gradient, C)
                 for gradient in gradients]
    # gradients, global_norm = tf.clip_by_global_norm(gradients, 1.1)
    train_op = optimizer.apply_gradients(zip(gradients, variables), global_step=tf.train.get_global_step())
else:
    gradients, variables = zip(*optimizer.compute_gradients(cost))
    gradients = [None if gradient is None else tf.clip_by_norm(gradient, C)
                 for gradient in gradients]
    gradients = [None if gradients is None else add_noise(gradient) for gradient in gradients]
    train_op = optimizer.apply_gradients(zip(gradients, variables), global_step=tf.train.get_global_step())
correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

init = tf.initialize_all_variables()

#saver = tf.train.Saver()
sess = tf.Session()
sess.run(init)
w=open('./result/'+filename,'w')
for step in range(10001):
    batch_xs, batch_ys = mnist.train.next_batch(batch_size)
    batch_xs = batch_xs.reshape([batch_size, n_steps, n_inputs])
    sess.run([train_op], feed_dict={
        x: batch_xs,
        y: batch_ys,
    })
    if step % 20 == 0:
        result=sess.run([cost,accuracy], feed_dict={x: batch_xs,y: batch_ys})
        print('{} {}'.format(step,result))
        w.write('{},{},{}\n'.format(step,str(result[0]),str(result[1])))
    step += 1
#saver.save(sess,model_dir)
