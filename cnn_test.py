from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("./data/mnistdata/", one_hot=False)
import os
import tensorflow as tf
import numpy as np


# Training Parameters
learning_rate = .15
batch_size = 256
model_dir = './checkpoints/a0.1c0.8'
filename = './result/a0.1c0.8.txt'
epochs =200#训练次数
noise = 1 #方式选择 1：先裁剪后加噪  2：使用差分隐私框架加噪  0：不加噪  3：先加噪裁剪
stddev = 0.1#高斯分布等噪声的数据
C=1.1#裁剪阈值
# Network Parameters
num_input = 784 # MNIST data input (img shape: 28*28)
num_classes = 10 # MNIST total classes (0-9 digits)
dropout = 0 # Dropout, probability to drop a unit


if not model_dir:
    os.mkdir(model_dir)

# Create the neural network
def conv_net(x_dict, n_classes, dropout, reuse, is_training):
    # Define a scope for reusing the variables
    with tf.variable_scope('ConvNet', reuse=reuse):
        # TF Estimator input is a dict, in case of multiple inputs
        x = x_dict['images']

        # MNIST data input is a 1-D vector of 784 features (28*28 pixels)
        # Reshape to match picture format [Height x Width x Channel]
        # Tensor input become 4-D: [Batch Size, Height, Width, Channel]
        x = tf.reshape(x, shape=[-1, 28, 28, 1])

        # Convolution Layer with 32 filters and a kernel size of 5
        conv1 = tf.layers.conv2d(x, 32, 5, activation=tf.nn.relu)
        # Max Pooling (down-sampling) with strides of 2 and kernel size of 2
        conv1 = tf.layers.max_pooling2d(conv1, 2, 2)

        # Convolution Layer with 64 filters and a kernel size of 3
        conv2 = tf.layers.conv2d(conv1, 64, 3, activation=tf.nn.relu)
        # Max Pooling (down-sampling) with strides of 2 and kernel size of 2
        conv2 = tf.layers.max_pooling2d(conv2, 2, 2)

        # Flatten the data to a 1-D vector for the fully connected layer
        fc1 = tf.contrib.layers.flatten(conv2)

        # Fully connected layer (in tf contrib folder for now)
        fc1 = tf.layers.dense(fc1, 1024)
        # Apply Dropout (if is_training is False, dropout is not applied)
        fc1 = tf.layers.dropout(fc1, rate=dropout, training=is_training)

        # Output layer, class prediction
        out = tf.layers.dense(fc1, n_classes)
    return out

#生成满足拉普拉斯分布的噪音数据，并将其转换为一个tensor
def generate_noise(v,loc=0,scale=0.01):
    c1=np.random.laplace(loc,scale,v.shape.as_list())
    c=tf.convert_to_tensor(c1,dtype=tf.float32)
    return v+c

#针对数据的数据添加噪音
def add_noise(v):
    v=v + tf.random_normal(tf.shape(v), stddev=stddev)
    #v=v + tf.random_uniform(tf.shape(v), minval=0.0, maxval=1.0)
    #v=v + tf.random.poisson(stddev,tf.shape(v),dtype=tf.dtypes.float32,seed=None,name=None)
    #v = generate_noise(v)
    return v

# Define the model function (following TF Estimator Template)
def model_fn(features, labels, mode):
    # Build the neural network
    # Because Dropout have different behavior at training and prediction time, we
    # need to create 2 distinct computation graphs that still share the same weights.
    logits_train = conv_net(features, num_classes, dropout, reuse=False,
                            is_training=True)
    logits_test = conv_net(features, num_classes, dropout, reuse=True,
                           is_training=False)

    # Predictions
    pred_classes = tf.argmax(logits_test, axis=1)
    pred_probas = tf.nn.softmax(logits_test)

    # If prediction mode, early return
    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode, predictions=pred_classes)

        # Define loss and optimizer
    loss_op = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
        logits=logits_train, labels=tf.cast(labels, dtype=tf.int32)))
    global_step = tf.Variable(0, name="global_step", trainable=False)
    optimizer = tf.train.AdadeltaOptimizer(learning_rate=learning_rate)
    #optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
    #堆梯度进行加噪处理
    if noise==1:
        gradients, variables = zip(*optimizer.compute_gradients(loss_op))
        gradients = [None if gradient is None else tf.clip_by_norm(gradient, C)
                    for gradient in gradients]
        gradients = [None if gradients is None else add_noise(gradient) for gradient in gradients]
        #gradients, global_norm = tf.clip_by_global_norm(gradients, 1.1)
        train_op=optimizer.apply_gradients(zip(gradients, variables),global_step=tf.train.get_global_step())
    elif noise==2:
        '''
         optimizer = dp_optimizer.DPAdadeltaGaussianOptimizer(
            l2_norm_clip=C,
            noise_multiplier = stddev,
            num_microbatches = batch_size,
            ledger=None,
            learning_rate = learning_rate
        )
        train_op = optimizer.minimize(loss_op,global_step=tf.train.get_global_step())
        '''
        pass
    elif noise==3:
        gradients, variables = zip(*optimizer.compute_gradients(loss_op))
        gradients = [None if gradients is None else add_noise(gradient) for gradient in gradients]
        gradients = [None if gradient is None else tf.clip_by_norm(gradient, C)
                     for gradient in gradients]
        # gradients, global_norm = tf.clip_by_global_norm(gradients, 1.1)
        train_op = optimizer.apply_gradients(zip(gradients, variables), global_step=tf.train.get_global_step())
    else:
        train_op = optimizer.minimize(loss_op,global_step=tf.train.get_global_step())
    # Evaluate the accuracy of the model
    acc_op = tf.metrics.accuracy(labels=labels, predictions=pred_classes)

    # TF Estimators requires to return a EstimatorSpec, that specify
    # the different ops for training, evaluating, ...
    estim_specs = tf.estimator.EstimatorSpec(
        mode=mode,
        predictions=pred_classes,
        loss=loss_op,
        train_op=train_op,
        eval_metric_ops={'accuracy': acc_op}

    )
    return estim_specs

session_config = tf.ConfigProto(log_device_placement=False)
#session_config.gpu_options.per_process_gpu_memory_fraction = 0.5
run_config = tf.estimator.RunConfig().replace(session_config=session_config)

# Build the Estimator
model = tf.estimator.Estimator(model_fn,
                               config=run_config,
                               model_dir=model_dir)
epoch_per_step = 60000//batch_size
w=open(filename,'w+')
for epoch in range(1,epochs+1):
    print('第epoch次训练{}'.format(epoch))
    train_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={'images': mnist.train.images}, y=mnist.train.labels,
        batch_size=batch_size, num_epochs=epochs, shuffle=True)
    model.train(train_input_fn, steps=epoch_per_step)
    # Evaluate the Model
    # Define the input function for evaluating
    test_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={'images': mnist.test.images}, y=mnist.test.labels,
        batch_size=batch_size, shuffle=False,num_epochs=epochs)
    e = model.evaluate(test_input_fn)
    w.write('{}:{}:{} \n'.format(epoch,e['accuracy'],e['loss']))
    print("Testing Accuracy:{} loss:{}".format(e['accuracy'],e['loss']))
