import os
import numpy as np
import tensorflow as tf
from sklearn.utils import shuffle
from tensorflow.contrib.layers import flatten
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("MNIST_data", reshape=False)
X_train, y_train           = mnist.train.images, mnist.train.labels
X_validation, y_validation = mnist.validation.images, mnist.validation.labels
X_test, y_test             = mnist.test.images, mnist.test.labels

assert(len(X_train) == len(y_train))
assert(len(X_validation) == len(y_validation))
assert(len(X_test) == len(y_test))

print()
print("Image Shape: {}".format(X_train[0].shape))
print()
print("Training Set:   {} samples".format(len(X_train)))
print("Validation Set: {} samples".format(len(X_validation)))
print("Test Set:       {} samples".format(len(X_test)))


# Pad images with 0s
X_train      = np.pad(X_train, ((0,0),(2,2),(2,2),(0,0)), 'constant')
X_validation = np.pad(X_validation, ((0,0),(2,2),(2,2),(0,0)), 'constant')
X_test       = np.pad(X_test, ((0,0),(2,2),(2,2),(0,0)), 'constant')

print("Updated Image Shape: {}".format(X_train[0].shape))
'''
# Pad images with 0s
X_train = tf.pad(X_train, [[0, 0], [2, 2], [2, 2], [0, 0]], mode="CONSTANT")
X_validation = tf.pad(X_validation, [[0, 0], [2, 2], [2, 2], [0, 0]], mode="CONSTANT")
X_test = tf.pad(X_test, [[0, 0], [2, 2], [2, 2], [0, 0]], mode="CONSTANT")

print("Updated Image Shape: {}".format(X_train.get_shape()))
'''
X_train, y_train = shuffle(X_train, y_train)

# Parameters
LEARNING_RATE = 0.001
EPOCHS = 10
BATCH_SIZE = 128

# Number of samples to calculate validation and accuracy
# Decrease this if you're running out of memory to calculate accuracy
TEST_VALID_SIZE = 512

# Network Parameters
n_classes = 10  # MNIST total classes (0-9 digits)
dropout = 0.75  # Dropout, probability to keep units
mu = 0
sigma = 0.1

# Store conv_net layers weight & bias
weights_cnet = {
    'wc1': tf.Variable(tf.truncated_normal(shape=(5, 5, 1, 32), mean = mu, stddev = sigma)),
    'wc2': tf.Variable(tf.truncated_normal(shape=(1, 1, 32, 64), mean = mu, stddev = sigma)),
    'wd1': tf.Variable(tf.truncated_normal(shape=(7*7*64, 1024), mean = mu, stddev = sigma)),
    'out': tf.Variable(tf.truncated_normal(shape=(1024, n_classes), mean = mu, stddev = sigma))}

biases_cnet = {
    'bc1': tf.Variable(tf.random_normal([32])), # tf.zeros(32)
    'bc2': tf.Variable(tf.random_normal([64])), # tf.zeros(64)
    'bd1': tf.Variable(tf.random_normal([1024])), # tf.zeros(1024)
    'out': tf.Variable(tf.random_normal([n_classes]))} # tf.zeros(n_classes)

# Store LeNet layers weight & bias
weights_lenet = {
    'wc1': tf.Variable(tf.truncated_normal(shape=(5, 5, 1, 6), mean = mu, stddev = sigma)),
    'wc2': tf.Variable(tf.truncated_normal(shape=(5, 5, 6, 16), mean = mu, stddev = sigma)),
    'wd1': tf.Variable(tf.truncated_normal(shape=(400, 120), mean = mu, stddev = sigma)),
    'wd2': tf.Variable(tf.truncated_normal(shape=(120, 84), mean = mu, stddev = sigma)),
    'out': tf.Variable(tf.truncated_normal(shape=(84, n_classes), mean = mu, stddev = sigma))}

biases_lenet = {
    'bc1': tf.Variable(tf.random_normal([6])), # tf.zeros(6)
    'bc2': tf.Variable(tf.random_normal([16])), # tf.zeros(16)
    'bd1': tf.Variable(tf.random_normal([120])), # tf.zeros(120)
    'bd2': tf.Variable(tf.random_normal([84])), # tf.zeros(84)
    'out': tf.Variable(tf.random_normal([n_classes]))} # tf.zeros(n_classes)


def conv2d(x, W, b, strides=1):
    x = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding='VALID')
    x = tf.nn.bias_add(x, b)
    return tf.nn.relu(x)

def maxpool2d(x, k=2):
    return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, k, k, 1], padding='VALID')

def conv_net(x, weights, biases, dropout):
    # Layer 1 - 28*28*1 to 14*14*32
    # conv1 = conv2d(x, weights['wc1'], biases['bc1'])
    # Layer 1 - 32*32*1 to 14*14*32
    conv1 = conv2d(x, weights['wc1'], biases['bc1'])
    # print("conv1 shape is: {}".format(conv1.get_shape()))
    conv1 = maxpool2d(conv1, k=2)
    # print("mac_pool conv1 shape is: {}".format(conv1.get_shape()))
    # Layer 2 - 14*14*32 to 7*7*64
    conv2 = conv2d(conv1, weights['wc2'], biases['bc2'])
    # print("conv2 shape is: {}".format(conv2.get_shape()))
    conv2 = maxpool2d(conv2, k=2)

    # Fully connected layer - 7*7*64 to 1024
    # fc1 = tf.reshape(conv2, [-1, weights['wd1'].get_shape().as_list()[0]])
    # print("max_pool conv2 shape is: {}".format(conv2.get_shape()))
    fc1 = flatten(conv2)
    fc1 = tf.add(tf.matmul(fc1, weights['wd1']), biases['bd1'])
    fc1 = tf.nn.relu(fc1)
    fc1 = tf.nn.dropout(fc1, dropout)

    # Output Layer - class prediction - 1024 to 10
    out = tf.add(tf.matmul(fc1, weights['out']), biases['out'])
    return out

from tensorflow.contrib.layers import flatten

def LeNet(x, weights, biases, dropout):

    # Layer 1: Convolutional. Input = 32x32x1. Output = 28x28x6.
    # Layer 1 - 32*32*1 to 28*28*6 to 14*14*6
    conv1 = conv2d(x, weights['wc1'], biases['bc1'])
    # Activation.
    conv1   = tf.nn.relu(conv1)
    # Pooling
    conv1 = maxpool2d(conv1, k=2)

    # Layer 2: Convolutional. Input = 14*14*6. Output = 5x5x16.
    # Layer 2 - 14*14*6 to 10*10*6 to 5*5*6
    conv2 = conv2d(conv1, weights['wc2'], biases['bc2'])
    # Activation.
    conv2   = tf.nn.relu(conv2)
    # Pooling
    conv2 = maxpool2d(conv2, k=2)

    # Flatten. Input = 5x5x16. Output = 400.
    fc0   = flatten(conv2)

    # Layer 3: Fully Connected. Input = 400. Output = 120.
    fc1 = tf.add(tf.matmul(fc0, weights['wd1']), biases['bd1'])
    fc1 = tf.nn.relu(fc1)
    fc1 = tf.nn.dropout(fc1, dropout)

    # Layer 4: Fully Connected. Input = 120. Output = 84.
    fc2 = tf.add(tf.matmul(fc1, weights['wd2']), biases['bd2'])
    fc2 = tf.nn.relu(fc2)
    fc2 = tf.nn.dropout(fc2, dropout)

    # Layer 5: Fully Connected. Input = 84. Output = 10.
    logits = tf.add(tf.matmul(fc2, weights['out']), biases['out'])

    return logits

# tf Graph input
x = tf.placeholder(tf.float32, [None, 32, 32, 1])
# y = tf.placeholder(tf.float32, [None, n_classes])
y = tf.placeholder(tf.int32, [None])
one_hot_y = tf.one_hot(y, n_classes)
keep_prob = tf.placeholder(tf.float32)

# Model
# logits = conv_net(x, weights_cnet, biases_cnet, keep_prob)
logits = LeNet(x, weights_lenet, biases_lenet, keep_prob)

# Define loss (cost) and optimizer (training_operation)
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=one_hot_y))
optimizer = tf.train.GradientDescentOptimizer(learning_rate=LEARNING_RATE)
# model_save_dir = 'TRAINED_model_sgd'
optimizer = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE)
model_save_dir = 'TRAINED_model_adam'
training_operation = optimizer.minimize(cost)

# Accuracy (accuracy_operation)
correct_pred = tf.equal(tf.argmax(logits, 1), tf.argmax(one_hot_y, 1))
accuracy_op = tf.reduce_mean(tf.cast(correct_pred, tf.float32))


if not os.path.exists(model_save_dir):
    os.makedirs(model_save_dir)

saver = tf.train.Saver()

def evaluate_data(X_data, y_data):
    num_examples = len(X_data)
    total_loss, total_accuracy = 0, 0
    sess = tf.get_default_session()
    for offset in range(0, num_examples, BATCH_SIZE):
        batch_x, batch_y = X_data[offset:offset+BATCH_SIZE], y_data[offset:offset+BATCH_SIZE]
        # cost is loss_operation
        loss, accuracy = sess.run([cost, accuracy_op], feed_dict={x: batch_x, y: batch_y, keep_prob:1.})
        total_loss     += (loss * len(batch_x))
        total_accuracy += (accuracy * len(batch_x))
    return total_loss / num_examples, total_accuracy / num_examples


# Initializing the variables
init = tf.global_variables_initializer()


# Launch the graph
with tf.Session() as sess:
    sess.run(init)
    num_examples = len(X_train)

    # steps_per_epoch = num_examples // BATCH_SIZE
    print("Training...")
    print()

    for epoch in range(EPOCHS):
        X_train, y_train = shuffle(X_train, y_train) # to ensure training isn't biased by the order of images
        for offset in range(0, num_examples, BATCH_SIZE):
            end = offset + BATCH_SIZE
            batch_x, batch_y = X_train[offset:end], y_train[offset:end]
            sess.run(training_operation, feed_dict={x: batch_x, y: batch_y, keep_prob: dropout})

            '''
            # Calculate batch loss and accuracy
            loss = sess.run(cost, feed_dict={x: batch_x, y: batch_y, keep_prob: 1.})
            valid_acc = sess.run(accuracy_op, feed_dict={
                x: X_validation[:TEST_VALID_SIZE],
                y: y_validation[:TEST_VALID_SIZE],
                keep_prob: 1.})

            print('Epoch {:>2}, Batch {:>3} - Loss: {:>10.4f} Validation Accuracy: {:.6f}'.format(
                epoch + 1,
                batch + 1,
                loss,
                valid_acc))


        val_loss, val_acc = eval_data(mnist.validation)
        print("EPOCH {} ...".format(epoch+1))
        print("Validation loss = {:.3f}".format(val_loss))
        print("Validation accuracy = {:.3f}".format(val_acc))
        print()
        '''

        validation_loss, validation_accuracy = evaluate_data(X_validation, y_validation)
        print("EPOCH {} ...".format(epoch+1))
        print("Validation Loss = {:.3f}".format(validation_loss))
        print("Validation Accuracy = {:.3f}".format(validation_accuracy))
        print()


    '''
    # Calculate Test Accuracy
    test_acc = sess.run(accuracy_op, feed_dict={
        x: mnist.test.images[:TEST_VALID_SIZE],
        y: mnist.test.labels[:TEST_VALID_SIZE],
        keep_prob: 1.})
    print('Testing Accuracy with prob: {}'.format(test_acc))

    # Evaluate on the test data
    test_loss, test_acc = eval_data(mnist.test)
    print("Eval_func Test loss = {:.3f}".format(test_loss))
    print("Eval_func Test accuracy = {:.3f}".format(test_acc))
    '''

    saver.save(sess, model_save_dir+'/lenet_wip')
    print("Model saved")


def evaluate_test_data():
    with tf.Session() as sess:
        saver.restore(sess, tf.train.latest_checkpoint('TRAINED_model/'))

        test_loss, test_accuracy = evaluate_data(X_test, y_test)
        print("Test Loss = {:.3f}".format(test_loss))
        print("Test Accuracy = {:.3f}".format(test_accuracy))

# train_model()
evaluate_test_data()
