import tensorflow as tf


sess = tf.Session()

'''
XOR port:

X        Y
[0 0] -> 0
[1 0] -> 1
[0 1] -> 1
[1 1] -> 0


O----O
 \  / \
  \/   \
  /\   /--O
 /  \ /
O----0

2 inputs
1 hidden layer with 2 neurons
1 output layer with 1 output-neuron

The formula I used: a = W*x + b

shape a = (1,1)
shape W = (1,2)
shape x = (2,1)

bv:

W = [
    [20, 20], 
    [-20, -20]
]
x = [
    [1], 
    [0]
]


b = [
    [-10],
    [30]
]

dan is a:

a = [
    [20x1 + 20x0 - 10],
    [-20x1 - 20x0 + 30]
]
=
[
[10],
[10]
]



These values can be normalised with an activation function.






b was left out for now




Example was based on TensorFlow Getting started (https://www.tensorflow.org/get_started/get_started) and youtube: 
https://www.youtube.com/watch?v=kNPGXgzxoHw
'''
W1 = tf.Variable(tf.random_uniform([2,2]), dtype=tf.float32, name = 'Weights')
x = tf.placeholder(shape=(2,None), dtype=tf.float32, name='input')
b1 = tf.Variable(tf.random_uniform([2, 1]), dtype=tf.float32, name = 'bias1')


hidden1 = tf.sigmoid(tf.matmul(W1, x) + b1)

W2 = tf.Variable(tf.random_uniform([1, 2]), dtype=tf.float32, name = 'Weights2')
b2 = tf.Variable(tf.random_uniform([1, 1]), dtype=tf.float32, name = 'bias2')

a = tf.sigmoid(tf.matmul(W2, hidden1) + b2)




init = tf.global_variables_initializer()
sess.run(init)

y = tf.placeholder(shape=(1, None), dtype=tf.float32)

loss = tf.reduce_sum(tf.square(a - y)) # sum of the squares
optimizer = tf.train.GradientDescentOptimizer(0.8)
train = optimizer.minimize(loss)

# training data
x_train = [
    [1, 0, 0, 1], 
    [0, 1, 0, 1]
]
y_train = [
    [1, 1, 0, 0]
]
 
# training loop
init = tf.global_variables_initializer()
sess.run(init) # reset values to wrong
for i in range(10000):
    sess.run(train, {x: x_train, y: y_train})

# evaluate training accuracy
curr_W1, curr_W2, curr_b1, curr_b2, curr_loss = sess.run(
    [W1, W2, b1, b2, loss], {x: [[1],[0]], y:[[1]]})
print("\nW1: %s \nW2: %s \nb1: %s \nb2: %s \nloss: %s" % (curr_W1, curr_W2, curr_b1, curr_b2, curr_loss))



