import tensorflow as tf


sess = tf.Session()

'''
The formula I used: a = W*x + b

shape a = (3,1)
shape W = (3,4)
shape x = (4,1)

b was left out for now

Example was based on TensorFlow Getting started (https://www.tensorflow.org/get_started/get_started).
'''

W = tf.Variable([
    [.2, .3, -.1, -.2],
    [.2, -.1, .7, -.1],
    [.1, .4, -.4, -.3],
], dtype=tf.float32, name = 'Weights')

x = tf.placeholder(shape=(4,1), dtype=tf.float32, name='input')

a = tf.matmul(W,x)

init = tf.global_variables_initializer()
sess.run(init)

y = tf.placeholder(shape=(3,1), dtype=tf.float32)

loss = tf.reduce_sum(tf.square(a - y)) # sum of the squares
optimizer = tf.train.GradientDescentOptimizer(0.01)
train = optimizer.minimize(loss)


# training data
x_train = [[1], [2], [-1], [6]]
y_train = [[8.], [11.], [13.]]
 
# training loop
init = tf.global_variables_initializer()
sess.run(init) # reset values to wrong
for i in range(1000):
    sess.run(train, {x: x_train, y: y_train})

# evaluate training accuracy
curr_W, curr_loss = sess.run([W, loss], {x: x_train, y: y_train})
print("W: \n%s \n\nloss: %s" % (curr_W, curr_loss))
