import tensorflow as tf

'''
Dimensie van W = (aantal neurons, aantal inputs)
Dimensie van x = (aantal inputs, aantal samples (None))
Dimensie van b = (aantal neurons, 1)

dim(W) = (1, 2)
dim(x) = (2, None)
dim(b) = (1, 1)

'''


sess = tf.Session()

threshold = 0.2
W = tf.Variable(tf.random_uniform([1,2], minval=-1, maxval=1), dtype=tf.float32, name = 'Weights')
x = tf.placeholder(shape=(2, None), dtype=tf.float32, name='input')
b = tf.Variable([[threshold]], dtype=tf.float32, name = 'bias')



Y = tf.round(tf.nn.sigmoid(tf.matmul(W,x) + b)) # round(sigmoid()) == step()



learning_rate = 0.1



init = tf.global_variables_initializer()
sess.run(init)



Yd = tf.placeholder(shape=(1, None), dtype=tf.float32, name = 'label')

error = Yd - Y
loss = tf.reduce_sum(tf.abs(error))

optimizer = tf.train.GradientDescentOptimizer(learning_rate)
train = optimizer.minimize(loss)
print(sess.run([error, loss], {x: [[0,0,1,1], [0,1,0,1]], Yd: [[0,1,1,1]]}))
exit()
# training data
x_train = [
    [1, 0, 0, 1], 
    [0, 1, 0, 1]
]
y_train = [
    [1, 1, 0, 1]
]
 
# training loop
init = tf.global_variables_initializer()
sess.run(init) # reset values to wrong
for i in range(1000000):
    sess.run(train, {x: x_train, y: y_train})

# evaluate training accuracy
curr_W1, curr_W2, curr_b1, curr_b2, curr_loss = sess.run(
    [W, b, loss], {x: x_train, y:y_train})
#print("\nW1: %s \nW2: %s \nb1: %s \nb2: %s \nloss: %s" % (curr_W1, curr_W2, curr_b1, curr_b2, curr_loss))