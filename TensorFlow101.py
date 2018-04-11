#Basic tensorflow usage to determine the weight and bias variable for a model

import tensorflow as tf

#Defined parameters
W = tf.Variable([.3], tf.float32)
b = tf.Variable([-.3], tf.float32)

#Declared a simple model with input and output placeholders
x = tf.placeholder(tf.float32)
linear_model = W * x + b
y = tf.placeholder(tf.float32)

#Calculation of the loss function
squared_delta = tf.square(linear_model - y)
loss = tf.reduce_sum(squared_delta)

#Optimizing the model for best vales of W and b
optimizer = tf.train.GradientDescentOptimizer(0.01)
train = optimizer.minimize(loss)


#Initializing all global variables
init = tf.global_variables_initializer()

#Create session object to launch graph
sess = tf.Session()
sess.run(init)
for i in range(1000):
    sess.run(train,{x:[1,2,3,4], y:[0,-1,-2,-3]})
print (sess.run([W,b]))
#print(sess.run(loss, {x:[1,2,3,4], y:[0,-1,-2,-3]}))
