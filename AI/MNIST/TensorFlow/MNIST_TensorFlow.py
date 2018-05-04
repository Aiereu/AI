import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import scipy.misc
import numpy

# 데이터 셋
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

# 회귀
x = tf.placeholder(tf.float32, [None, 784])

# 변수
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))

# 모델
y = tf.nn.softmax(tf.matmul(x, W) + b)

# 학습
y_ = tf.placeholder(tf.float32, [None, 10])

# 크로스 엔트로피
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))

# 경사 하강법 사용
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

# 세션에서 모델을 실행하여 변수 초기화
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

# 학습
for i in range(1000):
    batch_xs, batch_ys = mnist.train.next_batch(100)
    sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

# 모델 평가, 정확도
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
print(sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))
