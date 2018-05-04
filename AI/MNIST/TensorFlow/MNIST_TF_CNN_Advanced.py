import tensorflow as tf
import os
import cv2
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from tensorflow.examples.tutorials.mnist import input_data

"""
tensorflow : tensorflow 기능을 제공
os : 디렉토리 경로 호출
cv2 : 이미지 파일 로딩
numpy as np : 데이터 처리, 행렬 연산 기능 제공
sklearn.preprocessing : 문자인 폴더 리스트를 숫자형 array로 변환 및
one-hot vector 변환에 사용
tensorflow.examples.tutorials.mnist import input_data : 학습 데이터
"""

# 데이터 셋
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

TEST_DIR = './MNIST_data/Test_Set/'
test_folder_list = np.array(os.listdir(TEST_DIR))

# TensorFlow InteractiveSession 시작
sess = tf.InteractiveSession()

# 변수
test_input = []
test_label = []

# Label Encoder / 문자열로 구성된 Test_Set을 숫자형 리스트로 변환
label_encoder = LabelEncoder()
integer_encoded = label_encoder.fit_transform(test_folder_list)

# OneHot Encoder /  integer_encoded의 shape을 (10,)에서 (10,1)로 변환 후 One Hot Vector 형태로 변환
onehot_encoder = OneHotEncoder(sparse=False)
integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
onehot_encoded = onehot_encoder.fit_transform(integer_encoded)

# for문 경로 연산 및 파일명 추출
for index in range(len(test_folder_list)):
    path = os.path.join(TEST_DIR, test_folder_list[index])
    path = path + '/'
    img_list = os.listdir(path)

    # 이미지 데이터, label을 np.array 형식으로 각각 리스트에 입력
    for img in img_list:
        img_path = os.path.join(path, img)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        test_input.append([np.array(img)])
        test_label.append([np.array(onehot_encoded[index])])

# 크기에 적절하게 형태 변환 및 입력
test_input = np.reshape(test_input, (-1, 784))
test_label = np.reshape(test_label, (-1, 10))

# 각 리스트의 데이터 타입을 float32로 변환
test_input = np.array(test_input).astype(np.float32)
test_label = np.array(test_label).astype(np.float32)

# 각 리스트를 파일로 저장
np.save("test_input.npy", test_input)
np.save("test_label.npy", test_label)

# 회귀
x = tf.placeholder(tf.float32, shape=[None, 784])
y_ = tf.placeholder(tf.float32, shape=[None, 10])

# 변수
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))

"""
# 단층 신경망
# 세션 - 초기화 / 심층에선 모델의 훈련 및 평가 부분 위치
sess.run(tf.global_variables_initializer())

# 소프트맥스, 크로스 엔트로피 사용
y = tf.nn.softmax(tf.matmul(x, W) + b)
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))

# 모델 훈련
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

# 50개의 훈련 sample
for i in range(1000):
    batch = mnist.train.next_batch(50)
    train_step.run(feed_dict={x: batch[0], y_: batch[1]})

# 모델 평가하기
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
print(accuracy.eval(feed_dict={x: mnist.test.images, y_: mnist.test.labels}))
"""


# 가중치 초기화 leaky_ReLU함수 (활성함수)
def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


# 합성곱 & 풀링
def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                          strides=[1, 2, 2, 1], padding='SAME')


# 첫 번째 합성곱, leaky_ReLU 함수 (활성함수)
W_conv1 = weight_variable([5, 5, 1, 32])
b_conv1 = bias_variable([32])

x_image = tf.reshape(x, [-1, 28, 28, 1])

h_conv1 = tf.nn.leaky_relu(conv2d(x_image, W_conv1) + b_conv1, alpha=-0.1)
h_pool1 = max_pool_2x2(h_conv1)

# 두 번째 합성곱
W_conv2 = weight_variable([5, 5, 32, 64])
b_conv2 = bias_variable([64])

h_conv2 = tf.nn.leaky_relu(conv2d(h_pool1, W_conv2) + b_conv2, alpha=-0.1)
h_pool2 = max_pool_2x2(h_conv2)

# Fully-Connected Layer
W_fc1 = weight_variable([7 * 7 * 64, 1024])
b_fc1 = bias_variable([1024])

h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 64])
h_fc1 = tf.nn.leaky_relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1, alpha=-0.1)

# 과적합 방지, Dropout
keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

# 소프트맥스
W_fc2 = weight_variable([1024, 10])
b_fc2 = bias_variable([10])

y_conv = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

# 크로스 엔트로피, ADAM 최적화 알고리즘 > 경사 하강법
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y_conv), reduction_indices=[1]))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# 변수 초기화
sess.run(tf.global_variables_initializer())

# 훈련 및 평가, 훈련을 많이 할 수록 정확도와 시간이 비례함
for i in range(1000):
    batch = mnist.train.next_batch(50)
    if i % 100 == 0:
        train_accuracy = accuracy.eval(feed_dict={
            x: batch[0], y_: batch[1], keep_prob: 1.0})
        print("step %d, training accuracy %g" % (i, train_accuracy))
    train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})

# 앞서 준비한 테스트 이미지를 비교하여 정확도를 출력
print("test accuracy %g" % accuracy.eval(feed_dict={
    x: test_input, y_: test_label, keep_prob: 1.0}))
