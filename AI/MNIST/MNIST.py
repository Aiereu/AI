# 3계층의 신경망으로 MNIST 데이터를 학습하는 코드

import numpy

# 시그모이드 함수 expit() 사용을 위해 scipy.special 불러오기

import scipy.special

# 행렬을 시각화하기 위한 라이브러리
import scipy.misc

import matplotlib.pyplot

# 신경망 클래스의 정의


class NeuralNetwork:

    # 신경망 초기화하기

    def __init__(self, inputnodes, hiddennodes, outputnodes, learningrate):
        # 입력, 은닉, 출력 계층의 노드 개수 설정

        self.inodes = inputnodes

        self.hnodes = hiddennodes

        self.onodes = outputnodes

        # 가중치 행렬 wih 와 who

        # 배열내 가중치는 w_i_j로 표기, 노드 i에서 다음 계층의 노드 j로 연결됨을 의미

        # w11 w21

        # w12 w22 등

        self.wih = numpy.random.normal(0.0, pow(self.inodes, -0.5), (self.hnodes, self.inodes))

        self.who = numpy.random.normal(0.0, pow(self.hnodes, -0.5), (self.onodes, self.hnodes))

        # 학습률

        self.lr = learningrate

        # 활성화 함수로는 시그모이드 함수를 이용

        self.activation_function = lambda x: scipy.special.expit(x)

        pass

    # 신경망 학습시키기

    def train(self, inputs_list, targets_list):
        # 입력 리스트를 2차원의 행렬로 변환

        inputs = numpy.array(inputs_list, ndmin=2).T

        targets = numpy.array(targets_list, ndmin=2).T

        # 은닉 계층으로 들어오는 신호를 계산

        hidden_inputs = numpy.dot(self.wih, inputs)

        # 은닉 계층에서 나가는 신호를 계산

        hidden_outputs = self.activation_function(hidden_inputs)

        # 최종 출력 계층으로 들어오는 신호를 계산

        final_inputs = numpy.dot(self.who, hidden_outputs)

        # 최종 출력 계층에서 나가는 신호를 계산

        final_outputs = self.activation_function(final_inputs)

        # 출력 계층의 오차는 (실제값 - 계산값)

        output_errors = targets - final_outputs

        # 은닉 계층의 오차는 가중치에 의해 나뉜 출력 계층의 오차들을 재조합해 계산

        hidden_errors = numpy.dot(self.who.T, output_errors)

        # 은닉 계층과 출력 계층 간의 가중치 업데이트

        self.who += self.lr * numpy.dot((output_errors * final_outputs * (1.0 - final_outputs)),
                                        numpy.transpose(hidden_outputs))

        # 입력 계층과 은닉 계층 간의 가중치 업데이트

        self.wih += self.lr * numpy.dot((hidden_errors * hidden_outputs * (1.0 - hidden_outputs)),
                                        numpy.transpose(inputs))

        pass

    # 신경망에 질의하기

    def query(self, inputs_list):
        # 입력 리스트를 2차원 행렬로 변환

        inputs = numpy.array(inputs_list, ndmin=2).T

        # 은닉 계층으로 들어오는 신호를 계산

        hidden_inputs = numpy.dot(self.wih, inputs)

        # 은닉 계층에서 나가는 신호를 계산

        hidden_outputs = self.activation_function(hidden_inputs)

        # 최종 출력 계층으로 들어오는 신호를 계산

        final_inputs = numpy.dot(self.who, hidden_outputs)

        # 최종 출력 계층에서 나가는 신호를 계산

        final_outputs = self.activation_function(final_inputs)

        return final_outputs


# 입력, 은닉, 출력 계층의 노드 개수 설정
input_nodes = 784
hidden_nodes = 200
output_nodes = 10

# 학습률
learning_rate = 0.15
# 인공 신경망 생성
n = NeuralNetwork(input_nodes, hidden_nodes, output_nodes, learning_rate)

# 학습데이타를 로딩한다.
# mnist training data CSV 파일을 목록에로드합니다.

training_data_file = open("mnist_dataset/mnist_train_100.csv", "r")

training_data_list = training_data_file.readlines()

training_data_file.close()

# 그리고 학습데이타로 학습을 시킨다.

# epochs는 교육 데이터 세트가 교육에 사용 된 횟수입니다.

epochs = 100

for e in range(epochs):
    # 교육 데이터 세트의 모든 레코드를 검토합니다.
    for record in training_data_list:
        # split the record by the ',' commas
        all_values = record.split(',')
        # 입력을 축척하고 시프트하십시오. 입력 스케일 및 이동
        inputs = (numpy.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01
        # 목표 출력 값 생성( 원하는 레이블 0.99를 제외한 모든 0.01)
        targets = numpy.zeros(output_nodes) + 0.01
        # all_Value[0]는 이 레코드의 표적 라벨입니다.
        targets[int(all_values[0])] = 0.99
        n.train(inputs, targets)
        pass
    pass

# 이젠 테스트 데이타를 로딩한다.

test_data_file = open("mnist_dataset/mnist_test_10.csv", "r")

test_data_list = test_data_file.readlines()

test_data_file.close()

all_values = test_data_list[0].split(',')

print(all_values[0])

image_array = numpy.asfarray(all_values[1:]).reshape((28, 28))

matplotlib.pyplot.imshow(image_array, cmap='Greys', interpolation='None')

print("searching images...")
img_array = scipy.misc.imread('./query-images/test_three.png', flatten=True)

img_data = 255.0 - img_array.reshape(784)
img_data = (img_data / 255.0 * 0.99) + 0.01
print("min = ", numpy.min(img_data))
print("max = ", numpy.max(img_data))

matplotlib.pyplot.imshow(img_data.reshape(28, 28), cmap='Greys', interpolation='None')

outputs = n.query(img_data)
print(outputs)

label = numpy.argmax(outputs)
print("network says ", label)
