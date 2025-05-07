import marimo

__generated_with = "0.13.6"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import sys, os
    sys.path.append(os.pardir)  # 부모 디렉터리의 파일을 가져올 수 있도록 설정
    import numpy as np
    from common.layers import Affine, SoftmaxWithLoss, Relu
    from common.gradient import numerical_gradient
    from collections import OrderedDict
    from ch3.mnist import load_mnist
    return Affine, OrderedDict, Relu, SoftmaxWithLoss, load_mnist, mo, np


@app.cell
def _(mo):
    mo.md(
        r"""
    ## 5.7 오차역전파법 구현하기

    - AS-IS
        - 전제: 신경망에는 가중치와 편향이 존재하며, 데이터에 적응시킬 수 있음; 학습이란 이런 매개변수를 훈련 데이터에 적응하도록 조정하는 것
        - 미니배치: 훈련 데이터 중 일부를 무작위로 가져온 것. 이 미니배치에 대해 추론했을 때의 손실함수 출력을 줄이는게 학습의 목표
        - 기울기 산출: 손실함수에서의 가중치 매개변수의 기울기를 **수치미분**해서 구한다. 기울기를 통해 손실함수의 출력을 적게하는 방향을 알 수 있다.
        - 매개변수 갱신: 기울기 방향으로 가중치 매개변수를 살짝 조정한다.
        - 반복

    - TO-BE
        - 기울기 산출: 손실함수에서의 가중치 매개변수의 기울기를 **역전파**해서 구한다. 훨씬 빠른 속도로 효율적으로 구할 수 있다.
    """
    )
    return


@app.cell
def _(Affine, OrderedDict, Relu, SoftmaxWithLoss, np):
    # 구현
    class TwoLayerNet:
        def __init__(self, input_size, hidden_size, output_size, weight_init_std = 0.01):
            # 가중치
            self.params = {}
            self.params['W1'] = weight_init_std * np.random.randn(input_size, hidden_size)
            self.params['b1'] = np.zeros(hidden_size)
            self.params['W2'] = weight_init_std * np.random.randn(hidden_size, output_size) 
            self.params['b2'] = np.zeros(output_size)

            # 계층 생성
            self.layers = OrderedDict() # https://docs.python.org/3/library/collections.html#collections.OrderedDict
            ## 파이썬 내장 라이브러리에 포함되어있다.
            ## 순차적으로 객체를 순회할 수 있다.
            self.layers['Affine1'] = Affine(self.params['W1'], self.params['b1'])
            self.layers['Relu1'] = Relu()
            self.layers['Affine2'] = Affine(self.params['W2'], self.params['b2'])

            self.lastLayer = SoftmaxWithLoss() # 마지막 레이어는 따로 구분했음. 나중에 역전파할 때 reversed하는 부분이 있음. 거기에 포함 안시키려고 구분한 것 같음.
        
        def predict(self, x):
            # ⭐️ 각 레이어를 순차적으로 순회하면서 순전파
            for layer in self.layers.values():
                x = layer.forward(x)
        
            return x
        
        def loss(self, x, t): # x - 입력, t - 정답 레이블
            y = self.predict(x)
            return self.lastLayer.forward(y, t)
    
        def accuracy(self, x, t):
            y = self.predict(x)
            y = np.argmax(y, axis=1)
            if t.ndim != 1 : t = np.argmax(t, axis=1)
        
            accuracy = np.sum(y == t) / float(x.shape[0])
            return accuracy
        
        def numerical_gradient(self, x, t): # x - 입력, t - 정답 레이블
            loss_W = lambda W: self.loss(x, t)
        
            grads = {}
            grads['W1'] = numerical_gradient(loss_W, self.params['W1'])
            grads['b1'] = numerical_gradient(loss_W, self.params['b1'])
            grads['W2'] = numerical_gradient(loss_W, self.params['W2'])
            grads['b2'] = numerical_gradient(loss_W, self.params['b2'])
        
            return grads

        # 역전파를 이용해 기울기 구하기
        def gradient(self, x, t):
            # 순전파
            self.loss(x, t)

            # 역전파
            dout = 1
            dout = self.lastLayer.backward(dout)
        
            layers = list(self.layers.values())
            layers.reverse() # 역전파를 위해 뒤집는다.
            for layer in layers:
                dout = layer.backward(dout)

            # 결과 저장
            grads = {}
            grads['W1'], grads['b1'] = self.layers['Affine1'].dW, self.layers['Affine1'].db
            grads['W2'], grads['b2'] = self.layers['Affine2'].dW, self.layers['Affine2'].db

            return grads

    return (TwoLayerNet,)


@app.cell
def _(mo):
    mo.md(
        r"""
    ### 오차역전파법으로 구한 기울기 검증

    - 수치미분은 느린 대신 구현하기 쉽다
    - 오차역전파를 이용한 미분은 빠른 대신 복잡하고 구현하기 어려워 실수가 잦다.
    - 오차역전파로 구현하되, 잘 구현했는지 수치미분 결과와 비교하는 방식을 기울기 확인이라고 한다. 완전 같을 수는 없고 거의 비슷한지를 따져본다.
    """
    )
    return


@app.cell
def _(TwoLayerNet, load_mnist, np):
    # 기울기 확인

    (x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, one_hot_label=True)

    network = TwoLayerNet(input_size=784, hidden_size=50, output_size=10)

    x_batch = x_train[:3]
    t_batch = t_train[:3]

    grad_numerical = network.numerical_gradient(x_batch, t_batch) # 수치미분
    grad_backprop = network.gradient(x_batch, t_batch) # 역전파

    # 각 가중치의 절대 오차의 평균을 구한다.
    for key in grad_numerical.keys():
        diff = np.average( np.abs(grad_backprop[key] - grad_numerical[key]) )
        print(key + ":" + str(diff))
    # 매우 작다는 것을 확인할 수 있다
    return t_test, t_train, x_test, x_train


@app.cell
def _(TwoLayerNet, np, t_test, t_train, x_test, x_train):
    # 실제 학습 돌리기

    # 데이터 읽기
    # (x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, one_hot_label=True) # cell-4에서 읽었음

    practice_network = TwoLayerNet(input_size=784, hidden_size=50, output_size=10)

    iters_num = 10000
    train_size = x_train.shape[0]
    batch_size = 100
    learning_rate = 0.1

    train_loss_list = []
    train_acc_list = []
    test_acc_list = []

    iter_per_epoch = max(train_size / batch_size, 1)

    for i in range(iters_num):
        batch_mask = np.random.choice(train_size, batch_size)
        practice_x_batch = x_train[batch_mask]
        practice_t_batch = t_train[batch_mask]
    
        # 기울기 계산
        # grad = practice_network.numerical_gradient(practice_x_batch, practice_t_batch) # 수치 미분 (ch4)
        grad = practice_network.gradient(practice_x_batch, practice_t_batch) # 오차역전파법으로 변경
    
        # 갱신
        for layer_key in ('W1', 'b1', 'W2', 'b2'):
            practice_network.params[layer_key] -= learning_rate * grad[layer_key]
    
        loss = practice_network.loss(practice_x_batch, practice_t_batch)
        train_loss_list.append(loss)
    
        if i % iter_per_epoch == 0:
            train_acc = practice_network.accuracy(x_train, t_train)
            test_acc = practice_network.accuracy(x_test, t_test)
            train_acc_list.append(train_acc)
            test_acc_list.append(test_acc)
            print(train_acc, test_acc)
    return


if __name__ == "__main__":
    app.run()
