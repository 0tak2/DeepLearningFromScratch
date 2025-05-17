import marimo

__generated_with = "0.13.6"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import pickle
    import numpy as np
    from collections import OrderedDict
    from common.layers import Relu, Affine, Pooling, Convolution, SoftmaxWithLoss
    from common.gradient import numerical_gradient
    import matplotlib.pyplot as plt
    from ch3.mnist import load_mnist
    from common.trainer import Trainer
    return (
        Affine,
        Convolution,
        OrderedDict,
        Pooling,
        Relu,
        SoftmaxWithLoss,
        Trainer,
        load_mnist,
        mo,
        np,
        pickle,
        plt,
    )


@app.cell
def _(mo):
    mo.md(
        r"""
    # 7.4 합성곱/풀링 계층 구현하기
    ## 7.4.1 4차원 배열
    """
    )
    return


@app.cell
def _(np):
    # 4차원 배열
    # 미니 배치를 쓴다면 CNN에서 계층 사이를 흐르는 데이터는 4차원 배열이다.
    # 먼저 이것을 Numpy로 다루는 방법을 연습한다.

    np_4dm_test = np.random.rand(10, 1, 28, 28) # 높이 28, 너비 28, 채널 1개인 데이터가 10인 경우
    print(np_4dm_test.shape)

    print(np_4dm_test[0]) # 이 중 0번째 데이터
    print(np_4dm_test[0].shape) # (1, 28, 28) # 높이 18, 너비 28, 채널 1개인 데이터 한 건

    print(np_4dm_test[0, 0]) # 0번째 데이터의 0번째 채널을 이루는 값들
    print(np_4dm_test[0][0]) # 위와 같음
    print(np_4dm_test[0, 0].shape) # (28, 28)
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ## 7.4.2 im2col로 데이터 전개하기

    - 합성곱 연산은 중첩된 for문으로 구현할 수는 있지만 성능 문제가 있음
    - im2col 방식을 이용하면 행렬 연산으로 합성곱을 할 수 있음

    - 필터가 적용될 영역 각각이 한 행이 되어 2차원 행렬로 변환한다.
    - 그럼 필터 각각을 한 열로 변환하여 각각 행렬 곱을 하면 된다.
    """
    )
    return


@app.cell
def _(mo):
    mo.image("ch7/public/im2col1.png")
    return


@app.cell
def _(mo):
    mo.image("ch7/public/im2col2.png")
    return


@app.cell
def _(np):
    def im2col(input_data, filter_h, filter_w, stride=1, pad=0):
        """다수의 이미지를 입력받아 2차원 배열로 변환한다(평탄화).
    
        Parameters
        ----------
        input_data : 4차원 배열 형태의 입력 데이터(이미지 수, 채널 수, 높이, 너비)
        filter_h : 필터의 높이
        filter_w : 필터의 너비
        stride : 스트라이드
        pad : 패딩
    
        Returns
        -------
        col : 2차원 배열
        """
        N, C, H, W = input_data.shape
        out_h = (H + 2*pad - filter_h)//stride + 1 # 출력될 피쳐맵의 높이 == 필터가 높이상 몇 번 적용되는가
        out_w = (W + 2*pad - filter_w)//stride + 1 # 출력될 피쳐맵의 너비 == 필터가 너비상 몇 번 적용되는가
        # `//` -> 나머지가 있는 경우 정수로 내림

        img = np.pad(input_data, [(0,0), (0,0), (pad, pad), (pad, pad)], 'constant') # 패딩 추가
        col = np.zeros((N, C, filter_h, filter_w, out_h, out_w)) # Shape에 맞게 0으로 초기화

        for y in range(filter_h): # 0행부터 필터 끝까지 순회
            y_max = y + stride*out_h # 스트라이드를 고려했을 때 최종 열
            for x in range(filter_w): # 0열부터
                x_max = x + stride*out_w # x_max 행까지
                col[:, :, y, x, :, :] = img[:, :, y:y_max:stride, x:x_max:stride] # 슬라이싱 -> start:end:step
                #                            y: 증가, y_max: y만큼 증가, stride: 고정 

        col = col.transpose(0, 4, 5, 1, 2, 3).reshape(N*out_h*out_w, -1) # shape 축의 순서 바꾸고 2차원으로 reshape
        return col

    # 테스트1
    x1 = np.random.rand(1, 3, 7, 7) # 데이터 수, 채널 수, 높이, 너비
    col1 = im2col(x1, 5, 5, stride=1, pad=0)
    print(col1.shape) # (9, 75)
        # 윈도우 개수: 9개 -> 9행, 25열
        # 필터 채널 3개 -> 25행, 3*5*5열
        # 행렬을 곱하면 9행 75열

    # 테스트2
    x2 = np.random.rand(10, 3, 7, 7) # 데이터 수 10배
    col2 = im2col(x2, 5, 5, stride=1, pad=0)
    print(col2.shape) # (90, 75) # 행 10배
    return (im2col,)


@app.cell
def _(mo):
    mo.md(r"""## 7.4.3 합성곱 계층 구현하기""")
    return


@app.cell
def _(col2im, im2col, np):
    class ConvolutionPractice:
        def __init__(self, W, b, stride=1, pad=0):
            self.W = W
            self.b = b
            self.stride = stride
            self.pad = pad
        
            # 중간 데이터（backward 시 사용）
            self.x = None   
            self.col = None
            self.col_W = None
        
            # 가중치와 편향 매개변수의 기울기
            self.dW = None
            self.db = None

        def forward(self, x):
            FN, C, FH, FW = self.W.shape
            N, C, H, W = x.shape
            out_h = 1 + int((H + 2*self.pad - FH) / self.stride)
            out_w = 1 + int((W + 2*self.pad - FW) / self.stride)

            # ⭐️
            col = im2col(x, FH, FW, self.stride, self.pad) # 인풋 전개
            col_W = self.W.reshape(FN, -1).T # 필터 전개. -1을 넣으면 수동으로 명시할 필요 없이 알아서 딱 맞게 묶어줌
            out = np.dot(col, col_W) + self.b # 인풋과 필터 곱한 후 편향을 더함
            out = out.reshape(N, out_h, out_w, -1).transpose(0, 3, 1, 2) # 다시 4차원으로 만든다

            self.x = x
            self.col = col
            self.col_W = col_W

            return out

        def backward(self, dout):
            FN, C, FH, FW = self.W.shape
            dout = dout.transpose(0,2,3,1).reshape(-1, FN)

            self.db = np.sum(dout, axis=0)
            self.dW = np.dot(self.col.T, dout)
            self.dW = self.dW.transpose(1, 0).reshape(FN, C, FH, FW)

            dcol = np.dot(dout, self.col_W.T)
            dx = col2im(dcol, self.x.shape, FH, FW, self.stride, self.pad) # 역전파할 때는 col2im을 사용

            return dx
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ## 7.4.4 풀링 계층 구현하기

    1. im2col을 통해 입력 데이터를 2차원으로 전개
    2. 전개된 행렬을 행별로 순회하면서 행별 최댓값을 계산
    3. 다시 reshape
    """
    )
    return


@app.cell
def _(col2im, im2col, np):
    class PoolingPractice:
        def __init__(self, pool_h, pool_w, stride=1, pad=0):
            self.pool_h = pool_h
            self.pool_w = pool_w
            self.stride = stride
            self.pad = pad
        
            self.x = None

        def forward(self, x):
            N, C, H, W = x.shape
            out_h = int(1 + (H - self.pool_h) / self.stride)
            out_w = int(1 + (W - self.pool_w) / self.stride)

            # 1
            col = im2col(x, self.pool_h, self.pool_w, self.stride, self.pad)
            col = col.reshape(-1, self.pool_h*self.pool_w)

            # 2
            out = np.max(col, axis=1) # 1번째 차원(행)에 대해 최댓값을 구해 모은다. 0번째 차원이면 열이다.

            # 3
            out = out.reshape(N, out_h, out_w, C).transpose(0, 3, 1, 2)

            self.x = x

            return out

        def backward(self, dout):
            dout = dout.transpose(0, 2, 3, 1)
        
            pool_size = self.pool_h * self.pool_w
            dmax = np.zeros((dout.size, pool_size))
            dmax[np.arange(self.arg_max.size), self.arg_max.flatten()] = dout.flatten()
            dmax = dmax.reshape(dout.shape + (pool_size,)) 
        
            dcol = dmax.reshape(dmax.shape[0] * dmax.shape[1] * dmax.shape[2], -1)
            dx = col2im(dcol, self.x.shape, self.pool_h, self.pool_w, self.stride, self.pad)
        
            return dx
    return


@app.cell
def _(mo):
    mo.md(r"""## 7.5 CNN 구현하기""")
    return


@app.cell
def _(
    Affine,
    Convolution,
    OrderedDict,
    Pooling,
    Relu,
    SoftmaxWithLoss,
    np,
    pickle,
):
    class SimpleConvNet:
        """단순한 합성곱 신경망
    
        conv - relu - pool - affine - relu - affine - softmax
    
        Parameters
        ----------
        input_size : 입력 크기（MNIST의 경우엔 784）
        hidden_size_list : 각 은닉층의 뉴런 수를 담은 리스트（e.g. [100, 100, 100]）
        output_size : 출력 크기（MNIST의 경우엔 10）
        activation : 활성화 함수 - 'relu' 혹은 'sigmoid'
        weight_init_std : 가중치의 표준편차 지정（e.g. 0.01）
            'relu'나 'he'로 지정하면 'He 초깃값'으로 설정
            'sigmoid'나 'xavier'로 지정하면 'Xavier 초깃값'으로 설정
        """
        def __init__(self, input_dim=(1, 28, 28), 
                     conv_param={'filter_num':30, 'filter_size':5, 'pad':0, 'stride':1}, # ⭐️
                     hidden_size=100, output_size=10, weight_init_std=0.01):
            filter_num = conv_param['filter_num']
            filter_size = conv_param['filter_size']
            filter_pad = conv_param['pad']
            filter_stride = conv_param['stride']
            input_size = input_dim[1]
            conv_output_size = (input_size - filter_size + 2*filter_pad) / filter_stride + 1
            pool_output_size = int(filter_num * (conv_output_size/2) * (conv_output_size/2))

            # 가중치 초기화
            self.params = {}
        
            # 합성곱 계층
            self.params['W1'] = weight_init_std * \
                                np.random.randn(filter_num, input_dim[0], filter_size, filter_size)
            self.params['b1'] = np.zeros(filter_num)

            # 완전연결 계층
            self.params['W2'] = weight_init_std * \
                                np.random.randn(pool_output_size, hidden_size)
            self.params['b2'] = np.zeros(hidden_size)
            self.params['W3'] = weight_init_std * \
                                np.random.randn(hidden_size, output_size)
            self.params['b3'] = np.zeros(output_size)

            # 계층 생성
            self.layers = OrderedDict()
            self.layers['Conv1'] = Convolution(self.params['W1'], self.params['b1'], # 1
                                               conv_param['stride'], conv_param['pad'])
            self.layers['Relu1'] = Relu()
            self.layers['Pool1'] = Pooling(pool_h=2, pool_w=2, stride=2)
            self.layers['Affine1'] = Affine(self.params['W2'], self.params['b2']) # 2
            self.layers['Relu2'] = Relu()
            self.layers['Affine2'] = Affine(self.params['W3'], self.params['b3']) # 3

            self.last_layer = SoftmaxWithLoss()

        def predict(self, x):
            for layer in self.layers.values():
                x = layer.forward(x)

            return x

        def loss(self, x, t):
            """손실 함수를 구한다.

            Parameters
            ----------
            x : 입력 데이터
            t : 정답 레이블
            """
            y = self.predict(x)
            return self.last_layer.forward(y, t)

        def accuracy(self, x, t, batch_size=100):
            if t.ndim != 1 : t = np.argmax(t, axis=1)
        
            acc = 0.0
        
            for i in range(int(x.shape[0] / batch_size)):
                tx = x[i*batch_size:(i+1)*batch_size]
                tt = t[i*batch_size:(i+1)*batch_size]
                y = self.predict(tx)
                y = np.argmax(y, axis=1)
                acc += np.sum(y == tt) 
        
            return acc / x.shape[0]

        def numerical_gradient(self, x, t):
            """기울기를 구한다（수치미분）.

            Parameters
            ----------
            x : 입력 데이터
            t : 정답 레이블

            Returns
            -------
            각 층의 기울기를 담은 사전(dictionary) 변수
                grads['W1']、grads['W2']、... 각 층의 가중치
                grads['b1']、grads['b2']、... 각 층의 편향
            """
            loss_w = lambda w: self.loss(x, t)

            grads = {}
            for idx in (1, 2, 3):
                grads['W' + str(idx)] = numerical_gradient(loss_w, self.params['W' + str(idx)])
                grads['b' + str(idx)] = numerical_gradient(loss_w, self.params['b' + str(idx)])

            return grads

        def gradient(self, x, t):
            """기울기를 구한다(오차역전파법).

            Parameters
            ----------
            x : 입력 데이터
            t : 정답 레이블

            Returns
            -------
            각 층의 기울기를 담은 사전(dictionary) 변수
                grads['W1']、grads['W2']、... 각 층의 가중치
                grads['b1']、grads['b2']、... 각 층의 편향
            """
            # forward
            self.loss(x, t)

            # backward
            dout = 1
            dout = self.last_layer.backward(dout)

            layers = list(self.layers.values())
            layers.reverse()
            for layer in layers:
                dout = layer.backward(dout)

            # 결과 저장
            grads = {}
            grads['W1'], grads['b1'] = self.layers['Conv1'].dW, self.layers['Conv1'].db
            grads['W2'], grads['b2'] = self.layers['Affine1'].dW, self.layers['Affine1'].db
            grads['W3'], grads['b3'] = self.layers['Affine2'].dW, self.layers['Affine2'].db

            return grads
        
        def save_params(self, file_name="params.pkl"):
            params = {}
            for key, val in self.params.items():
                params[key] = val
            with open(file_name, 'wb') as f:
                pickle.dump(params, f)

        def load_params(self, file_name="params.pkl"):
            with open(file_name, 'rb') as f:
                params = pickle.load(f)
            for key, val in params.items():
                self.params[key] = val

            for i, key in enumerate(['Conv1', 'Affine1', 'Affine2']):
                self.layers[key].W = self.params['W' + str(i+1)]
                self.layers[key].b = self.params['b' + str(i+1)]
    return (SimpleConvNet,)


@app.cell
def _(SimpleConvNet, Trainer, load_mnist, np, plt):
    # 학습 시연

    ## 데이터 읽기
    (x_train, t_train), (x_test, t_test) = load_mnist(flatten=False)
    x_train, t_train = x_train[:5000], t_train[:5000] # 5000개, 1000개 데이터만 활용
    x_test, t_test = x_test[:1000], t_test[:1000]

    max_epochs = 20

    network = SimpleConvNet(input_dim=(1,28,28), 
                            conv_param = {'filter_num': 30, 'filter_size': 5, 'pad': 0, 'stride': 1},
                            hidden_size=100, output_size=10, weight_init_std=0.01)
                        
    trainer = Trainer(network, x_train, t_train, x_test, t_test,
                      epochs=max_epochs, mini_batch_size=100,
                      optimizer='Adam', optimizer_param={'lr': 0.001},
                      evaluate_sample_num_per_epoch=1000)
    trainer.train()

    ## 매개변수 보존
    network.save_params("params.pkl")
    print("Saved Network Parameters!")

    ## 그래프 그리기
    markers = {'train': 'o', 'test': 's'}
    x = np.arange(max_epochs)
    plt.plot(x, trainer.train_acc_list, marker='o', label='train', markevery=2)
    plt.plot(x, trainer.test_acc_list, marker='s', label='test', markevery=2)
    plt.xlabel("epochs")
    plt.ylabel("accuracy")
    plt.ylim(0, 1.0)
    plt.legend(loc='lower right')
    plt.show()
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ## 7.6 CNN 시각화하기

    - 첫번째 층 가중치의 shape: (30, 1, 5, 5) => 5*5 채널 1개인 필터가 30개
    - 채널이 1개이므로 회색조 이미지로 시각화 가능
    """
    )
    return


@app.cell
def _(SimpleConvNet, np, plt):
    plt.rcParams['font.family'] = 'AppleGothic'
    plt.rcParams['axes.unicode_minus'] = False 

    def filter_show(filters, nx=8, margin=3, scale=10, title="제목없음"):
        """
        c.f. https://gist.github.com/aidiary/07d530d5e08011832b12#file-draw_weight-py
        """
        FN, C, FH, FW = filters.shape
        ny = int(np.ceil(FN / nx))

        fig = plt.figure()
        fig.suptitle(title)
        fig.subplots_adjust(left=0, right=1, bottom=0, top=1, hspace=0.05, wspace=0.05)

        for i in range(FN):
            ax = fig.add_subplot(ny, nx, i+1, xticks=[], yticks=[])
            ax.imshow(filters[i, 0], cmap=plt.cm.gray_r, interpolation='nearest')
        plt.show()


    network2 = SimpleConvNet()
    # 무작위(랜덤) 초기화 후의 가중치
    filter_show(network2.params['W1'], title="학습 전")

    # 학습된 가중치
    network2.load_params("params.pkl")
    filter_show(network2.params['W1'], title="학습 후")
    return


if __name__ == "__main__":
    app.run()
