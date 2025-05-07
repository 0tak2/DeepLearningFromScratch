import marimo

__generated_with = "0.13.6"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import os
    import sys
    sys.path.append(os.pardir)  # 부모 디렉터리의 파일을 가져올 수 있도록 설정
    from common.functions import cross_entropy_error, softmax
    import numpy as np
    return cross_entropy_error, mo, np, softmax


@app.cell
def _(mo):
    mo.md(
        r"""
    # 5.6 Affine/Softmax 계층 구현하기

    ## 5.6.1 Affine 계층

    - 은닉층에서 이전층과 다음층의 차원을 맞춰주는 계층을 본문에서는 Affine 계층이라고 부른다. affine은 기하학 용어로, 연결되어있는 상태를 말하는 라틴어 affinis에서 유래했다.
    - 그렇다면 Y = np.dot(X, W) + B와 같이 계산하던 부분에 대한 역전파를 생각해봐야 한다.
    - 스칼라가 아니라 행렬의 곱과 합이므로 전개해서 고민해봐야할 듯 하지만 본문에서 생략한다.
    - 저자가 점점 논리 전개와 유도 과정의 각 단계를 도약해버리는 부분이 많아진다. 지문의 문제였을지 저자의 글쓰기 습관인지는 잘 모르겠다. 행렬의 차원을 일치시키려고 노력하면 자연히 유도될 거라고 한다.
    - 작은 차원의 행렬을 임의로 만들고 계산해봐야할 것 같다. [ChatGPT - 행렬의 곱과 합의 미분](https://chatgpt.com/share/681b704b-dea4-800a-bc68-134d4459f55c)

    특히,

    $$
    \frac{\partial L}{\partial X} = \frac{\partial L}{\partial Y} \cdot W^{\mathrm{T}}
    $$

    $$
    \frac{\partial L}{\partial W} = X^{\mathrm{T}} \cdot \frac{\partial L}{\partial Y}
    $$

    이 부분에 대한 유도가 필요하다. ($W^{\mathrm{T}}$는 $W$의 전치행렬을 뜻한다.)

    아무튼 계산 그래프는 다음과 같다. 구현은 이것만으로도 가능하다.

    - 실제 코드에서는 배치로 행렬을 묶어서 계산하게 될 것인데, 그렇다고 역전파 방법이 변하지는 않는다.
    - (2, )과 같았던 쉐입이 (N, 2)로 바뀌게 될 뿐이다. (N -> 배치 크기)
    """
    )
    return


@app.cell
def _(mo):
    mo.image(src = "ch5/affine.jpeg")
    return


@app.cell
def _(np):
    # 구현
    class Affine:
        def __init__(self, W, b):
            self.W = W
            self.b = b
            self.x = None
            self.dW = None
            self.db = None

        def forward(self, x):
            self.x = x
            out = np.dot(x, self.W) + self.b
    
            return out

        def backward(self, dout):
            dx = np.dot(dout, self.W.T)
            self.dW = np.dot(self.x.T, dout) # 이 두 값을 왜 저장해두는 걸까?
            self.db = np.sum(dout, axis=0)

            return dx
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ## 5.6.2 Softmax-with-Loss 계층

    MNIST 손글씨 추론 신경망에서, Affine 계층을 거쳐 입력이 10개가 되면, 최종적으로는 Softmax 함수를 통해 정규화된 점수를 얻어낸다. 본문에서는 여기에 손실함수로 교차 엔트로피 오차까지 함께 구현한다.

    유도 과정은 부록으로 분리해서 다루고 있다.

    클래스가 3개인 경우의 계산 그래프는 다음과 같다.
    """
    )
    return


@app.cell
def _(mo):
    mo.image(src = "ch5/softmax-loss.jpg")
    return


@app.cell
def _(mo):
    mo.image(src = "ch5/softmax-loss-simplified.jpg")
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ### 소프트맥스 함수의 역전파
      - $y_i - t_i$ 꼴로, 정답레이블과 출력레이블의 차이와 같다.
      - 교차 엔트로피 오차를 사용하여 이렇게 떨어질 수 있는 것이라고 본문에서 밝히고 있다.
      - 예시
          - 정답 (0, 1, 0)일 때 Softmax 계층의 출력이 (0.3, 0.2, 0.5)이라면 역전파 결과는 (0.3, -0.8, 0.5)로 매우 크다.
              - 이 기울기를 바탕으로 알맞은 가중치를 찾아가게 될 것이다.
          - 정답 (0, 1, 0)일 때 Softmax 계층의 출력이 (0.01, 0.99, 0)이라면 역전파 결과는 (0.01, -0.01, 0)으로 매우 작다.
              - 학습하는 정도도 작아지게 될 것 이다.
    """
    )
    return


@app.cell
def _(cross_entropy_error, softmax):
    # 구현
    class SoftmaxWithLoss:
        def __init__(self):
            self.loss = None # 손실함수
            self.y = None # softmax의 출력
            self.t = None # 정답 레이블 (원-핫 벡터)

        def forward(self, x, t):
            self.t = t
            self. y = softmax(x)
            self.loss = cross_entropy_error(self.y, self.t)

            return self.loss

        def backward(self, dout=1):
            batch_size = self.t.shape[0]
            dx = (self.y - self.t) / batch_size # 그대로 구현하되 배치 크기로 나눠 데이터 1개당 오차를 내보냄

            return dx
    return


if __name__ == "__main__":
    app.run()
