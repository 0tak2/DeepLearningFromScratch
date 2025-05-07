import marimo

__generated_with = "0.13.6"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import numpy as np
    return mo, np


@app.cell
def _(mo):
    mo.md(
        r"""
    ## 5.5 활성화 함수 계층 구현하기

    ### 5.5.1 ReLU 계층

    ReLu는 다음과 같다.

    $$
    \mathrm{y} =
    \begin{cases}
    x & \text{if } x > 0 \\
    0 & \text{if } x \leq 0
    \end{cases}
    $$

    그에 대한 미분은 다음과 같다.

    $$
    \frac{\partial\ \mathrm{y}}{\partial x} =
    \begin{cases}
    1 & \text{if } x > 0 \\
    0 & \text{if } x \leq 0
    \end{cases}
    $$

    - 순전파 때의 입력를 의미하는 x가 0 초과이면, 역전파 시 이전의 값을 그대로 내보냄
    - x가 0 이하면, 값을 내보내지 않음 (0)
    """
    )
    return


@app.cell
def _(mo):
    mo.image(src = "ch5/relu.jpeg")
    return


@app.cell
def _(np):
    # 구현
    class Relu:
        def __init__(self):
            self.mask = None

        def forward(self, x): # x는 numpy 배열
            self.mask = (x <= 0) # x와 동일한 shape을 가지는 불리언 numpy 배열
            print("mask init...", self.mask)
        
            out = x.copy()
            out[self.mask] = 0

            return out

        def backward(self, dout):
            dout[self.mask] = 0
            dx = dout

            return dx

    # 사용
    relu = Relu()
    forwarded = relu.forward(np.array([-1, 1, 0, 3, -2]))
    print(forwarded)

    backwarded = relu.backward(np.array([0.1, -0.2, -1.3, 3.1, 2.1])) # 순전파 시 x > 0 이었던 인덱스는 그대로 내보내진다
    print(backwarded)
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ### 5.5.2 Sigmoid 계층

    $$
    y = \frac{1}{1 + \exp(-x)}
    $$

    이를 계산 그래프로 나타내면,
    """
    )
    return


@app.cell
def _(mo):
    mo.image(src = "ch5/sigmoid.jpeg")
    return


@app.cell
def _(mo):
    mo.md(r"""시그모이드 함수의 역전파""")
    return


@app.cell
def _(mo):
    mo.image(src = "ch5/sigmoid-backward-1.jpeg")
    return


@app.cell
def _(mo):
    mo.image(src = "ch5/sigmoid-backward-2.jpeg")
    return


@app.cell
def _(mo):
    mo.image(src = "ch5/sigmoid-backward-3.jpeg")
    return


@app.cell
def _(mo):
    mo.md(r"""즉, $\frac{\partial L}{\partial x}y(1 - y)$를 구하면 된다.""")
    return


@app.cell
def _(np):
    # 구현

    class Sigmoid:
        def __init__(self):
            self.out = None

        def forward(self, x):
            out = 1 / (1 + np.exp(-x))
            self.out = out

            return out

        def backward(self, dout):
            return dout * (1.0 - self.out) * self.out # 위에서 유도한 그대로 구현한다
    return


if __name__ == "__main__":
    app.run()
