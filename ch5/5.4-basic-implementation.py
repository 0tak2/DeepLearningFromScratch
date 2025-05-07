import marimo

__generated_with = "0.13.6"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell
def _(mo):
    mo.md(
        r"""
    ## 5.4 단순한 계층 구현하기

    - 이전의 사과 쇼핑 예를 파이썬으로 구현해본다.
    - 곱셈노드는 MulLayer, 덧셈노드는 AddLayer라 한다.
    - 모든 계층은 순전파를 담당하는 forward(), 역전파를 담당하는 backward()라는 메서드가 구현된 한 개의 클래스로 구현한다.

    ### 5.4.1 곱셈 계층
    """
    )
    return


@app.cell
def _(mo):
    mo.image(src = "ch5/apple.png")
    return


@app.class_definition
# 구현체
class MulLayer:
    def __init__(self):
        self.x = None
        self.y = None

    def forward(self, x, y): # Bob: x, y를 매번 순전파할 때마다 세팅하는 게 의미가 있나? 재사용할 게 아니라면 생성자 초기화하면 어떤가?
        self.x = x
        self.y = y
        out = x * y

        return out

    def backward(self, dout): # dout은 순전파 출력에 대한 미분
        dx = dout * self.y # 서로 바꾸어 곱한다
        dy = dout * self.x

        return dx, dy


@app.cell
def _():
    # 세팅
    apple_price = 100
    apple_num = 2
    tax = 1.1

    mul_apple_layer = MulLayer()
    mul_tax_layer = MulLayer()

    # 순전파
    apple_price_sum = mul_apple_layer.forward(apple_price, apple_num)
    apple_taxed_price = mul_tax_layer.forward(apple_price_sum, tax)

    print(apple_taxed_price)
    return mul_apple_layer, mul_tax_layer


@app.cell
def _(mul_apple_layer, mul_tax_layer):
    # 역전파
    dprice = 1
    dapple_price_sum, dtax = mul_tax_layer.backward(dprice)
    dapple_price, dapple_num = mul_apple_layer.backward(dapple_price_sum)

    print(dapple_price, dapple_num, dtax)
    return


@app.cell
def _(mo):
    mo.md(r"""### 5.4.2 덧셈 계층""")
    return


@app.cell
def _(mo):
    mo.image(src = "ch5/apple-and-orange.png")
    return


@app.class_definition
# 구현체
class AddLayer:
    def __init__(self):
        # self.x = None
        # self.y = None
        # 역전파 할 때 필요 없으므로 초기화하지 않는다
        pass

    def forward(self, x, y):
        return x + y

    def backward(self, dout): # dout은 순전파 출력에 대한 미분
        dx = dout * 1 # 그대로 내보낸다
        dy = dout * 1
        
        return dx, dy


@app.cell
def _():
    # 세팅
    ex2_apple_price = 100
    ex2_apple_num = 2
    ex2_orange_price = 150
    ex2_orange_num = 3
    ex2_tax = 1.1

    ex2_mul_apple_layer = MulLayer()
    ex2_mul_orange_layer = MulLayer()
    ex2_add_apple_orange_layer = AddLayer()
    ex2_mul_tax_layer = MulLayer()

    # 순전파
    ex2_apple_price_sum = ex2_mul_apple_layer.forward(ex2_apple_price, ex2_apple_num)
    ex2_orange_price_sum = ex2_mul_orange_layer.forward(ex2_orange_price, ex2_orange_num)
    ex2_all_price_sum = ex2_add_apple_orange_layer.forward(ex2_apple_price_sum, ex2_orange_price_sum)
    ex2_final_taxed_price = ex2_mul_tax_layer.forward(ex2_all_price_sum, ex2_tax)

    print(ex2_final_taxed_price)
    return (
        ex2_add_apple_orange_layer,
        ex2_mul_apple_layer,
        ex2_mul_orange_layer,
        ex2_mul_tax_layer,
    )


@app.cell
def _(
    ex2_add_apple_orange_layer,
    ex2_mul_apple_layer,
    ex2_mul_orange_layer,
    ex2_mul_tax_layer,
):
    # 역전파
    ex2_dprice = 1
    ex2_dfinal_taxed_price, ex2_dtax = ex2_mul_tax_layer.backward(ex2_dprice)
    ex2_dapple_price_sum, ex2_dorange_price_sum = ex2_add_apple_orange_layer.backward(ex2_dfinal_taxed_price)
    ex2_dorange_price, ex2_dorange_num = ex2_mul_orange_layer.backward(ex2_dorange_price_sum)
    ex2_dapple_price, ex2_dapple_num = ex2_mul_apple_layer.backward(ex2_dapple_price_sum)

    print(ex2_dapple_num, ex2_dapple_price, ex2_dorange_price, ex2_dorange_num, ex2_dtax)
    return


if __name__ == "__main__":
    app.run()
