# -*- coding: utf-8 -*-

# TensowFlowのインポート
import tensorflow as tf

class MNIST():
    def __init__(self):
# 訓練画像を入れる変数
# 訓練画像は28x28pxであり、これらを1行784列のベクトルに並び替え格納する
# Noneとなっているのは訓練画像がいくつでも入れられるようにするため
        self.x = tf.placeholder(tf.float32, [None, 784], name='inputX')
#tf.identity(x, name='inputY')

# 重み
# 訓練画像のpx数の行、ラベル（0-9の数字の個数）数の列の行列
# 初期値として0を入れておく
        self.W = tf.Variable(tf.zeros([784, 10]))

# バイアス
# ラベル数の列の行列
# 初期値として0を入れておく
        self.b = tf.Variable(tf.zeros([10]))

# ソフトマックス回帰を実行
# yは入力x（画像）に対しそれがある数字である確率の分布
# matmul関数で行列xとWの掛け算を行った後、bを加算する。
# yは[1, 10]の行列
        self.y = tf.nn.softmax(tf.matmul(self.x, self.W) + self.b, name='outputX')

