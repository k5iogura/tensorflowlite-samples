# -*- coding: utf-8 -*-
import os,sys

# TensowFlowのインポート
import tensorflow as tf
from mnist import MNIST
#tfe.enable_eager_execution()

# MNISTを読み込むためinput_data.pyを同じディレクトリに置きインポートする
# input_data.pyはチュートリアル内にリンクがあるのでそこから取得する
# https://tensorflow.googlesource.com/tensorflow/+/master/tensorflow/examples/tutorials/mnist/input_data.py
import tensorflow.examples.tutorials.mnist.input_data as input_data

import time

# 開始時刻
start_time = time.time()
print "開始時刻: " + str(start_time)

# MNISTデータの読み込み
# 60000点の訓練データ（mnist.train）と10000点のテストデータ（mnist.test）がある
# 訓練データとテストデータにはそれぞれ0-9の画像とそれに対応するラベル（0-9）がある
# 画像は28x28px(=784)のサイズ
# mnist.train.imagesは[60000, 784]の配列であり、mnist.train.lablesは[60000, 10]の配列
# lablesの配列は、対応するimagesの画像が3の数字であるならば、[0,0,0,1,0,0,0,0,0,0]となっている
# mnist.test.imagesは[10000, 784]の配列であり、mnist.test.lablesは[10000, 10]の配列
print "--- MNISTデータの読み込み開始 ---"
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
print "--- MNISTデータの読み込み完了 ---"

net = MNIST(train=False)

# 交差エントロピー
# y_は正解データのラベル
y_ = tf.placeholder(tf.float32, [None, 10])
#  cross_entropy = -tf.reduce_sum(y_*tf.log(net.y))

# 勾配硬化法を用い交差エントロピーが最小となるようyを最適化する
#  train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)

# 用意した変数Veriableの初期化を実行する
#   init = tf.initialize_all_variables()

# Sessionを開始する
# runすることで初めて実行開始される（run(init)しないとinitが実行されない）

sess = tf.Session()
#   sess.run(init)

# 1000回の訓練（train_step）を実行する
# next_batch(100)で100つのランダムな訓練セット（画像と対応するラベル）を選択する
# 訓練データは60000点あるので全て使いたいところだが費用つまり時間がかかるのでランダムな100つを使う
# 100つでも同じような結果を得ることができる
# feed_dictでplaceholderに値を入力することができる
saver = tf.train.Saver()
#  print "--- 訓練開始 ---"
#  for i in range(1000):
    #  batch_xs, batch_ys = mnist.train.next_batch(100)
    #  sess.run(train_step, feed_dict={net.x: batch_xs, y_:batch_ys})
#  print "--- 訓練終了 ---"
#  saver.save(sess,"./ckpt/mnist.ckpt",global_step=100)
saver.restore(sess,"ckpt/mnist.ckpt-100")

# 正しいかの予測
# 計算された画像がどの数字であるかの予測yと正解ラベルy_を比較する
# 同じ値であればTrueが返される
# argmaxは配列の中で一番値の大きい箇所のindexが返される
# 一番値が大きいindexということは、それがその数字である確率が一番大きいということ
# Trueが返ってくるということは訓練した結果と回答が同じということ
correct_prediction = tf.equal(tf.argmax(net.y_conv,1), tf.argmax(y_,1))

# 精度の計算
# correct_predictionはbooleanなのでfloatにキャストし、平均値を計算する
# Trueならば1、Falseならば0に変換される
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

# 精度の実行と表示
# テストデータの画像とラベルで精度を確認する
# ソフトマックス回帰によってWとbの値が計算されているので、xを入力することでyが計算できる
print "精度"
print(sess.run(accuracy, feed_dict={net.x: mnist.test.images, y_: mnist.test.labels}))

frzdef = tf.compat.v1.graph_util.convert_variables_to_constants(
    sess,
    sess.graph_def,
    ['outputX']
)

prefix = 'mnist_'

with open(prefix+'frozen.pb', 'wb') as f:
    f.write(frzdef.SerializeToString())

tf.train.write_graph(
    sess.graph_def, '.',
    prefix+'frozen.pbtxt',
    as_text=True
)

tf.saved_model.simple_save(
    sess,
    './models',
    inputs={"input":net.x},
    outputs={"output":net.y_conv}
)
# 終了時刻
end_time = time.time()
print "終了時刻: " + str(end_time)

