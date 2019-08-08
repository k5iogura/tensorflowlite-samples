import tensorflow as tf
#gf = tf.compat.v1.GraphDef
gf = tf.GraphDef()
m_file = open('mnist_model_1000_epoch_50_batch.pb','rb')
gf.ParseFromString(m_file.read())

with open('somefile.txt', 'a') as the_file:
    for n in gf.node:
        the_file.write(n.name+'\n')

file = open('somefile.txt','r')
data = file.readlines()
print("output name = ")
print(data)

print("Input name = ")
file.seek ( 0 )
print(file.readline())
