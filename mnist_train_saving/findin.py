import tensorflow as tf
#gf = tf.compat.v1.GraphDef
#gf = tf.GraphDef()
gf = tf.compat.v1.GraphDef()
m_file = open('mnist.pb','rb')
gf.ParseFromString(m_file.read())

with open('somefile.txt', 'a') as the_file:
    for n in gf.node:
        the_file.write(n.name+'\n')

file = open('somefile.txt','r')
data = file.readlines()
print("output name = ",data)

print("Input name = ")
file.seek ( 0 )
print(file.readline())
