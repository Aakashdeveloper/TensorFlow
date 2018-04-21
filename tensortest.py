import tensorflow as tf 
sess =tf.Session()
hello = tf.constant("hello tensorflow")
print(sess.run(hello))


a = tf.constant(20)
b = tf.constant(22)
print('a+b={0}'.format(sess.run(a+b)))
