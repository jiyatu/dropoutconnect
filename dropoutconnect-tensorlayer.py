import tensorflow as tf
import tensorlayer as tl # 注意！！！ TensorLayer 现在更名为 TuneLayer 了 https://github.com/zsdonghao/tunelayer
from tensorlayer.layers import set_keep
import numpy as np
import time
'''
@author jyt
'''

X_train, y_train, X_val, y_val, X_test, y_test = \
                                    tl.files.load_mnist_dataset(shape=(-1,784))

sess = tf.InteractiveSession()



# placeholder
x = tf.placeholder(tf.float32, shape=[None, 784], name='x')
y_ = tf.placeholder(tf.int64, shape=[None, ], name='y_')

network = tl.layers.InputLayer(x, name='input_layer')
network = tl.layers.DropconnectDenseLayer(network, keep = 0.8,
                                                n_units=800, act = tf.nn.relu,
                                                name='dropconnect_relu1')
network = tl.layers.DropconnectDenseLayer(network, keep = 0.5,
                                                n_units=800, act = tf.nn.relu,
                                                name='dropconnect_relu2')
network = tl.layers.DropconnectDenseLayer(network, keep = 0.5,
                                                n_units=10,
                                                act = tl.activation.identity,

                                                name='output_layer')

y = network.outputs
y_op = tf.argmax(tf.nn.softmax(y), 1)
cost = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(y, y_))

params = network.all_params

# train
n_epoch = 500
batch_size = 128
learning_rate = 0.0001
print_freq = 10
train_op = tf.train.AdamOptimizer(learning_rate, beta1=0.9, beta2=0.999,
                                epsilon=1e-08, use_locking=False).minimize(cost)

sess.run(tf.initialize_all_variables()) # initialize all variables

network.print_params()
network.print_layers()

print('   learning_rate: %f' % learning_rate)
print('   batch_size: %d' % batch_size)


for epoch in range(n_epoch):
        start_time = time.time()
        for X_train_a, y_train_a in tl.iterate.minibatches(X_train, y_train,
                                                    batch_size, shuffle=True):
            feed_dict = {x: X_train_a, y_: y_train_a}
            feed_dict.update( network.all_drop )    # enable all dropout/dropconnect/denoising layers
            sess.run(train_op, feed_dict=feed_dict)

           if epoch + 1 == 1 or (epoch + 1) % print_freq == 0:
                 print("Epoch %d of %d took %fs" % (epoch + 1, n_epoch, time.time() - start_time))
                 dp_dict = tl.utils.dict_to_one( network.all_drop ) # disable all dropout/dropconnect/denoising layers
                 feed_dict = {x: X_train, y_: y_train}
                 feed_dict.update(dp_dict)
                 print("   train loss: %f" % sess.run(cost, feed_dict=feed_dict))
                 dp_dict = tl.utils.dict_to_one( network.all_drop )
                 feed_dict = {x: X_val, y_: y_val}
                 feed_dict.update(dp_dict)
                 print("   val loss: %f" % sess.run(cost, feed_dict=feed_dict))
                 print("   val acc: %f" % np.mean(y_val ==
                                         sess.run(y_op, feed_dict=feed_dict)))
                 try:
                     # You can visualize the weight of 1st hidden layer as follow.
                     tl.visualize.W(network.all_params[0].eval(), second=10,
                                        saveable=True, shape=[28, 28],
                                        name='w1_'+str(epoch+1), fig_idx=2012)
                     # You can also save the weight of 1st hidden layer to .npz file.
                     # tl.files.save_npz([network.all_params[0]] , name='w1'+str(epoch+1)+'.npz')
                 except:
                     raise Exception("You should change visualize_W(), if you want \
                                 to save the feature images for different dataset")


print('Evaluation')
dp_dict = tl.utils.dict_to_one( network.all_drop )
feed_dict = {x: X_test, y_: y_test}
feed_dict.update(dp_dict)
print("   test loss: %f" % sess.run(cost, feed_dict=feed_dict))
print("   test acc: %f" % np.mean(y_test == sess.run(y_op,
                                                        feed_dict=feed_dict)))


tl.files.save_npz(network.all_params , name='model.npz')
tl.files.save_npz([network.all_params[0]] , name='model.npz')
# Then, restore the parameters as follow.
# load_params = tl.utils.load_npz(path='', name='model.npz')

# In the end, close TensorFlow session.
sess.close()