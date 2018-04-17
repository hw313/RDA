import tensorflow as tf
import numpy as np


def batches(l, n):
    """Yield successive n-sized chunks from l."""
    for i in range(0, l, n):
        yield range(i,min(l,i+n))


class Dropout_Sparse_Autoencoder():
    def __init__(self, sess, input_dim_list=[784,400], sparsities = [0.5]):
        """input_dim_list must include the original data dimension"""
        assert len(input_dim_list) >= 2
        assert len(sparsities) == len(input_dim_list) - 1
        self.W_list = []
        self.encoding_b_list = []
        self.decoding_b_list = []
        self.dim_list = input_dim_list
        for i in sparsities:
            assert i <= 1
        self.sparsities = sparsities

        ## Encoders parameters
        for i in range(len(input_dim_list)-1):
            init_max_value = np.sqrt(6. / (self.dim_list[i] + self.dim_list[i+1]))
            self.W_list.append(tf.Variable(tf.random_uniform([self.dim_list[i],self.dim_list[i+1]],
                                                             np.negative(init_max_value),init_max_value)))
            self.encoding_b_list.append(tf.Variable(tf.random_uniform([self.dim_list[i+1]],-0.1,0.1)))
        ## Decoders parameters
        for i in range(len(input_dim_list)-2,-1,-1):
            self.decoding_b_list.append(tf.Variable(tf.random_uniform([self.dim_list[i]],-0.1,0.1)))

        self.input_x = tf.placeholder(tf.float32,[None,self.dim_list[0]])
        ## coding graph :
        last_layer = self.input_x
        for weight,bias,sparse in zip(self.W_list,self.encoding_b_list,self.sparsities):
            hidden = tf.sigmoid(tf.matmul(last_layer,weight) + bias)
            hidden = tf.layers.dropout(hidden, rate = 1 - sparse)
            last_layer = hidden
        self.hidden = hidden
        ## decode graph:
        for weight,bias in zip(reversed(self.W_list),self.decoding_b_list):
            hidden = tf.sigmoid(tf.matmul(last_layer,tf.transpose(weight)) + bias)
            last_layer = hidden
        self.recon = last_layer

        #self.cost = tf.reduce_mean(tf.square(self.input_x - self.recon))
        self.cost = tf.losses.log_loss(self.recon, self.input_x)

        sess.run(tf.global_variables_initializer())

    def fit(self, X, sess, learning_rate=0.15,
            iteration=200, batch_size=50, verbose=False):
        assert X.shape[1] == self.dim_list[0]

        opt = tf.train.GradientDescentOptimizer(learning_rate)
        train_step = opt.minimize(self.cost)

        sample_size = X.shape[0]

        for i in range(iteration):
            for one_batch in batches(sample_size, batch_size):
                sess.run(train_step,feed_dict = {self.input_x:X[one_batch]})

            if verbose and i%20==0:
                e = self.cost.eval(session = sess,feed_dict = {self.input_x: X[one_batch]})
                print ("    iteration : ", i ,", cost : ", e)
        return 
    
    def transform(self, X, sess):
        return self.hidden.eval(session = sess, feed_dict={self.input_x: X})

    def getRecon(self, X, sess):
        return self.recon.eval(session = sess,feed_dict={self.input_x: X})
if __name__ == '__main__':
    import time
    import os

    os.chdir("../../")
    # x = np.load(r"./data/data.npk")
    x = np.loadtxt(r"./data/data.txt", delimiter=",")

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        sae = Dropout_Sparse_Autoencoder(sess = sess, input_dim_list=[784,784,784],sparsities=[0.5,0.5])
        print ("x type",x.shape,x.dtype)
        sae.fit(x, sess = sess, iteration = 200,learning_rate = 0.01,batch_size=97,verbose = True)

        h = sae.transform(x,sess=sess)
        np.save("transform",h)
        import ImShow as I
        import matplotlib.pyplot as plt

        Xpic = I.tile_raster_images(X=h, img_shape=(28, 28), tile_shape=(10, 10))
        plt.imshow(Xpic, cmap='gray')
        plt.show()
        plt.savefig('1.png')
        one_pic = h[0].reshape(28, 28)
        plt.imshow(one_pic, cmap='gray')
        plt.show()
        plt.savefig('2.fig')
        # print ("h shape",h.shape)
        # R = sae.getRecon(x,sess=sess)
        # print ("R",R.shape,R.dtype)
        # sae.fit(R, sess = sess, iteration = 200,learning_rate = 0.01,batch_size=97,verbose = True)

