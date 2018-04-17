import tensorflow as tf
import numpy as np
def batch(l,n):
    for i in range(0,l,n):
        yield range(i,min(i+n,l))

class DeepAE(object):
    def __init__(self,sess,layers_sizes):
        self.layers_sizes=layers_sizes


        self.W=[]
        self.b_encoding=[]
        self.b_deconding=[]
        for i in range(len(self.layers_sizes)-1):
            init_max_value = np.sqrt(6. / (self.layers_sizes[i] + self.layers_sizes[i + 1]))
            # self.W.append(tf.Variable(tf.random_uniform([self.layers_sizes[i], self.layers_sizes[i + 1]],
            #             #                                                  np.negative(init_max_value), init_max_value)))
            self.W.append(tf.get_variable(name="W"+str(i),shape=[self.layers_sizes[i],self.layers_sizes[i+1]],initializer=tf.contrib.layers.xavier_initializer()))
            # self.W.append(tf.Variable(tf.random_uniform(shape=[self.layers_sizes[i],self.layers_sizes[i+1]])))
            # self.b_encoding.append(tf.Variable(tf.random_uniform(shape=[self.layers_sizes[i+1]], -0.1, 0.1)))
            self.b_encoding.append(tf.Variable(tf.random_uniform([self.layers_sizes[i + 1]], -0.1, 0.1)))
        for j in range(len(self.layers_sizes)-2,-1,-1):
            self.b_deconding.append(tf.Variable(tf.random_uniform( [self.layers_sizes[j]],-0.1,0.1)))
        self.input_X=tf.placeholder(tf.float32,shape=[None,self.layers_sizes[0]])
        self.last_layer=self.input_X
        for Weight,b in zip(self.W,self.b_encoding):
            hidden=tf.sigmoid(tf.matmul(self.last_layer,Weight)+b)
            self.last_layer=hidden
        self.hidden=hidden
        for Weight,b in zip(reversed(self.W),self.b_deconding):
            hidden =tf.sigmoid( tf.matmul(self.last_layer, tf.transpose(Weight)) + b)
            self.last_layer = hidden
        self.recon=self.last_layer
        self.cost=tf.reduce_mean(tf.square(self.input_X-self.recon))
        self.train_step=tf.train.AdamOptimizer().minimize(self.cost)
        sess.run(tf.global_variables_initializer())

    # def fit(self,X,sess,iteration,batch_size,verbose,learning_rate):
    #     sample_size=X.shape[0]
    #
    #     for i in range(iteration):
    #         for one_batch in batch(sample_size,batch_size):
    #             sess.run(self.train_step,feed_dict={self.input_X:X[one_batch]})
    #
    #         if  i % 20 == 0:
    #             e = self.cost.eval(session=sess, feed_dict={self.input_X: X[one_batch]})
    #             print("    iteration : ", i, ", cost : ", e)
    def fit(self, X, sess, learning_rate=0.15,
            iteration=200, batch_size=50, init=False,verbose=False):

        if init:
            sess.run(tf.global_variables_initializer())
        sample_size = X.shape[0]
        for i in range(iteration):
            for one_batch in batch(sample_size, batch_size):
                sess.run(self.train_step,feed_dict = {self.input_X:X[one_batch]})

            if verbose and i%20==0:
                e = self.cost.eval(session = sess,feed_dict = {self.input_X: X[one_batch]})
                print( "    iteration : ", i ,", cost : ", e)
    def transform(self, X, sess):
        return self.hidden.eval(session=sess, feed_dict={self.input_X: X})

    def getRecon(self, X, sess):
        return self.recon.eval(session=sess, feed_dict={self.input_X: X})

if __name__ == "__main__":
    import time
    import os

    os.chdir("../../")
    # x = np.load(r"./data/data.npk")
    x = np.loadtxt(r"./data/data.txt", delimiter=",")
    start_time = time.time()
    with tf.Session() as sess:
        ae = DeepAE(sess=sess, layers_sizes=[784, 225, 100])
        error = ae.fit(x, sess=sess, learning_rate=0.01, batch_size=500, iteration=500, verbose=True)
        print(1)
        R = ae.getRecon(x, sess=sess)
        print("size 100 Runing time:" + str(time.time() - start_time) + " s")
        ae.fit(R, sess=sess, learning_rate=0.01, batch_size=500, iteration=500, verbose=True)







