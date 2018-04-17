import numpy as np
import tensorflow as tf
import DAE_tensorflow as dae


# def l21(x):
#     return tf.reduce_sum(tf.sqrt(tf.reduce_sum(x ** 2 , reduction_indices = 1)))

class RobustL21Autoencoder():
    """
    @author: Chong Zhou
    first version.
    complete: 10/20/2016

    Des:
        X = L + S
        L is a non-linearly low dimension matrix and S is a sparse matrix.
        argmin ||L - Decoder(Encoder(L))|| + ||S||_2,1
        Use Alternating projection to train model
        The idea of shrink the l21 norm comes from the wiki 'Regularization' link: {
            https://en.wikipedia.org/wiki/Regularization_(mathematics)
        }
    Improve:
        1. fix the 0-cost bugs

    """
    def __init__(self, sess, layers_sizes, lambda_=1.0, error = 1.0e-5):
        self.lambda_ = lambda_
        self.layers_sizes = layers_sizes
        self.error = error
        self.errors=[]

        self.AE = dae.Deep_Autoencoder( sess = sess, input_dim_list = self.layers_sizes)

    def l21shrink(self, epsilon, x):
        """
        auther : Chong Zhou
        date : 10/20/2016
        Args:
            epsilon: the shrinkage parameter
            x: matrix to shrink on
        Ref:
            wiki Regularization: {https://en.wikipedia.org/wiki/Regularization_(mathematics)}
        Returns:
            The shrunk matrix
        """
        output = x.copy()
        norm = np.linalg.norm(x, ord=2, axis=0)

        for i in range(x.shape[1]):
            if norm[i] > epsilon:
                for j in xrange(x.shape[0]):
                    output[j,i] = x[j,i] - epsilon * x[j,i] / norm[i]
            elif norm[i] < -epsilon:
                for j in xrange(x.shape[0]):
                    output[j,i] = x[j,i] + epsilon * x[j,i] / norm[i]
            else:
                output[:,i] = 0.
        return output

    def fit(self, X, sess, learning_rate=0.15, inner_iteration = 50,
            iteration=20, batch_size=40, verbose=False):
        ## The first layer must be the input layer, so they should have same sizes.
        assert X.shape[1] == self.layers_sizes[0]
        ## initialize L, S, mu(shrinkage operator)
        self.L = np.zeros(X.shape)
        self.S = np.zeros(X.shape)
        assert X.shape[1] == self.layers_sizes[0]
        ## initialize L, S
        self.L = np.zeros(X.shape)
        self.S = np.zeros(X.shape)
        ## one_over_mu
        ## this scaling suppose to get sparsicty about 5 iteration
        ##one_over_mu = np.linalg.norm(np.linalg.norm(X,2,axis=0),1) / X.shape[1]
        one_over_mu = 1.
        LS0 = self.L + self.S
        ## To estimate the size of input X
        XFnorm = np.linalg.norm(X,'fro')
        if verbose:
            print ("X shape: ", X.shape)
            print ("L shape: ", self.L.shape)
            print ("S shape: ", self.S.shape)
            print ("XFnorm: ", XFnorm)

        for it in range(iteration):
            if verbose:
                print ("Out iteration: " , it)
            ## alternating project, first project to L
            self.L = X - self.S
            ## Using L to train the auto-encoder
            self.errors.extend(self.AE.fit(self.L, sess = sess,
                                           iteration = inner_iteration,
                                           learning_rate = learning_rate,
                                           batch_size = batch_size,
                                           verbose = verbose))
            ## get optmized L
            self.L = self.AE.getRecon(X = self.L, sess = sess)
            ## alternating project, now project to S and shrink S
            self.S = self.l21shrink(self.lambda_ * one_over_mu, (X - self.L).T).T
            ## Converge Condition: the L and S are close enough to X
            c1 = np.linalg.norm(X - self.L - self.S, 'fro') / XFnorm
            if c1 < self.error:
                print ("early break")
                if verbose:
                    print ("c1: ", c1)
                break
            LS0 = self.L + self.S
        return self.L , self.S
    def transform(self, X, sess):
        L = X - self.S
        return self.AE.transform(X = L, sess = sess)
    def getRecon(self, X, sess):
        return self.AE.getRecon(self.L, sess = sess)
if __name__ == "__main__":
    import os
    os.chdir("../../")
    x=np.loadtxt(r'data/data.txt',delimiter=',')
    with tf.Session() as sess:
        rae = RobustL21Autoencoder(sess = sess, lambda_= 4000, layers_sizes=[784,400,255,100])

        L, S = rae.fit(x, sess = sess, inner_iteration = 600, iteration = 30,verbose = True)
        print (rae.errors)
