import numpy as np
import math

class Utils:

	@staticmethod
	def sigmoid(X):
		return 1.0 / (1.0 + np.exp(-X))


	@staticmethod
	def dsigmoid(X):
		return np.multiply(Utils.sigmoid(X), (1 - Utils.sigmoid(X)))


	@staticmethod
	def add_column_with_ones(X):
		return np.c_[np.ones(len(X)), X]


	@staticmethod
	def get_epsilon(l_in, l_out=1):
		return math.sqrt(6)/math.sqrt(l_in + l_out)


	@staticmethod
	def shuffle_data(x, y):
		z = np.concatenate((x,y), axis=1)
		np.random.shuffle(z)
		return z[:,:-1], np.array(z[:,-1])


	@staticmethod
	def vectorize_output(Y, nb_labels):
		Y_vec = np.zeros((Y.shape[0], nb_labels))
		for i in range(nb_labels):
			Y_vec[:, i] = (Y == i)[:,0]
		return Y_vec
