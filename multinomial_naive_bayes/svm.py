import numpy as np
import scipy.io, math, sys, time
from matplotlib import pyplot

from utils import Utils
from statistics import Statistics
from features import Features
from reader import Reader
from plots import Plots



class NaiveBayes:
	"""
	Simple NaiveBayes
	"""

	def __init__(self, smooth = 1e-9):
		self.smooth = smooth 	# zero probabilities
		self.priors = None
		self.likelihood = None
		self.classes = None
		self.weights = None
		self.nb_words = 0


	def train(self, X, Y, Z, vocabulary_size):

		# X = input
		# Y = output
		# Z = counting

		nb_examples, nb_classes = len(Y), len(np.unique(Y))
		nb_words = vocabulary_size
		self.nb_words = nb_words

		self.priors = np.zeros((nb_classes, 1))
		self.likelihood = np.zeros((nb_words, nb_classes))
		self.classes = np.unique(Y)

		for k in range(nb_classes):

			idx = (Y == self.classes[k])
			self.priors[k] = np.sum(idx) / nb_examples

			word_count_in_class = Z[:, k]
			total_words_in_class = np.sum(word_count_in_class) 

			self.likelihood[:, k] = (word_count_in_class + self.smooth) / (total_words_in_class + self.smooth*nb_words)


		self.weights = np.zeros((nb_words+1, nb_classes))
		for k in range(nb_classes):
			self.weights[0, k] = np.log(self.priors[k])
			self.weights[1:, k] = np.log(self.likelihood[:, k])



	def count_row(self, row):
		d = np.zeros((self.nb_words+1, 1))
		d[0] = 1 # bias for priory probability
		for x in row:
			d[x+1] += 1
		return d.T


	def predict(self, X):
		S = np.zeros((len(X), len(self.classes)))
		for i in range(len(X)):
			S[i] = np.dot(self.count_row(X[i]), self.weights)
		return np.argmax(S, axis=1).T


	def evaluate(self, truth, predicted):
		return np.mean(truth == predicted) * 100.0


if __name__ == '__main__':

	X, Y, Z, ds = Reader.read_data('dados/enron5', True)
	X_train, Y_train, X_val, Y_val = Reader.split_data(X, Y, use_random=True, val=0.2)
	X_val, Y_val, X_test, Y_test   = Reader.split_data(X_val, Y_val, use_random=True, val=0.5) 
	# X_test, Y_test = Reader.read_data_with_dataset('dados/enron1', ds)
	

	# data = {'X_train':X_train, 'Y_train':Y_train, 'X_val':X_val, 'Y_val':Y_val, 'Z':Z}
	# Reader.save_mat('dados/enron1.mat', data)

	# mat = Reader.load_mat('dados/enron1.mat')
	# X_train, Y_train, X_val, Y_val, Z = mat['X_train'], mat['Y_train'], mat['X_val'], mat['Y_val'], mat['Z']


	classifier = NaiveBayes()
	classifier.train(X_train, Y_train, Z, len(Z))
	
	Y_star = classifier.predict(X_train)
	print('Train accuracy:', classifier.evaluate(Y_train, Y_star))
	Statistics.confusion_mat(Y_train, Y_star)

	Y_vstar = classifier.predict(X_val)
	print('Valid accuracy:', classifier.evaluate(Y_val, Y_vstar))
	Statistics.confusion_mat(Y_val, Y_vstar)

	Y_tstar = classifier.predict(X_test)
	print('Test  accuracy:', classifier.evaluate(Y_test, Y_tstar))
	Statistics.confusion_mat(Y_test, Y_tstar)
	
