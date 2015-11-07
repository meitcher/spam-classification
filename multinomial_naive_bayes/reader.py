import numpy as np
import scipy.io, random
from dataset import Dataset

class Reader:

	@staticmethod
	def read_data(dirname, return_dataset=False):
		
		ds = Dataset(dirname, ['spam', 'ham'])
		emails, classes = [], []

		for sentences, email_type in ds.get_text():
			ds.build_vocabulary(sentences)
			emails.append(sentences)
			classes.append(email_type)

		
		# transform word to indices
		emails = [list(map(ds.get_word_indices().get, s)) for s in emails]

		# count how many times a word appear with the ith class
		counts = np.zeros((len(ds.vocabulary), len(set(classes))))
		for i, e in enumerate(emails):
			for w in e:
				counts[w, classes[i]] += 1 


		# emails = ds.bag_of_words(emails) # using bow we dont need counts

		if return_dataset:
			return np.array(emails), np.array(classes), counts, ds
		return np.array(emails), np.array(classes), counts



	@staticmethod
	def read_data_with_dataset(dirname, ds_orig, return_dataset=False):
		
		ds = Dataset(dirname, ['spam', 'ham'])
		emails, classes = [], []

		for sentences, email_type in ds.get_text():
			ds.build_vocabulary(sentences)
			emails.append(sentences)
			classes.append(email_type)

		
		# transform word to indices
		dic = ds_orig.get_word_indices()
		for i, s in enumerate(emails):
			for j, x in enumerate(s):
				if x in dic:
					emails[i][j] = dic[x]
				else:
					emails[i][j] = random.randint(0, len(dic)-1) # if the word was not seen before, we pick a random one

		return np.array(emails), np.array(classes)



	@staticmethod
	def split_data(X, Y, use_random=False, val=0.2):
		total = len(Y)
		indices = np.arange(total)
		if use_random:
			np.random.shuffle(indices)

		total = int(total - total*val)
		indices_pre = indices[:total]
		indices_pro = indices[total:]

		return X[indices_pre], Y[indices_pre], X[indices_pro], Y[indices_pro]


	@staticmethod
	def load_mat(filename):
		return scipy.io.loadmat(filename)

	
	@staticmethod
	def save_mat(filename, mat):
		scipy.io.savemat(filename, mat)
