import numpy as np

class Features:

	@staticmethod
	def normalize(X):

		mu = np.zeros(X.shape[1])
		sigma = np.zeros(X.shape[1])

		for i in range(X.shape[1]):
			F = X[:,i]
			mu = np.mean(F)
			sigma = np.std(F)
			X[:,i] = (F - mu) / sigma

		return X, mu, sigma


	@staticmethod
	def get(X, mu=0, sigma=1):
		return (X-mu)/sigma


	@staticmethod
	def map(X, degree=1):

		degree += 1

		if degree > 2:
			n = (degree*(degree+1))/2
			v = np.matrix(np.zeros((X.shape[0], n-1)))
			x1, x2 = X[:, 0], X[:, 1]
			k = 0

			for i in range(1, degree):
				for j in range(i+1):
					v[:, k] = np.multiply(np.power(x1, (i-j)), np.power(x2, j))
					k += 1
			return v
		return X
		
