import numpy as np
from matplotlib import pyplot
from features import Features

class Plots:

	@staticmethod
	def scatterplot(x, y):
		t = np.squeeze(np.asarray(y))
		pos, neg = t==1, t==0

		pyplot.plot(x[pos,0], x[pos,1], 'bo', label='y = 1')
		pyplot.plot(x[neg,0], x[neg,1], 'r*', label='y = 0')
		
		pyplot.xlim(np.min(x[:,0])-1, np.max(x[:,0])+1)
		pyplot.ylim(np.min(x[:,1])-1, np.max(x[:,1])+1)
		pyplot.legend(loc='upper right', numpoints=1)


	@staticmethod
	def lineplot(x, y, label=None, color='b'):
		pyplot.plot(x, y, color+'-', label=label)
		# pyplot.xlim(min(x), max(x))



	@staticmethod
	def draw_boundary(X, theta, degree=1):

		def map_f(x, y, degree=2):
			degree += 1
			n = (degree*(degree+1))/2
			v = np.ones((x.shape[0], n))
			k = 1

			for i in range(1, degree):
				for j in range(i+1):
					v[:, k] = np.multiply(np.power(x, (i-j)), np.power(y, j))
					k += 1
			return v

		if degree > 1:
			dim = np.linspace(np.min(X[:,0])-1, np.max(X[:,0])+1, 50)
			dx, dy = np.meshgrid(dim, dim)
			v = map_f(dx.flatten(), dy.flatten(), degree)
			z = (np.dot(v, theta)).reshape(50, 50)
			
			colors = "bbbrcmykw"
			cs = pyplot.contour(dx, dy, z, levels=[0], colors=colors[degree])
			cs.collections[0].set_label('Degree {}'.format(degree))

		else:
			plot_x = [np.min(X[:,0])-1,  np.max(X[:,0])+1]
			plot_y = [0, 0]
			plot_y[0] = (-1/theta[2]) * (theta[1]*plot_x[0] + theta[0])
			plot_y[1] = (-1/theta[2]) * (theta[1]*plot_x[1] + theta[0])
			pyplot.plot(plot_x, plot_y, 'g-', label='Degree {}'.format(degree))


	@staticmethod
	def draw_boundary_rnn(X, f_predict, nb_hidden=2):

		h = 100
		dim = np.linspace(np.min(X[:,0])-1, np.max(X[:,0])+1, h)
		dx, dy = np.meshgrid(dim, dim)
		v = np.c_[dx.flatten(), dy.flatten()]
		
		z = f_predict(v).reshape(h, h)
		z = np.array(z, dtype=np.float)
		
		# pyplot.contourf(dx, dy, z, 8, alpha=.75, cmap='jet')
		# C = pyplot.contour(dx, dy, z, 8, colors='black', linewidth=.5)
		colors = "bgbrcmykw"
		cs = pyplot.contour(dx, dy, z, levels=[0], colors=colors[nb_hidden])
		cs.collections[0].set_label('Hidden size {}'.format(nb_hidden))
