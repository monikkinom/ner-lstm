import numpy as np, time
import matplotlib.pyplot as plt
from scipy.stats import logistic

class NN3:

	W = []
	b = []
	epsilon = 0.1
	hType = 1
	oType = 1
	regLambda = 0
	bias = False

	def __init__(self, iSize, hSize, oSize, **kwargs):
		self.iSize = iSize
		self.hSize = hSize
		self.oSize = oSize
		if 'hType' in kwargs:
			self.hType = kwargs['hType']
		if 'oType' in kwargs:
			self.oType = kwargs['oType']
		if 'bias' in kwargs:
			self.bias = kwargs['bias']
		if 'epsilon' in kwargs:
			self.epsilon = kwargs['epsilon']
		if 'regLambda' in kwargs:
			self.regLambda = kwargs['regLambda']
		self.init()

	def exceptionHandler(eID):
		if eID == 1:
			print('Model shape mismatch.')
		elif eID == 2:
			print('I/O sample number mismatch.')

	def init(self):
		self.W.append(np.random.randn(self.hSize, self.iSize))
		self.W[0] /= np.sqrt(self.iSize)
		self.W.append(np.random.randn(self.oSize, self.hSize))
		self.W[1] /= np.sqrt(self.hSize)
		if self.bias:
			self.b.append(np.zeros((1, self.hSize)))
			self.b.append(np.zeros((1, self.oSize)))
	
	def scost(O, T):
			return \
			np.sum(np.power(O - T, 2)) / (2 * O.shape[0])

	'''
	Note this function does not implement the actual derivative.
	Rather it helps in computation of the derivative.
	'''
	def func(mat, function, deriv = False):
		mat = np.array(mat)
		if function == 0:
			if deriv:
				return 1
			else:
				return mat
		elif function == 1:
			if deriv:
				return mat * (1 - mat)
			else:
				return logistic._cdf(mat)
		elif function == 2:
			if deriv:
				return 1 - np.power(mat, 2)
			else:
				return np.tanh(mat)

	def predict(self, X):
		#Exceptions.
		if X.shape[1] != self.iSize:
			NN3.exceptionHandler(1)
			return None
		#Feedforwarding.
		Z1 = X.dot(self.W[0].T)
		if self.bias:
			Z1 += self.b[0]
		A1 = NN3.func(Z1, self.hType)
		Z2 = A1.dot(self.W[1].T)
		if self.bias:
			Z2 += self.b[1]
		A2 = NN3.func(Z2, self.oType)
		return [A1, A2]

	def train(self, X, Y, epoch = 100):
		if X.shape[0] != Y.shape[0]:
			NN3.exceptionHandler(2)
			return None

		if X.shape[1] != self.iSize or \
		Y.shape[1] != self.oSize:
			NN3.exceptionHandler(1)

		J = []
		for itr in range(epoch):
			#FeedForward.
			AD1 = X.dot(self.W[0].T)
			if self.bias:
				AD1 += self.b[0]
			AD1 = NN3.func(AD1, self.hType)
			AD2 = AD1.dot(self.W[1].T)
			if self.bias:
				AD2 += self.b[1]
			AD2 = NN3.func(AD2, self.oType)
			#Calculating cost for the layer
			cost = NN3.scost(AD2, Y)
			#Backward propogation.
			AD2 -= Y
			#Gradient of W1
			dW1 = (AD2.T).dot(AD1)

			AD1 = AD2.dot(self.W[1]) \
			* NN3.func(AD1, self.hType, True)
			
			#Gradient of W0.
			dW0 = (AD1.T).dot(X)
			if self.regLambda != 0:
				#Regularizing.
				dW1 += self.regLambda * self.W[1]
				dW0 += self.regLambda * self.W[0]
				
				cost = cost + self.regLambda \
				* np.sum(np.power(self.W[0], 2)) / 2
				
				cost = cost + self.regLambda \
				* np.sum(np.power(self.W[1], 2)) / 2
			
			self.W[1] -= self.epsilon * dW1
			self.W[0] -= self.epsilon * dW0
			if self.bias:
				
				db1 = \
				np.sum(AD2, axis = 0, keepdims = True)
				
				db0 = \
				np.sum(AD1, axis = 0, keepdims = True)
				
				self.b[1] -= self.epsilon * db1
				self.b[0] -= self.epsilon * db0
			J.append(cost)
		return J

	def batch(self, X, Y, isCost = False):
		#FeedForward.
		AD1 = X.dot(self.W[0].T)
		if self.bias:
			AD1 += self.b[0]
		AD1 = NN3.func(AD1, self.hType)
		AD2 = AD1.dot(self.W[1].T)
		if self.bias:
			AD2 += self.b[1]
		AD2 = NN3.func(AD2, self.oType)
		#Backward propogation.
		AD2 -= Y
		if isCost:
			cost = np.sum(np.power(AD2, 2))
		#Gradient of W1.
		dW1 = (AD2.T).dot(AD1)
		
		AD1 = AD2.dot(self.W[1]) \
		* NN3.func(AD1, self.hType, True)

		#Gradient of W0.
		dW0 = (AD1.T).dot(X)
		self.W[1] -= self.epsilon * dW1
		self.W[0] -= self.epsilon * dW0
		if self.bias:

			db1 = \
			np.sum(AD2, axis = 0, keepdims = True)
				
			db0 = \
			np.sum(AD1, axis = 0, keepdims = True)
				
			self.b[1] -= self.epsilon * db1
			self.b[0] -= self.epsilon * db0
		if isCost:
			return cost

def main():
	'''
	X = np.diag([1, 1, 1, 1, 1, 1, 1, 1])
	Y = np.diag([1, 1, 1, 1, 1, 1, 1, 1])
	model = NN3(8, 3, 8, hType = 1, oType = 1)
	J = []
	J = model.train(X, Y, 10000)
	plt.plot(J)
	plt.show()
	params = model.predict(X)
	print(params[0])
	print(np.argmax(params[1], axis = 1))
	'''
	start = time.time()
	np.tanh([1, 2, 3, 4, 5])
	print((time.time() -start) * 1000)

if __name__ == '__main__':
	main()
