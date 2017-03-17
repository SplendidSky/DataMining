from keras.models import Sequential
from keras.layers import Dropout, Dense, Activation
from keras.optimizers import SGD, RMSprop, Adam, Nadam
from keras.layers.advanced_activations import LeakyReLU
import numpy as np
import pandas as pd
import csv

def read_train_data(filename):
	with open(filename) as csvfile:
		train_data = pd.read_csv(csvfile)
		#print(train_data)
		return train_data.as_matrix()

def read_test_data(filename):
	with open(filename) as csvfile:
		test_data = pd.read_csv(csvfile)
		return test_data.as_matrix()

def write_result(filename, result):
	with open(filename, 'w', newline='') as csvfile:
		writer = csv.writer(csvfile)
		writer.writerow(['Id', 'reference'])

		i = 0

		for r in result:
			writer.writerow([str(i), str(r).lstrip('[').rstrip(']')])
			i = i + 1

if __name__ == '__main__':
	train_data = read_train_data('save_train.csv')
	test_data = read_test_data('save_test.csv')

	# sgd = SGD(lr=0.001)
	rmsprop = RMSprop()
	adam = Nadam()
	leakyReLU = LeakyReLU(alpha=0.3)

	model = Sequential()
	model.add(Dense(200, input_dim=384))
	model.add(Dropout(0.1))
	model.add(Activation('tanh'))
	model.add(Dense(100))
	model.add(Activation('tanh'))
	model.add(Dense(1))
	model.compile(optimizer=adam, loss='mse')
	# print(train_data[5])
	# print('set x0 successfully.')
	print('training')
	history = model.fit(train_data[:, 1:385], train_data[:, 385], nb_epoch=1000, batch_size=50, verbose=1, validation_split=0.2)
	print(history.history)
	# result = model.predict(test_data[:, 1:385], verbose=0)
	# write_result('result_sgd.csv', result)
