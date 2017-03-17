import tensorflow as tf
import pandas as pd
import numpy as np
import csv

#训练数据：25000 * 384

def read_train_data(filename):
	with open(filename) as csvfile:
		train_data = pd.read_csv(csvfile)
		#print(train_data)
		return train_data.as_matrix()

def read_test_data(filename):
	with open(filename) as csvfile:
		test_data = pd.read_csv(csvfile)
		return test_data.as_matrix()

def regression(train_data, test_data, times):
	graph = tf.Graph()
	result = []

	with graph.as_default():
		train_x = tf.placeholder(tf.float32, [None, 384])
		test_x = tf.placeholder(tf.float32, [None, 384])
		W = tf.Variable(tf.zeros([384, 1]), dtype = tf.float32)
		b = tf.Variable(tf.zeros(1), dtype = tf.float32)
		y = tf.matmul(train_x, W) + b
		y_ = tf.placeholder(tf.float32, [None, 1])
		predict_y = tf.matmul(test_x, W) + b
		#cross_entropy = -tf.reduce_sum(y_ * tf.log(y))
		cross_entropy = tf.sqrt(tf.reduce_mean(tf.square(y_ - y)))
		train_step = tf.train.GradientDescentOptimizer(0.001).minimize(cross_entropy)

	with tf.Session(graph = graph) as session:
		tf.global_variables_initializer().run()
		for j in range(times) :
			# for i in train_data.index:
			# 	#print('training...', i)
			# 	batch_xs = train_data.ix[i][1:385]
			# 	batch_xs = np.array(batch_xs).T
			# 	batch_xs = np.reshape(batch_xs, (1,384))
			# 	batch_ys = train_data.ix[i][385]
			# 	batch_ys = np.array(batch_ys).reshape(1, 1)
			# 	session.run(train_step, feed_dict={train_x:batch_xs, y_:batch_ys})
				#print(batch_xs)
				#print(batch_ys)
				#print(W.eval())
				#print(b.eval())
			for row in train_data:
				batch_xs = row[1:385]
				batch_xs.shape = (1, 384)
				# batch_xs = np.array(batch_xs).T
				# batch_xs = np.reshape(batch_xs, (1,384))
				batch_ys = np.array(row[385])
				batch_ys.shape = (1, 1)
				session.run(train_step, feed_dict={train_x:batch_xs, y_:batch_ys})
			print('finish training')

			print(b.eval())
			#print(W.eval())

		# for i in test_data.index:
		# 	#print('predicting:', i)
		# 	batch_xs = test_data.ix[i][1:385]
		# 	batch_xs = np.array(batch_xs).T
		# 	batch_xs = np.reshape(batch_xs, (1,384))
		# 	r = predict_y.eval(feed_dict={test_x:batch_xs})
		for row in test_data:
			#print(r)
			batch_xs = row[1:385]
			batch_xs.shape = (1, 384)
			result.append(predict_y.eval(feed_dict={test_x:batch_xs}))

	return result



	

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
	result = regression(train_data, test_data, 1)
	print('finish regression\n')
	#print(result)
	print('\n')
	#print(result.shape)
	print('\n')
	# result = [1, 2, 3]
	write_result('result.csv', result)
	print('finish')
	

	#print(train_data.head())

