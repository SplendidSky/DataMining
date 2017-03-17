import numpy as np
import pandas as pd
import csv

# 以下乘法均为矩阵乘法
# theta = (X_T * X)' * X_T * y
# 此例中 X_T * X 不可逆（不可逆的原因通常为数据冗余或数据的个数小于等于特征数，即：m <= n + 1），无法用常规方法求解

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

	#将训练数据和测试数据的第一列置为 1
	for row in train_data:
		row[0] = 1
	for row in test_data:
		row[0] = 1

	X = train_data[:, :385]
	Y = train_data[:, 385]
	X_T = X.T
	theta = np.matmul(np.matmul(np.linalg.inv(np.matmul(X_T, X)), X_T), Y)
	result = np.matmul(theta, test_data)
	write_result('result_noramll_equation.csv', result)