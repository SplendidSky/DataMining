import tensorflow as tf

def basic_operation():
	graph = tf.Graph()

	with graph.as_default():
		value1 = tf.Variable([1, 2])
		value2 = tf.constant([3, 4])
		mul = value1 * value2

	with tf.Session(graph = graph) as session:
		tf.global_variables_initializer().run()
		print(mul.eval())
		print(mul.eval())

def use_placeholder():
	graph = tf.Graph()
	with graph.as_default():
		value1 = tf.placeholder(dtype = tf.float64)
		value2 = tf.Variable([3, 4], dtype = tf.float64)
		sub = value1 - value2

	with tf.Session(graph = graph) as session:
		tf.global_variables_initializer().run()
		value = load_from_remote()
		for partialValue in load_partial(value, 2):
			evalResult = sub.eval(feed_dict = {value1 : partialValue})
			print(evalResult)


def load_from_remote():
	return [-x for x in range(1000)]

def load_partial(value, step):
	index = 0
	while index < len(value):
		yield value[index:index + step]
		index += step
	return



if __name__ == '__main__':
	basic_operation()
	print("--------------")
	use_placeholder()