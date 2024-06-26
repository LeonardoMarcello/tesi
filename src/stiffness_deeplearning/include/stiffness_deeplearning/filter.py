from queue import Queue

class Filter:
	"""
    A base class for implementing a 1D low pass filter.

    Args:
        t0 (float): initial timestamp in millis.
        x0 (float, default 0): initial state values.
        omega (float, default 0.5): cut frequency in rad/s.
        beta (float, default 1): parameter in [0,1] to use for a fading filter implementation.
        queue_size (int, default 2): lenght of values stored.

    Attributes:
        x (queue.Queue): values array.
        y (queue.Queue): measures array.
        omega (float): cut frequency in rad/s.
        beta (float): parameter in [0,1] to use for a fading filter implementation.

    Methods:        
		update: Use a new measure to update filter state
		get_time_and_measure: get current state of the filter
    """
	def __init__(self, t0, x0 = .0, omega = 0.5, beta = 1, queue_size = 2):
		""" 
		Initialization of a class to filters signals with a low-pass
		H(s) = omega/(s+omega). It can be used as a first order 
		fading filter by setting beta params and updating with 
		'fading filter' method 

        :param t0: initial timestamp 
        :type y: float
        :param omega: filter cut frequency in [rad/s]
        :type omega: float
        :param beta: coefficient for a fading filter method
        :type beta: float in [0,1]
        :param queue_size: number of measure and estimation saved in buffers. It should be greater than 2
        :type queue_size: int
        """
		# init memory buffer
		self.x = Queue(maxsize=queue_size)
		self.y = Queue(maxsize=queue_size)
		
		self.x.put((t0,x0))			# init state values
		self.y.put((t0,0.0))		# init measure values
		
		# set filter params
		self.omega = omega			# cut frequency in [rad/s]
		self.beta = beta			# coefficient in [0,1] for a fading filter method

	def update(self, y, t_millis, method = 'bilinear'):
		""" 
		Update the filter state with a new measure by using
		an approximation of the lowpass H(s) = omega/(s+omega) evaluated with
		one of the following method: bilinear approx 'bilinear' or backward euler
		'EI'. The method 'fading filter' can be used to implement a filter that uses
		beta param  

        :param y: measure obtained
        :type y: float
        :param t_millis: time at which the measure is obtained
        :type t_millis: float
        :param method: method used to approximate the low-pass
        :type method: string

        :return: filtered signal
        :rtype: float
        """
		# load previous values
		t_meno,x_meno = self.x.queue[self.x.qsize()-1]
		_,y_meno = self.y.queue[self.y.qsize()-1]
		
		# evaluate current state
		if method == 'bilinear':
			# s = 2/dt*(z-1)/(z+1)
			dt = (t_millis - t_meno)/1000.0
			w0dt = self.omega*dt
			a1 = (2-w0dt)/(2+w0dt)
			b0 = w0dt/(2+w0dt)
			b1 = w0dt/(2+w0dt)

			# x(k) = a1*x(k-1)+b0*y(k)+b1*y(k-1)
			x_k = a1*x_meno + b0*y + b1*y_meno

		elif method == 'EI':
			# s = 1/dt*(z-1)/z
			dt = (t_millis - t_meno)/1000.0
			w0dt = self.omega*dt
			a1 = 1/(1+w0dt)
			b0 = w0dt/(1+w0dt)

			# x(k) = a1*x(k-1)+b0*y(k)		
			x_k = a1*x_meno + b0*y

		elif method == 'fading filter':
			# x(k) = (1-beta)*x(k-1) + beta*y(k)
			x_k = (1-self.beta)*x_meno + self.beta*y

		# if full queue free space
		if self.x.full():
			_ = self.x.get()
		
		if self.y.full():
			_ = self.y.get()


		# update queue
		self.x.put((t_millis, x_k))
		self.y.put((t_millis, y))

		return x_k
	
	def get_time_and_measure(self):
		""" 
		Return the current state of the filter with the timestamp

        :return: tupla with time in millis and filtered signal
        :rtype: (float, float) tuple
        """
		time, x = self.x.queue[self.x.qsize()-1]
		
		return (time, x)