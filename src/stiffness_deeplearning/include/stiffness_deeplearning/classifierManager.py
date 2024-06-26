import numpy as np
import tensorflow as tf
from keras.models import load_model

class ClassifierManager:
	"""
    A base class for implementing MLP Classifier models.

    Args:
        path (str): Path to a trained model to load.
        labels (string array, optional): names of classes used in training session.

    Attributes:
        net (Keras.model): The model of the net.
        labels (string array): names of classes.
		is_sensing (boolean): contact condition.

    Methods:        
		load_weights: update net weights
		probabilities: All classification probabilities
		predict: predicted classe with its probability
    """
	def __init__(self, path, labels = None):
		""" 
		Instance of a class to classify samples of Force-Displacement 

        :param path: path to MPL net model 
        :type path: string
        :param labels: array with classes name, it must be in the same order used in the net
        :type labels: string array
        :param is_sensing: collision condition 
        :type is_sensing: boolean
        """
		# load MPL model by path
		self.net = load_model(path)

		# set no contact condition
		self.is_sensing = False

		# define classes name 
		if labels == None:
			# DEFAULT values
			self.labels = ['grasso sottile', 'grasso spesso','vena','arteria']
		else:
			self.labels = labels

	def load_weights(self,path):
		""" 
		The function update net weights  

        :param path: Measured force [N]
        :type string: float

        :return: None
        :rtype: None
        """
		self.net.load_weights(path)		

	def probabilities(self, force, displacement):
		""" 
		The function evaluates a single sample and returns
		an array of probabilities 

        :param force: Measured force [N]
        :type force: float
        :param displacement: Measured displacment [mm]
        :type displacement: float

        :return: classification probabilities
        :rtype: float np.array
        """
		data = np.array([[force,displacement]], dtype = float)
		preds = self.net.predict(data, verbose = 0)

		return preds[0]

	def predict(self, force, displacement):
		""" 
		The function predicts the class of a single sample and returns
		the predicted class with its probability 

        :param force: Measured force [N]
        :type force: float
        :param displacement: Measured displacment [mm]
        :type displacement: float

        :return: tuple with predicted class and its probability
        :rtype: (string, float) tuple
        """
		data = np.array([[force,displacement]], dtype = float)
		preds = self.net.predict(data, verbose = 0)
		pred = preds[0]
		idx = np.argmax(pred)
		
		return (self.labels[idx], pred[idx])