#Fine Tuned Mobilenet Model for flower classifaication.

try:
	from tensorflow.keras.preprocessing.image import ImageDataGenerator
	import tensorflow as tf
	from tensorflow.keras.layers import Dense
	from tensorflow.keras import Model
	from tensorflow.keras.optimizers import Adam
	from sklearn.metrics import confusion_matrix
	import matplotlib.pyplot as plt
	import numpy as np
	import itertools

except Exception as e:
	raise e

class MyMobileNet:
	""" Lets fine tune and train the  MobileNet module """
	def __init__(self):
		self.batch_size = 10
		self.img_width, self.img_height = 224, 224
		self.epochs = 30
		self.dataset_directories
		self.tensorboard_log_directory 
		self.image_preproces
		self.predictions = None

	#specify the dataset directories
	@property
	def dataset_directories(self):
		self.train_dir = "\\datasets\\train"
		self.valid_dir = "\\datasets\\validation"
		self.test_dir = "\\datasets\\test"
	
	#specify the log directories
	@property
	def tensorboard_log_directory(self):
		self.log_dir = "\\path\\to\\Logs"

	#preprocess the datasets
	@property
	def image_preproces(self):
		self.train_batches = ImageDataGenerator(preprocessing_function=tf.keras.applications.mobilenet.preprocess_input).flow_from_directory(directory=self.train_dir, \
		target_size=(self.img_width, self.img_height), batch_size=self.batch_size)
		self.valid_batches = ImageDataGenerator(preprocessing_function=tf.keras.applications.mobilenet.preprocess_input).flow_from_directory(directory=self.valid_dir, \
		target_size=(self.img_width, self.img_height), batch_size=self.batch_size)
		self.test_batches = ImageDataGenerator(preprocessing_function=tf.keras.applications.mobilenet.preprocess_input).flow_from_directory(directory=self.test_dir,  \
		target_size=(self.img_width, self.img_height), batch_size=self.batch_size, shuffle=False)

	def train_mobilenet_model(self):
		self.model = tf.keras.applications.mobilenet.MobileNet()
		#mobile.save("\\models\\mobilenet_versions_tf.h5") for future references
		#model_path = "\\path\\to\\model\\"
		#model = tf.keras.models.load_model(model_path)
		x = self.model.layers[-6].output
		output = Dense(units=5, activation='softmax')(x)
		self.model = Model(inputs=model.input, outputs=output)
		#model.summary()
		for layer in self.model.layers[:-13]:
			layer.trainable = False
		self.model.compile(optimizer=Adam(learning_rate=0.0001),
			loss='categorical_crossentropy', 
			metrics=['accuracy'])
		#print(len(self.model.layers))
		self.tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=self.log_dir, histogram_freq=1)
		self.model.fit(x=self.train_batches,
			steps_per_epoch=len(self.train_batches),
			validation_data=self.valid_batches,
			validation_steps=len(self.valid_batches),
            epochs=self.epochs,
            verbose=2, callbacks=[tensorboard_callback])
		#self.model.save('\\models\\mobilenet_versions_tf.h5')

	def mobilenet_model_predict(self):
		#model_path = "\\path\\to\\model\\"
		#self.model = tf.keras.models.load_model(model_path)
		self.predictions = self.model.predict(x=self.test_batches, steps=len(self.test_batches), verbose=0)
		test_labels = self.test_batches.classes
		test_indi = self.test_batches.class_indices
		label_dict = {v: k for k, v in test_indi.items()}
		pred_max = self.predictions.argmax(axis=1)
		for i in range(test_labels.size):
			print("No.:{} - Label:{} - Predicted:{}".format(i,label_dict[test_labels[i]],label_dict[pred_max[i]]))


	def main(self):
		try:
			self.train_mobilenet_model()
			self.mobilenet_model_predict()
		except Exception as e:
			raise e


if __name__=="__main__":
    call = MyMobileNet()
    call.main()


	
