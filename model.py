import pandas as pd
import numpy as np
from os import path, listdir
from scipy.misc import imresize

from keras.preprocessing import image
from keras.models import Sequential
from keras.layers.core import Dense,Activation,Flatten, Dropout, Lambda
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D
from keras import backend
from keras.regularizers import l2
from keras.callbacks import EarlyStopping

from scipy.ndimage.interpolation import rotate, shift
import cv2

def load_data(base):
	'''loads a log file'''
	df = pd.read_csv(path.join('data',base,'driving_log.csv'))
	df.columns = ['center', 'left', 'right', 'steering', 'throttle', 'brake', 'speed']
	for col in ['center','left','right']:
		df[col]=df[col].apply(lambda x: path.join(base,'IMG',x.split('IMG/')[-1]))
	return df

def load_img(f):
	'''loads an image'''
	return np.array(image.load_img(path.join('data',f), grayscale=False))

def trim_img(im):
	'''cut the botom off the images because this is mostly the hood of the car'''
	return im[:-25,:]

def square_img(im,s=64):
	'''resizes to a square shape'''
	return imresize(im,(s,s,3))

def process_img(im):
	'''perform all the image pre-processing'''
	return square_img(trim_img(im))


def flip_img(im,steering):
	'''flip image left to right, negate the steering angle'''
	return im[:,::-1], -1.0*steering
	
def bank_img(im,steering,angle):
	im = rotate(im,angle,reshape=False,mode='nearest')
	#TODO a banked turn should actually need less steering angle
	#but the adjustment depends on the speed I think
	return im, steering
	
def shift_img(im, steering, dx, dy, angle_per_pixel=0.002):
	im = shift(im,[dy,dx,0],mode='nearest')
	return im, steering - 1.0*angle_per_pixel*dx #shift right turn left
	
def darken_img(im, darken_p):
	#http://stackoverflow.com/questions/32609098/how-to-fast-change-image-brightness-with-python-opencv
	hsv = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)
	hsv[:,:,2]= hsv[:,:,2]*darken_p
	return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

def augment_img(x,y):
	'''performs all new image generation'''
	max_shift_x, max_shift_y=20,5
	max_bank_deg = 10.0
	min_darken = .8
	
	shift_p =0.1
	bank_p = 0.0
	darken_p =0.1
	flip_p = .5
	
	#maybe shift
	if np.random.rand() <= shift_p:
		dx = 2*max_shift_x*np.random.rand() - max_shift_x
		dy = 2*max_shift_y*np.random.rand() - max_shift_y
		x,y = shift_img(x,y, dx,dy)
		
	#maybe bank
	if np.random.rand() <= bank_p and np.abs(y) > .1:
		a = np.sign(y)*max_bank_deg*np.random.rand() #can only bank in direction of turn
		x,y = bank_img(x,y,a)
		
	#maybe darken
	if np.random.rand() <= darken_p:
		dark_p=min_darken+(1.0-min_darken)*np.random.rand()
		x = darken_img(x,dark_p)
		
	#maybe flip
	if np.random.rand() <= flip_p:
		x,y = flip_img(x,y)
	
	return x,y

def generate_batch(df, batch_size = 128, augment=True):
	'''generates a batch of data'''
	mask = df.steering == 0
	data = {'straight':df[mask],'curve':df[~mask]}
	sizes = dict([( k,len(data[k])) for k in data])
	steering_corrections = {'center':0,'left':.25, 'right':-.25}
	
	def get_sample():
		img_type = np.random.choice(['center','left','right'],p=[.70,.15,.15])
		drive_type = np.random.choice(['straight','curve'], p=[.1,.9])
		i = np.random.randint(0,sizes[drive_type])

		row=data[drive_type].iloc[i]

		x= load_img(row[img_type])
		y = row['steering'] + steering_corrections[img_type]

		return x,y

	while True:
		X=[]
		Y=[]
		for i in range(batch_size):
			
			x, y = get_sample()

			if augment:
				x,y = augment_img(x,y)
				
			x=process_img(x)
			
			X.append(x)
			Y.append(y)
			
		X = np.array(X)
		Y = np.array(Y)
		
		yield X,Y

def validation_loss():
	X,Y = next(generate_batch(valid_df,batch_size=500,augment=False))
	return model.test_on_batch(X,Y)

def autopilot_model():
	'''generates the keras model'''
	B = 1e-5
	model = Sequential()

	model.add(Lambda(lambda x: x/255.0, input_shape=(64,64,3), output_shape=(64,64,3)))

	model.add(Convolution2D(32,5,5, W_regularizer=l2(B)))
	model.add(Activation('elu'))
	model.add(Convolution2D(32,5,5, W_regularizer=l2(B)))
	model.add(MaxPooling2D((2, 2)))
	model.add(Dropout(0.5))
	model.add(Activation('elu'))

	model.add(Convolution2D(64,3,3, W_regularizer=l2(B)))
	model.add(Activation('elu'))
	model.add(Convolution2D(64,3,3, W_regularizer=l2(B)))
	model.add(MaxPooling2D((2, 2)))
	model.add(Dropout(0.5))
	model.add(Activation('elu'))

	model.add(Convolution2D(128,3,3, W_regularizer=l2(B)))
	model.add(Activation('elu'))
	model.add(Convolution2D(128,3,3, W_regularizer=l2(B)))
	model.add(MaxPooling2D((2, 2)))
	model.add(Dropout(0.5))
	model.add(Activation('elu'))

	model.add(Flatten())

	model.add(Dense(512,  W_regularizer=l2(B)))
	model.add(Activation('elu'))

	model.add(Dense(64,  W_regularizer=l2(B)))
	model.add(Activation('elu'))

	model.add(Dense(16,  W_regularizer=l2(B)))
	model.add(Activation('elu'))

	model.add(Dense(1))

	model.compile('adam', 'mse', metrics=['mse'])
	return model


if __name__=='__main__':

	np.random.seed(42)
	log = pd.concat([ load_data(base) for base in listdir('data') ])
	
	print('loaded data %d rows'%len(log))

	#filter out some rows we don't want
	log = log[log.speed > 25.0]
	log = log[log.steering.apply(lambda x: np.abs(x) < .9)]

	#split into train/test/validation set
	mask = np.random.rand(len(log)) < 0.7

	train_df = log[mask]
	other_df = log[~mask]

	mask = np.random.rand(len(other_df)) < 0.75

	test_df = other_df[mask]
	valid_df = other_df[~mask]

	print('using %d to train, %d to test, %d to validate'%(len(train_df), len(test_df), len(valid_df)))

	backend.clear_session()

	model = autopilot_model()
	#start from last model run?
	#model.load_weights('model.h5')

	early_stop = EarlyStopping(monitor='val_loss', min_delta=0, patience=1, verbose=True, mode='auto')

	model.fit_generator(
			generate_batch(train_df)
			,2*128*int(len(train_df)/128)
			,nb_epoch = 15
			,validation_data=generate_batch(test_df, batch_size=512, augment=False)
			,nb_val_samples=4*512
			,max_q_size=2000
			,nb_worker=8
			,pickle_safe=True
			,callbacks=[early_stop]
		)

	with open('model.json','w') as f:
		f.write(model.to_json())

	model.save_weights('model.h5')

	print('average loss on validation set: %f'%np.mean([ validation_loss() for i in range(10)]))

	print('model saved to model.json and model.h5')

	backend.clear_session()

	#avg validation loss: 0.012551, 0.017920, 0.017558