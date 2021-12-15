import cv2
import os
import shutil
import numpy as np
from os.path import isfile, join
from keras.preprocessing.image import ImageDataGenerator

def getFrame(sec):
    vidcap.set(cv2.CAP_PROP_POS_MSEC,sec*1000)
    hasFrames,image = vidcap.read() 
    if hasFrames:
        cv2.imwrite("images/"+str(count)+".jpg", image)
        return hasFrames

datagen = ImageDataGenerator()
os.mkdir('images')
all_dir_names = ['Mountain Pose', 'Raised Arms Pose', 'Standing Forward Bend', 'Garland Pose', 'Lunge Pose', 'Plank Pose', 'Staff Pose', 'Seated Forward Bend', 'Head To Knee Pose', 'Happy Baby Pose']

for dir_names in all_dir_names:
    X, y = [], []

    directory_path = dir_names + '/'
    directory_name = os.listdir(dir_names)
    directory_len = len(directory_name)
    print('Beginning Directory: ' + dir_names)
    
    for names in range(1, directory_len+1):
        shutil.rmtree('images')
        os.mkdir('images')
        
        name = directory_path+'sample'+str(names)+'.mp4'
        vidcap = cv2.VideoCapture(name)
        length = vidcap.get(cv2.CAP_PROP_FRAME_COUNT)
        ffppss = vidcap.get(cv2.CAP_PROP_FPS)
        
        sec = 0
        frameRate = 1/ffppss
        count = 1
        success = getFrame(sec)
        
        while success:
            count = count + 1
            sec = sec + frameRate
            sec = round(sec, 2)
            success = getFrame(sec)
                        
        pathIn = 'images/'
        files = [f for f in os.listdir(pathIn) if isfile(join(pathIn, f))]
        files.sort(key=lambda f: int(f.split('.')[0]))
        
        for i in range(len(files)%16, len(files), 16):
            list_frames = []
            rand_num = np.random.rand()
            rand_int = np.random.randint(-11, 11)
            rand_zoom = np.random.uniform(0.9, 1.1)
            
            for j in range(16):
                filename = pathIn + files[i + j]
                img = cv2.imread(filename)
                img = cv2.resize(img, (112, 112), cv2.INTER_AREA)
                boolean = False
                
                if rand_num > 0.7:
                    if rand_num > 0.75:
                        boolean = True
                    img = (datagen.apply_transform(x=img, transform_parameters={'theta':rand_int, 'tx':rand_int, 'ty':rand_int,'shear':rand_int, 'zx':rand_zoom, 'zy':rand_zoom, 'flip_horizontal':boolean})).astype('uint8')

                img = img / 255.
                list_frames.append(img)
                       
            X.append(list_frames)
            y.append(dir_names)
        print('% Completion in ' + dir_names + ' is '+ str(np.round(names/(directory_len+1)*100, decimals=3)))
        
    np.save('D:\Dataset\X '+dir_names+'.npy', X)
    np.save('D:\Dataset\y '+dir_names+'.npy', y)
	
import h5py
hfx = h5py.File('DataX.h5', 'r')
hfy = h5py.File('DataY.h5', 'r')
print(list(hfx.keys()))
print(list(hfy.keys()))

import numpy as np
x = np.asarray(hfx['X'])
y = np.asarray(hfy['labels'])
hfx.close()
hfy.close()
	
from sklearn.model_selection import train_test_split
X_train, X_val, y_train, y_val = train_test_split(x, y, test_size=0.2, stratify=y)
X_val, X_test, y_val, y_test = train_test_split(X_val, y_val, test_size=0.5, stratify=y_val)
print(X_train.shape)
print(y_train.shape)
print()
print(X_val.shape)
print(y_val.shape)
print()
print(X_test.shape)
print(y_test.shape)

from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Flatten
from keras.layers import Convolution3D, MaxPooling3D, ZeroPadding3D, BatchNormalization, AveragePooling3D

def get_model():
    model = Sequential()
    model.add(Convolution3D(64, (3, 3, 3), activation='relu', 
                            border_mode='same', name='conv1',
                            subsample=(1, 1, 1), 
                            input_shape=(16, 112, 112, 3)))
    model.add(BatchNormalization())
    model.add(MaxPooling3D(pool_size=(1, 2, 2), strides=(1, 2, 2), 
                           border_mode='valid', name='pool1'))
    model.add(Convolution3D(128, (3, 3, 3), activation='relu', 
                            border_mode='same', name='conv2',
                            subsample=(1, 1, 1)))
    model.add(BatchNormalization())
    model.add(MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2), 
                           border_mode='valid', name='pool2'))
    model.add(Convolution3D(256, (3, 3, 3), activation='relu', 
                            border_mode='same', name='conv3a',
                            subsample=(1, 1, 1)))
    model.add(Convolution3D(256, (3, 3, 3), activation='relu', 
                            border_mode='same', name='conv3b',
                            subsample=(1, 1, 1)))
    model.add(BatchNormalization())
    model.add(MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2), 
                           border_mode='valid', name='pool3'))
    model.add(Convolution3D(512, (3, 3, 3), activation='relu', 
                            border_mode='same', name='conv4a',
                            subsample=(1, 1, 1)))
    model.add(Convolution3D(512, (3, 3, 3), activation='relu', 
                            border_mode='same', name='conv4b',
                            subsample=(1, 1, 1)))
    model.add(BatchNormalization())
    model.add(MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2), 
                           border_mode='valid', name='pool4'))
    model.add(Convolution3D(512, (3, 3, 3), activation='relu', 
                            border_mode='same', name='conv5a',
                            subsample=(1, 1, 1)))
    model.add(Convolution3D(512, (3, 3, 3), activation='relu', 
                            border_mode='same', name='conv5b',
                            subsample=(1, 1, 1)))
    model.add(BatchNormalization())
    model.add(MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2), 
                           border_mode='valid', name='pool5'))
    model.add(AveragePooling3D((1, 3, 3), name='avg_pool'))
    model.add(Dropout(.5))
    model.add(Flatten())
    model.add(Dense(10, activation='softmax'))
    
    print(model.summary())
    return model
model = get_model()

import matplotlib.pyplot as plt
from keras.optimizers import SGD
from keras.utils import plot_model
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

learning_rate = 0.001
momentum = 0.9
patienceLR = 3
factor = 0.1
batch_size = 32
epochs = 50

checkpoint = ModelCheckpoint("yoga.h5",
                             monitor = 'val_acc',
                             mode = 'max',
                             save_best_only = True,
                             verbose = 1)

reduce_lr = ReduceLROnPlateau(monitor = 'val_loss',
							  factor = factor,
                              mode = 'min',
							  patience = patienceLR
							  verbose = 1)
						  
callbacks = [checkpoint, reduce_lr]

model.compile(loss = 'categorical_crossentropy', optimizer = SGD(lr=learning_rate, momentum=momentum, nesterov=True), metrics = ['accuracy'])
history = model.fit(X_train, y_train,
                    validation_data = (X_val, y_val),
                    epochs = epochs, batch_size = batch_size,
                    verbose = 1, callbacks = callbacks)
					
N = epochs
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, N), history.history["loss"], label="train_loss")
plt.plot(np.arange(0, N), history.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, N), history.history["acc"], label="train_acc")
plt.plot(np.arange(0, N), history.history["val_acc"], label="val_acc")
plt.title("Training Loss and Accuracy on Dataset")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="lower left")
plt.savefig("plot.png")

prediction = []
prediction.append(model.predict(X_test, batch_size=batch_size, verbose=1))
final = []

for i in range(len(prediction[0])):
	a = prediction[0][i]
	b = np.zeros_like(a)
	max_arg = a.argmax(0)
	for j in range(10):
		if j == max_arg:
			b[j] = 1
		else:
			b[j] = 0
	final.append(b.astype('uint8'))
	
from sklearn.metrics import classification_report
print(classification_report(np.argmax(y_test,axis=1), np.argmax(final, axis=1)))

index_names = ['Garland Pose', 'Happy Baby Pose', 'Head To Knee Pose', 'Lunge Pose', 'Mountain Pose', 'Plank Pose', 'Raised Arms Pose', 'Seated Forward Bend', 'Staff Pose', 'Seated Forward Bend']
from keras.models import load_model
model = load_model('C3D-9113.h5')

name = 'input.mp4'
vs = cv2.VideoCapture(name)
pred = []
results = []
writer = None

while True:
    (grabbed, frame) = vs.read()
    if not grabbed:
        break
    
    output = frame.copy()
    output = cv2.resize(output, (1280, 720), cv2.INTER_AREA)
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame = cv2.resize(frame, (112, 112), cv2.INTER_AREA)
    frame = (frame / 255.).astype('float32')
    pred.append(frame)
    
    if len(pred) == 16:
      p = model.predict(np.expand_dims(pred, axis=0))
      pos = p.argmax()
      if p[0][pos] > 0.5:
        label = index_names[pos]
      
      output = cv2.putText(output, label, (120, 120), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 5)
      pred.pop(0)
    
    if writer is None:
      fourcc = cv2.VideoWriter_fourcc(*'MJPG')
      writer = cv2.VideoWriter('Output '+name, fourcc, 30, (1280, 720), True)
    writer.write(output)

writer.release()
vs.release()