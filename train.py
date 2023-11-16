
import tensorflow as tf
from keras.callbacks import EarlyStopping, ModelCheckpoint
from utils import get_pose_data
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import coral_ordinal as coral

gru_output_size = 16    #no of nodes
fc1_size = 1

#MODEL 
model = tf.keras.Sequential([        #keras.sequential requires a list input
    tf.keras.layers.GRU(gru_output_size),   #trains model on sequence of trends = distance/angles etc bw keypoints changes
    tf.keras.layers.Dense(fc1_size, activation='sigmoid')
])
                                                           

model.compile(
    optimizer=tf.keras.optimizers.Adam(),   #dy/dx -- y=loss, x=predicted values, x depends of weights
    loss=tf.keras.losses.BinaryCrossentropy(),  #loss = y actual - y predicted
    metrics=[coral.MeanAbsoluteErrorLabels()])


#TRAIN
df = pd.read_csv("dataset/preprocessed/dataset.csv")
print ('Data is read')

x = df['File name']
y = df['fall']

X_train, X_test, y_train, y_test = train_test_split(x,y,
                                   random_state=98,
                                   test_size=0.1,
                                   shuffle=True)
print('Data has been split')

pose_list = []
for name in X_train:
    if 'notfall' in name:
        path = "dataset/input/notfall/videos/"
    else:
        path = "dataset/input/fall/videos/"
    print(path + name +'.csv')
    pose_list.append(tf.convert_to_tensor(pd.read_csv(path + name + '.csv')))

pose_data = tf.stack(pose_list)
pose_label_data = tf.convert_to_tensor(y_train)  

print("type (pose_list)", type (pose_list))
print("tf.shape(pose_data)", tf.shape(pose_data))



#pose_label_data = tf.convert_to_tensor(y, dtype=tf.int64)
print("pose_label_data", pose_label_data)


pose_list_test = []
for name in X_test:
    if 'notfall' in name:
        path = "dataset/input/notfall/videos/"
    else:
        path = "dataset/input/fall/videos/"
    print(path + name +'.csv')
    pose_list_test.append(tf.convert_to_tensor(pd.read_csv(path + name + '.csv')))

pose_data_test = tf.stack(pose_list_test)
pose_label_data_test = tf.convert_to_tensor(y_test) 


cb = [tf.keras.callbacks.EarlyStopping(patience = 15, min_delta = 0.001, restore_best_weights = True)]

print (">>>>>", pose_label_data_test)
# The history variable consists data about the training phase
# See - https://www.tensorflow.org/api_docs/python/tf/keras/callbacks/History
history = model.fit(
    x=pose_data,
    y=pose_label_data,
    epochs= 500,              #how many times the model will be trained for the train set
    batch_size= 16,          #number of videos (sample=video)
    callbacks=cb,
    validation_data=(pose_data_test, pose_label_data_test),
    verbose=1,  #to visually seperate output for easier analysis
)

test_loss, test_accuracy = model.evaluate(pose_data_test, pose_label_data_test)

print(f"Test Accuracy: {test_accuracy * 100:.2f}%")