
import tensorflow as tf
import matplotlib.pyplot as plt
from keras.callbacks import EarlyStopping, ModelCheckpoint
from utils import get_pose_data
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import coral_ordinal as coral

gru_output_size = 50   #no of nodes
fc1_size = 4
fc2_size = 1


#MODEL 
model = tf.keras.Sequential([        #keras.sequential requires a list input
    tf.keras.layers.GRU(gru_output_size),   #trains model on sequence of trends = distance/angles etc bw keypoints changes
    #tf.keras.layers.Dense(fc1_size),
    tf.keras.layers.Dense(fc2_size, activation='sigmoid')
])

        
model.compile(
    optimizer=tf.keras.optimizers.Adam(lr=0.01, beta_1=0.99, beta_2=0.999, epsilon=1e-7),   #dy/dx -- y=loss, x=predicted values, x depends of weights
    loss=tf.keras.losses.BinaryCrossentropy(),  #broadly loss = y actual - y predicted before the model exsists
    metrics=tf.keras.metrics.F1Score())


#TRAIN
df = pd.read_csv("dataset/preprocessed/dataset.csv")
print ('Data is read')

x = df['File name']
y = df['fall']

X_train, X_test, y_train, y_test = train_test_split(x,y,
                                   random_state=10,
                                   test_size=0.2,
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
pose_label_data = tf.convert_to_tensor(y_train, dtype=tf.float32)  

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
pose_label_data_test = tf.convert_to_tensor(y_test, dtype=tf.float32) 


cb = [tf.keras.callbacks.EarlyStopping(patience = 20, min_delta = 0.001, restore_best_weights = True)]


print (">>>>>", pose_label_data_test)
# The history variable consists data about the training phase
# See - https://www.tensorflow.org/api_docs/python/tf/keras/callbacks/History
history = model.fit(
    x=pose_data,
    y=pose_label_data,
    epochs= 500,              #how many times the model will be trained for the train set
    batch_size= 64,          #number of videos (sample=video)
    callbacks=[cb],
    validation_data=(pose_data_test, pose_label_data_test),
    verbose=1,  #to visually seperate output for easier analysis
)

#visualising the loss trends
training_loss = history.history['loss']
test_loss = history.history['val_loss']
training_f1 = history.history['f1_score']
test_f1 = history.history['val_f1_score']

epoch_count = range(1, len(training_loss) + 1)
plt.plot(epoch_count, training_loss, 'r--')
plt.plot(epoch_count, test_loss, 'b-')
plt.legend(['Training Loss', 'Test Loss'])
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.show()

plt.plot(epoch_count, training_f1, 'g--')
plt.plot(epoch_count, test_f1, 'm-')
plt.legend(['Training F1', 'Test F1'])
plt.xlabel('Epoch')
plt.ylabel('F1 Score')
plt.show()

test_loss, test_f1 = model.evaluate(pose_data_test, pose_label_data_test)
print()
print()
print()
print("Test Loss: ", {test_loss})
print("Test F1 Score: ",{float(test_f1) * 100},"%")
model.summary()