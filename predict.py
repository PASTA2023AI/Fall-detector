import pandas as pd
import tensorflow as tf
import pickle


Loaded_FallDetector = pickle.load(open('FallDetector.pkl', 'rb'))

holdout = pd.read_csv("dataset/holdout/holdout.csv")
classes = holdout['File name']
holdout_list_data = []

for class1 in classes:
    holdout_list_data.append(tf.convert_to_tensor(pd.read_csv('dataset/holdout/' + class1 + '.csv')))
holdout_data = tf.stack(holdout_list_data)    

pred = Loaded_FallDetector.predict(holdout_data)


for i in range(0, len(pred)):
    if pred[i]>0.5:
        pred[i] = 1
    else:
        pred[i] = 0
    
    if pred[i]==0:
        prediction = 'Not Fall'
    else:
        prediction = 'Fall'  
    print("=="*50)
    print("Prediction - ", prediction)
    print("Actual Class - ", classes[i])
