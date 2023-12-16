from utils import get_pose_data
import pandas as pd
import numpy as np
import os

FPS = 14
DURATION = 2.5


#Creating csv files for each video    
def preprocess():
    dfs = []
    for dir in ['fall', 'notfall']:
        df = pd.read_csv(f'dataset/input/{dir}/dataset.csv')
        for i, row in df.iterrows():
            print('Index', i, '; x1 =', row['File name'])    #for keeping track of videos
            path = f'dataset/input/{dir}/videos/'
            print(path + row['File name'] +'.mp4')
            keypoints = get_pose_data(FPS, DURATION, path + row['File name'] + '.mp4', True, False) #asigning dataframe created to variable
            if keypoints.shape == (34, 132):
                os.makedirs(path, exist_ok=True)     #checks for dir for video csv
                keypoints.to_csv(path + row['File name'] + '.csv', index=False)
                # Add a column 'fall' with value 1 for a fall or 0 otherwise
                df.at[i, 'fall'] = '1' if dir == 'fall' else '0'
                print()
            else:
                df.drop(i, inplace=True)
        dfs.append(df)
    # combine the names' dataframes
    combined_df = pd.concat(dfs, ignore_index=True)
    # Write to a csv file
    combined_df.to_csv('dataset/preprocessed/dataset.csv')

