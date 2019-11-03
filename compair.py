# -*- coding: utf-8 -*-

import pandas as pd
import cv2
import numpy as np
import os
import global_feature_extractor as fe
import pickle as pkl 

#image_path = '/home/mahdi/Pictures/test/1.jpeg'
#image_path = input('please input your image address :  ')
image_size = tuple((500, 500))

with open('features.pkl','rb') as f :
    features = pkl.load(f)


with open('image_path_list.pkl','rb') as f:
    image_path_list = pkl.load(f)
    
    
    
#features = pd.read_csv('features.csv',index_col=[0])
#features = features['features']
fe_obj = fe.Global_feature_extraction()
dir_path = 'dataset/train'
lables = os.listdir(dir_path)
lables.sort()
counter = 0
mse_all = []


def compair(single_feature):
    for i in range(0,len(features)):
        feature = features[i]
        im_path = image_path_list[i]
        feature=np.array(feature)
        mse = np.square(np.subtract(single_feature, feature)).mean()
        mse_all.append([im_path,mse])
    return mse_all



def input_image(image_path):
    Image = cv2.imread(image_path)
    Image = cv2.resize(Image,image_size)
    shape = fe_obj.shape(Image)
    texture   = fe_obj.texture(Image)
    color  = fe_obj.color(Image)
    global_feature = np.hstack([color, texture, shape])
    return global_feature


def run():
    image_path = input('please input your image address :  ')
    single_feature = input_image(image_path)
    mse_all = compair(single_feature)
    df_mse = pd.DataFrame(mse_all,index=[image_path_list])
    df_mse[1] = df_mse[1].sort_index()
    # scale features in the range (0-1)
#    scaler = MinMaxScaler(feature_range=(-1, 1))
#    mse_all_scale = scaler.fit_transform([mse_all])
    return df_mse

    
df_mse = run()


with open('data.pkl','wb') as f:
    pkl.dump(df_mse,f)
print('\n\n[STATUS] :   every thing is OK and image compared .  you best :) ')