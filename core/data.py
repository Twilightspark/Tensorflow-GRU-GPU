#-*- coding:utf-8 -*-
#author: Kai Zhang
import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import json

from core import db_handle
from conf import setting

def csvread(path):
    '''
    csv read as numpy
    parameter: path: read file path
    return: load data
    '''
    print('正在读取文件：%s'%(path))
    csv_data = pd.read_csv(path,header=None)
    data_set = np.array(csv_data,dtype=float)
    print('读取完毕!')
    return data_set

def csvwrite(path,data_set):
    '''
    numpy write as csv
    parameter: path: write file path
    parameter: data_set: data wants write
    return: None
    '''
    print('正在写文件：%s'%(path))
    csv_pd = pd.DataFrame(data_set)
    csv_pd.to_csv(path,header=False,index=False)
    print('写文件完毕!')
    return None

def normalize(data_set,hyperparameter):
    '''
    prepare data for gru nearul networks
    parameter: data_set: raw data
    return: prepared data
    '''
    row,col = data_set.shape
    input_size = col - 1
    output_size = 1

    for i in range(col):
        data_set[:,i] = (data_set[:,i] - hyperparameter['%s'%i][0])/(hyperparameter['%s'%i][1] - hyperparameter['%s'%i][0])
    
    time_steps = hyperparameter['time_steps']
    hyperparameter['input_size'] = input_size
    hyperparameter['output_size'] = output_size

    data_set_input = np.zeros((row-100,time_steps,input_size))#初始化
    data_set_output = np.zeros((row-100,output_size))#初始化
    for i in range(row-100):
        data_set_input[i] = data_set[i+101-time_steps:i+101,1:]
        data_set_output[i] = data_set[i+100,0]
    return data_set_input,data_set_output,hyperparameter

def acnormalize(data_set,hyperparameter):
    '''
    anti-normalization
    parameter: data_set: raw data
    parameter: hyperparameter: provide max and min number
    return: prepared data
    '''
    data_set = data_set*(hyperparameter['0'][1]-hyperparameter['0'][0])+hyperparameter['0'][0]
    return data_set

def maxmin(train_set,test_set,hyperparameter):
    '''
    find the maximum and minimum
    parameter: train_set & test_set: data_set
    parameter: hyperparameter
    return: None
    '''
    data_set = np.vstack((train_set,test_set))
    max,min = data_set.argmax(axis=0),data_set.argmin(axis=0)
    row,col = data_set.shape
    for i in range(col):
        hyperparameter['%s'%i] = [data_set[min[i],i],data_set[max[i],i]]
    return hyperparameter

def calcula_rmse(actual,predic):
    '''
    calculation rmse
    parameter: actual: actual data
    parameter: predic: predic data
    return: rmse
    '''
    rmse = np.sqrt(np.sum(np.square((actual - predic)))/len(actual))
    return rmse

def calcula_mape(actual,predic):
    '''
    calculation mape
    parameter: actual: actual data
    parameter: predic: predic data
    return: mape
    '''
    mape = np.sum(np.abs((actual - predic)/actual))/len(actual)
    return mape

def plot(actual,predict):
    '''
    plot actual and predict curvel
    parameter: actual: actual data
    parameter: predic: predic data
    return: None
    '''
    plt.plot(actual)
    plt.plot(predict)
    plt.show()

def data_load(data_name):
    '''
    load data set
    parameter: data_name: data name
    return: data_set
    '''
    load_path = db_handle.db_path(setting.DATABASE)
    data_path = r'%s\%s.%s'%(load_path,data_name,setting.DATABASE['type'])
    data_set = csvread(data_path)
    return data_set

def param_load():
    '''
    load hyperparameter
    '''
    load_path = db_handle.db_path(setting.PARAMBASE)
    param_path = r'%s\%s.%s'%(load_path,setting.PARAMBASE['name'],setting.PARAMBASE['type'])
    f = open(param_path,'r')
    hyperparameter = json.loads(f.read())#读
    f.close()
    return hyperparameter

def data_save(name,data_set):
    '''
    save data
    '''
    save_path = db_handle.db_path(setting.DATABASE)
    data_path = r'%s\%s.%s'%(save_path,name,setting.DATABASE['type'])
    csvwrite(data_path,data_set)

def param_save(hyperparameter):
    '''
    save hyperparameter
    '''
    save_path = db_handle.db_path(setting.PARAMBASE)
    param_path = r'%s\%s.%s'%(save_path,setting.PARAMBASE['name'],setting.PARAMBASE['type'])
    f = open(param_path,'w')
    f.write(json.dumps(hyperparameter))#存
    f.close()