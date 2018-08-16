#-*- coding:utf-8 -*-
#author: Kai Zhang
import numpy as np

from core import db_handle
from core import data
from core import neural

hyperparameter = {
    'time_steps':20,
    'batch_size':100,
    'steps':5000,
    'layers_size':2,
    'hidden_size':20,
    'learning_rate':1e-4,
    'output_size':1,
    'input_size':1,
    }

def interact_parameter(prompt):
    '''
    only accept numbers greater than 0
    parameter: prompt : prompr for number's means
    return: input number(type:int)
    '''
    while True:
        parameter_option = input('请输入%s>>'%prompt)
        if parameter_option.isdigit():
            if int(parameter_option) > 0:
                return int(parameter_option)
            else:
                print('请输入大于0的整数！')
        else:
            print('输入并非纯数字，请重新输入！')

def train():
    '''
    train new neural network with given parameters
    return:Flase
    '''
    global hyperparameter
    train_set = data.data_load('train') #load train data
    test_set = data.data_load('test') #load test data
    hyperparameter = data.maxmin(train_set,test_set,hyperparameter) #find max and min number in test and train set
    train_set_input,train_set_output,hyperparameter = data.normalize(train_set,hyperparameter) #prepare train data
    test_set_input,test_set_output,hyperparameter = data.normalize(test_set,hyperparameter)
    neural.sess_gru(hyperparameter,'train',train_set_input,train_set_output,test_set_input,test_set_output) #train gru
    return False

def train_customize():
    '''
    train new neural network with customize parameters
    return:Flase
    '''
    global hyperparameter
    hyperparameter['time_steps'] = interact_parameter('截断时间')
    hyperparameter['batch_size'] = interact_parameter('批量长度')
    hyperparameter['steps'] = interact_parameter('训练周期')
    hyperparameter['layers_size'] = interact_parameter('网络层数')
    hyperparameter['hidden_size'] = interact_parameter('每层个数')

    return train()

def test():
    '''
    testting test data
    return Flase
    '''
    global hyperparameter
    hyperparameter = data.param_load() #load hyperparameter
    test_set = data.data_load('test') #load test data
    test_set_input,test_set_output,hyperparameter = data.normalize(test_set,hyperparameter) #prepare test data
    neural.sess_gru(hyperparameter,'test',test_set_input,test_set_output) #gru test
    return False

def quit():
    '''
    for quit option, change exit_flag to True
    return:Trun
    '''
    print('感谢使用！')
    return True

def interactive():
    '''
    interact with user
    return:None
    '''
    menu = '''
    1.训练模型(默认)
    2.高级训练(自定义)
    3.测试模型
    4.退出
    '''
    menu_dic = {
        '1': train,
        '2': train_customize,
        '3': test,
        '4': quit,
        }
    exit_flag = False
    while not exit_flag:
        print(menu)
        user_option = input('请输入功能对应序号>>')
        if user_option in menu_dic:
            exit_flag = menu_dic[user_option]()
        else:
            print('请输入正确的序号！')

def run():
    '''
    this function will be called right a way when the program started
    return:None
    '''
    print('训练模型之前请勿测试!')
    interactive()