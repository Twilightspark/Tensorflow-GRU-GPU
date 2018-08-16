#-*- coding:utf-8 -*-
#author: Kai Zhang

import tensorflow as tf
import numpy as np

from core import data
from conf import setting

def sess_gru(hyperparameter,flag,*args):
    steps = hyperparameter['steps']
    batch_size = hyperparameter['batch_size']
    time_steps = hyperparameter['time_steps']
    input_size = hyperparameter['input_size']
    output_size = hyperparameter['output_size']
    layers_size = hyperparameter['layers_size']
    hidden_size = hyperparameter['hidden_size']
    learning_rate = hyperparameter['learning_rate']
    graph = tf.Graph()
    with graph.as_default():
        x = tf.placeholder(dtype=tf.float32,shape=[None,time_steps,input_size])
        y = tf.placeholder(dtype=tf.float32,shape=[None,output_size])

        cells = []
        for i in range(layers_size):
            gru = tf.nn.rnn_cell.GRUCell(hidden_size,activation=tf.nn.relu)
            cells.append(gru)
        cells = tf.nn.rnn_cell.MultiRNNCell(cells,state_is_tuple=False)

        rnn_output,state = tf.nn.dynamic_rnn(cells,x,dtype=tf.float32)
        hidden_output = rnn_output[:,-1,:]

        w_output = tf.Variable(tf.random_normal([hidden_size,output_size],stddev = 1,seed = 1))
        output = tf.nn.sigmoid(tf.matmul(hidden_output,w_output))
    
        loss = tf.reduce_mean(tf.square(output - y))#定义损失函数
        train_op = tf.train.AdamOptimizer(learning_rate).minimize(loss)#定义反向传播优化方法
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.6)
        init_op = tf.global_variables_initializer()
        saver = tf.train.Saver()
        if flag == 'train':
            train_input,train_output,test_input,test_output = args[0],args[1],args[2],args[3]
            with tf.Session(graph=graph,config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
                sess.run(init_op)
                for i in range(steps):
                    start = i*batch_size%len(train_input)
                    end = min(start+batch_size,len(train_input))
                    sess.run(train_op,feed_dict={x:train_input[start:end],y:train_output[start:end]})
                    if i%500 == 0:
                        total_cross_entropy = sess.run(loss,feed_dict={x:test_input,y:test_output})
                        print('训练%d次后，误差为%f'%(i,total_cross_entropy))
                total_cross_entropy = sess.run(loss,feed_dict={x:test_input,y:test_output})
                print('训练%d次后，误差为%f'%(i+1,total_cross_entropy))
                model_test = sess.run(output,feed_dict={x:test_input})
                model_train = sess.run(output,feed_dict={x:train_input})

                model_train = data.acnormalize(model_train,hyperparameter)
                model_test = data.acnormalize(model_test,hyperparameter)
                train_output = data.acnormalize(train_output,hyperparameter)
                test_output = data.acnormalize(test_output,hyperparameter)
                
                train_rmse = data.calcula_rmse(train_output,model_train)
                train_mape = data.calcula_mape(train_output,model_train)
                test_rmse = data.calcula_rmse(test_output,model_test)
                test_mape = data.calcula_mape(test_output,model_test)
                data.plot(train_output,model_train)
                print('训练RMSE,MAPE为%f,%f,测试为%f,%f'%(train_rmse,train_mape,test_rmse,test_mape))
                while True:
                    parameter_option = input('请输入是否保存神经网络？1.YES 2.NO：>>:')
                    if parameter_option.isdigit():
                        if int(parameter_option) == 1:
                            train_save = np.hstack((train_output,model_train))
                            test_save = np.hstack((test_output,model_test))
                            data.data_save('train_out',train_save)
                            data.data_save('test_out',test_save)
                            data.param_save(hyperparameter)
                            model_path = r'%s\%s\%s.%s'%(setting.MODELBASE['path'],setting.MODELBASE['folder'],setting.MODELBASE['name'],setting.MODELBASE['type'])
                            saver.save(sess,model_path)
                            print('已保存')
                        else:
                            print('未保存')
                        break
                    else:
                        print('请输入数字！')
        elif flag == 'test':
            test_input,test_output = args[0],args[1]
            with tf.Session(graph=graph,config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
                model_path = r'.\%s\%s\%s.%s'%(setting.MODELBASE['path'],setting.MODELBASE['folder'],setting.MODELBASE['name'],setting.MODELBASE['type'])
                saver.restore(sess,model_path)
                model_test = sess.run(output,feed_dict={x:test_input})
                model_test = data.acnormalize(model_test,hyperparameter)
                test_output = data.acnormalize(test_output,hyperparameter)
                test_rmse = data.calcula_rmse(test_output,model_test)
                test_mape = data.calcula_mape(test_output,model_test)
                data.plot(test_output,model_test)
                print('测试为%f,%f'%(test_rmse,test_mape))
                test_save = np.hstack((test_output,model_test))
                data.data_save('test_out',test_save)