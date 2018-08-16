#-*- coding:utf-8 -*-
#author: Kai Zhang
#handle database

def db_path(data_param):
    '''
    find the path to load
    return path
    '''
    path = r'%s\%s'%(data_param['path'],data_param['folder'])
    return path