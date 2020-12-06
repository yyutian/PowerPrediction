import pandas as pd
import numpy as np
import os
import re
import math
filepath = '../datasets/FillData'
station = pd.read_csv('../datasets/station.csv',encoding='gbk')
#计算特征和类的平均值
def calcMean(x,y):
    sum_x = sum(x)
    sum_y = sum(y)
    n = len(x)
    x_mean = float(sum_x+0.0)/n
    y_mean = float(sum_y+0.0)/n
    return x_mean,y_mean
#计算Pearson系数
def calcPearson(x,y):
    x_mean,y_mean = calcMean(x,y)	#计算x,y向量平均值
    n = len(x)
    sumTop = 0.0
    sumBottom = 0.0
    x_pow = 0.0
    y_pow = 0.0
    for i in range(n):
        sumTop += (x[i]-x_mean)*(y[i]-y_mean)
    for i in range(n):
        x_pow += math.pow(x[i]-x_mean,2)
    for i in range(n):
        y_pow += math.pow(y[i]-y_mean,2)
    sumBottom = math.sqrt(x_pow*y_pow)
    p = sumTop/sumBottom
    return p

startDate = '2018-01-01'
endDate = '2018-01-20'
if __name__=='__main__':
    #list = ['潘家','刘打磨','土马店','殷巷镇逯家','小马家','玉皇庙村','张六真','温王','刘集','文丰梁','殷巷镇崔家']
    list = ['潘家', '刘打磨', '土马店', '殷巷镇逯家', '小马家', '玉皇庙村', '张六真', '孙家', '许家', '平家', '殷巷镇崔家']
    list.remove('殷巷镇崔家')
    id = {}
    id[int(station[station['title'] == '殷巷镇崔家']['changZhanId'])] = '殷巷镇崔家'
    for i in range(len(station)):
        if station.iloc[i]['title'] in list:
            id[station.iloc[i]['changZhanId']] = station.iloc[i]['title']
    print(id.keys(),id.values())
    power = pd.DataFrame(columns=id.values())
    for file in os.listdir(filepath):
        no = int(re.findall(r"\d+\d*",file)[0])
        if no in id.keys():
            data = pd.read_csv(filepath + '/' + file)['dayPower']
            power[id[no]] = data
    power = power[['潘家', '刘打磨', '土马店', '殷巷镇逯家', '小马家', '玉皇庙村', '张六真', '孙家', '许家', '平家', '殷巷镇崔家']]
    for l in list:
        p = calcPearson(power[l].iloc[:200],power['殷巷镇崔家'].iloc[:200])
        print(l + ':',p)