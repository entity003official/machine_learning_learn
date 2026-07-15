'''
    通过knn算法实现鸢尾花的分类操作

    回顾：机器学习项目的研发流程
    1.加载数据
    2.数据处理
    3.特征工程(特征提取、特征处理、特征降维、特征选择、特征组合)
    4.模型训练 
    5.模型评估
    6.模型预测

特征与处理之归一化介绍：
    目的：
        消除特征的量纲（单位）影响，使其具有可比性
        所以通过公式把 各列的值 映射到 [0,1]区间
    公式：
        X' = (当前值X - 该列最小值min) / ( 该列最大值max - 该列最小值min )
        X'' = X' * (b-a) + a  #映射到[a,b]区间
    公式解释：
        1.当前值X：表示该列的某个值
        2.该列最小值min：表示该列的最小值
        3.该列最大值max：表示该列的最大值   
    弊端：
        容易受到最大值和最小值的影响，所以适合小数据集 
'''


#导包
encoding = 'utf-8'
from sklearn.datasets import load_iris      #加载鸢尾花数据集
import seaborn as sns                             #可视化库
import pandas as pd                                 #数据分析库
import matplotlib.pyplot as plt                     #可视化库   
from sklearn.model_selection import train_test_split  #分割训练集和测试集的
from sklearn.preprocessing import StandardScaler    #标准化对象
from sklearn.neighbors import KNeighborsClassifier  #KNN算法 分类对象
from sklearn.metrics import accuracy_score #模型评估指标

#1.定义函数，加载数据集，并查看数据集

def load_data():
    #1.加载数据集
    iris_data = load_iris()
    #2.查看数据集的描述信息
    # print(f'数据集的描述信息为：\n{iris_data}')         #字典形态
    # print(f'数据集的类型为：\n{type(iris_data)}')       #字典形态 
    #3.查看数据集所有的键
    # print(f'数据集的所有键为：\n{iris_data.keys()}')
    #4.查看数据集的键对应的值
    print(f'数据集的键对应的值为：\n{iris_data.data[:5]}')   #有150个样本，每个样本有4个特征,我们只看前5个样本的特征数据   
    #5.查看数据集的目标数据
    print(f'数据集的目标数据为：\n{iris_data.target[:5]}')   #有150个样本，每个样本有一个目标数据，我们只看前5个样本的目标数据

#2. 

#3.

#4.

#5.测试
if __name__ == '__main__':
    load_data()