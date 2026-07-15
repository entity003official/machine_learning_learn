'''
    通过knn算法实现鸢尾花的分类操作

    回顾：机器学习项目的研发流程

    

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
import sys
import io
# 解决 Windows 控制台中文乱码问题
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
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
    # print(f'数据集的键对应的值为：\n{iris_data.data[:5]}')   #有150个样本，每个样本有4个特征,我们只看前5个样本的特征数据   
    #5.查看数据集的目标数据
    # print(f'数据集的目标数据为：\n{iris_data.target[:5]}')   #有150个样本，每个样本有一个目标数据，我们只看前5个样本的目标数据

#2. 定义函数，绘制散点图
def draw_scatter():
    #1.加载数据集
    iris_data = load_iris()
    #2.将数据集封装成DataFrame对象
    iris_df = pd.DataFrame(data=iris_data.data, columns=iris_data.feature_names)
    #3给df新增一列，存放目标数据
    iris_df['target'] = iris_data.target
    # print(f'数据集的前5行数据为：\n{iris_df.head()}') 
    #4.通过seaborn绘制散点图
    #hue:分组字段  ，fit_reg:是否绘制回归线
    sns.lmplot(data=iris_df, x='sepal length (cm)', y='sepal width (cm)', hue='target', fit_reg=True)
    #5.设置标题，显示
    plt.title('iris data')
    plt.tight_layout()
    plt.show()
#3.定义函数，切分训练集和测试集
def split_data():
    #1.加载数据集
    iris_data = load_iris()
    #2.切分训练集和测试集，按照8:2的比例切分，设置随机种子为22（随机种子一样则每次生成随机数一样）
    X_train, X_test, y_train, y_test = train_test_split(iris_data.data, iris_data.target, test_size=0.2, random_state=22)
    return X_train, X_test, y_train, y_test
    #3.查看训练集和测试集的形状 
    print(f'训练集的特征数据形状为：{X_train.shape}, 测试集的特征数据形状为：{X_test.shape}')
    print(f'训练集的目标数据形状为：{y_train.shape}, 测试集的目标数据形状为：{y_test.shape}')

#4.定义函数，实现鸢尾花完整案例 ->加载数据，数据预处理，特征工程，模型训练，模型评估，模型预测
def iris_iris_evaluate():
    #1.加载数据集
    iris_data = load_iris()
    #2.数据处理,这里是150条数据
    X_train, X_test, y_train, y_test = train_test_split(iris_data.data, iris_data.target, test_size=0.2, random_state=22)
    #3.特征工程(提取，预处理）
    #思考1：特征提取：因为原数据只有四列特征列，且都是我们用的，所以这里无需做特征提取
    #思考2：特征预处理：因为原数据的特征差值不大，所以这里无需做特征预处理，但是加入特征预处理会让代码更完善，所以加入
    #3.1创建标准化对象
    transfer =  StandardScaler()
    #3.2对特征列标准化，xtrain：训练集的特征数据，xtest：测试集的特征数据
    #fit_transform:先训练(fit)再转化(transform)，适用于第一次进行标准化的时候,用于处理训练集
    x_train = transfer.fit_transform(X_train)
    #transform:直接转化(transform)，适用于已经训练过的标准化对象,用于处理测试集
    x_test = transfer.transform(X_test)
    #4.模型训练 
    #4.1创建模型对象
    estimator = KNeighborsClassifier(n_neighbors=3)  #n_neighbors:表示k值，默认是5,超参数
    #4.2训练
    estimator.fit(x_train, y_train)     #训练

    #5.模型评估
    #场景1： 对刚才切分的 测试集30条进行测试
    # 5.1 预测测试集的目标数据
    y_predict = estimator.predict(x_test)  
    # 5.2 打印预测结果
    print(f'预测的目标数据为：\n{y_predict}') 
    #场景2：对新数据进行预测
    #5.1自定义数据集
    my_data = [[5.1, 3.5, 1.4, 0.2], [6.7, 3.1, 4.7, 1.5], [7.2, 3.6, 6.1, 2.5]]
    #5.2对自定义数据集进行标准化处理
    my_data = transfer.transform(my_data) #不是第一次，所以是transfer不用fit 
    #5.3预测自定义数据集的目标数据 
    y_predict_my = estimator.predict(my_data)
    print(f'自定义数据集的预测结果为：\n{y_predict_my}')
    #5.4查看上述数据集每种分类的预测概率
    estimator.predict_proba(my_data)  #返回每个样本属于每个类别的概率
    y_predict_proba = estimator.predict_proba(my_data)
    print(f'自定义数据集的预测概率为：\n{y_predict_proba}')
    #6.模型预测
    #方式1：直接使用score方法，返回准确率，基于训练集特征和训练集标签
    print(f'模型的准确率为：{estimator.score(x_test, y_test)}')  #score方法内部会调用predict方法
    #方式2：使用accuracy_score方法，返回准确率，基于测试集特征和测试集标签
    print(f'模型的准确率为：{accuracy_score(y_test, y_predict)}')  #accuracy_score方法内部不会调用predict方法


#5.测试
if __name__ == '__main__':
    # load_data()    # 查看数据描述
    # draw_scatter()  # 绘制散点图
    split_data()     # 切分训练集和测试集
    iris_iris_evaluate()  # 鸢尾花完整案例