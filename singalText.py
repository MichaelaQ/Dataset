import numpy as np
import os
from sklearn.model_selection import train_test_split

""" 获取对应文件夹内文件名 """
def getName(directory,filePath,suffix):
    directory = directory
    filePath = filePath
    nums = 0 
    if os.path.exists(filePath):
        os.remove(filePath)
    for filename in os.listdir(directory):
        if filename.endswith(suffix):
            baseName = filename.split(suffix)[0]+'\n'        
            with open(filePath, 'a', encoding='utf-8') as file:
                file.write(baseName)
            nums += 1
    return nums

""" 处理OCEAN文件 """
def processOCEAN():

    big_five_path = '/sata/public/yyqi/Dataset/OCEAN/big_five.npy' 
    big_five_data = np.load(big_five_path)  # shape: (118, 5)

    # P_min = big_five_data.min(axis=0)
    # P_max = big_five_data.max(axis=0)
    # big_five_data_norm = 2 * ((big_five_data - P_min) / (P_max - P_min)) - 1


    # def three_way_split(value): #三分化处理
    #     if value < 1/3:
    #         return -1
    #     elif value > 2/3:
    #         return 1
    #     else:
    #         return 0

    # three_way_split_vec = np.vectorize(three_way_split)
    # big_five_data_split = three_way_split_vec(big_five_data)



    # 定义系数矩阵 NPE
    NPE = np.array([
        [-0.921, 0.928, -0.894, 0, -1],
        [0, 0, 0, -1, 0],
        [0, -0.857, 0.99, -1, 0.97],
        [-0.931, 0.938, -1, 0, -0.762]
    ])

    # 初始化Laban因子数组
    laban_factors = np.zeros((big_five_data.shape[0], 4))

    # 计算每个人的Laban因子值
    for idx, P in enumerate(big_five_data_norm):
        E_plus = np.zeros(4)
        E_minus = np.zeros(4)
        for i in range(4):
            positive_contributions = NPE[i] * P
            negative_contributions = NPE[i] * P
            
            positive_contributions = positive_contributions[positive_contributions > 0]
            negative_contributions = negative_contributions[negative_contributions < 0]
            
            if positive_contributions.size > 0:
                E_plus[i] = np.max(positive_contributions)
            if negative_contributions.size > 0:
                E_minus[i] = np.min(negative_contributions)
        
        laban_factors[idx] = E_plus + E_minus

    # 将Laban因子值与大五人格数据连接起来
    combined_data = np.hstack((big_five_data_norm, laban_factors))  # shape: (118, 9)

    # 保存新的数据到npy文件
    output_path = '/sata/public/yyqi/Dataset/OCEAN/big_five_with_laban.npy'  
    np.save(output_path, combined_data)


""" 划分数据集 """
def write_data(datapath, line_sen_list):
    '''
    datapath: 需要写入的文件地址
    line_sen_list: 需要写入的文件内容行列表
    '''
    with open(datapath, 'w', encoding = 'utf-8') as o:
        o.write(''.join(line_sen_list))
        o.close()


def dataSplit(nums):
    raw_data_path = '/sata/public/yyqi/Dataset/OCEAN/all.txt'
    train_data_path = '/sata/public/yyqi/Dataset/OCEAN/train.txt'
    validate_data_path = '/sata/public/yyqi/Dataset/OCEAN/val.txt'
    test_data_path = '/sata/public/yyqi/Dataset/OCEAN/test.txt'
 
    line_sen_list = []
    
    with open(raw_data_path, 'r', encoding = 'utf-8') as f:
        lines = f.readlines()
        # 按某种规律选取固定大小的数据集
        for line in lines:
            line_sen_list.append(''.join(line))
        f.close()
 
    label_list = [0] * nums  # 由于该数据形式为文本，且形式为数据和标签在一起，所以train_test_split()中标签可以给一个相同大小的0值列表，无影响。
 
    # 先将1.训练集，2.验证集+测试集，按照8：2进行随机划分
    X_train, X_validate_test, _, y_validate_test = train_test_split(line_sen_list, label_list, test_size = 0.2, random_state = 42)
    # 再将1.验证集，2.测试集，按照1：1进行随机划分
    X_validate, X_test, _, _ = train_test_split(X_validate_test, y_validate_test, test_size = 0.5, random_state = 42)
    
    # 分别将划分好的训练集，验证集，测试集写入到指定目录
    write_data(train_data_path, X_train)
    write_data(validate_data_path, X_validate)
    write_data(test_data_path, X_test)


""" 获取all.txt """
directory ='/sata/public/yyqi/Dataset/OCEAN/text'
filePath = '/sata/public/yyqi/Dataset/OCEAN/all.txt'

nums = getName(directory,filePath,'.txt')
""" 进行划分 """
dataSplit(nums)