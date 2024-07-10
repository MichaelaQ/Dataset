import numpy as np
import os

""" 获取对应文件夹内文件名 """
# directory = '/sata/public/yyqi/Dataset/OCEAN/texts/'
# fileText = '/sata/public/yyqi/Dataset/OCEAN/train.txt'
# for filename in os.listdir(directory):
#     if filename.endswith('.txt'):
#         baseName = filename.split('.txt')[0]+'\n'        
#         with open(fileText, 'a', encoding='utf-8') as file:
#             file.write(baseName)

""" 处理OCEAN文件 """
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


