import numpy as np
import os
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import re
import itertools


bodyParts = [
    'pelvis',
    'left_hip',
    'right_hip',
    'spine1',
    'left_knee',
    'right_knee',
    'spine2',
    'left_ankle',
    'right_ankle',
    'spine3',
    'left_foot',
    'right_foot',
    'neck',
    'left_collar',
    'right_collar',
    'head',
    'left_shoulder',
    'right_shoulder',
    'left_elbow',
    'right_elbow',
    'left_wrist',
    'right_wrist',
]

def partNameToIndex(partName):
    return bodyParts.index(partName)


def calcAvgSpeed(joints,pose_no_begin,pose_no_end,joint):
    total = 0
    count = 0
    for i in range(pose_no_begin,pose_no_end-1):
        total += speed(joints,i,i+1,joint)
        count += 1
    if count > 0:
        return total / count
    else:
        return total
    
def speed(joints,pose_no_start,pose_no_end,joint_a):
    # return glm.distance(joints[pose_no_start,partNameToIndex(joint_a),:],joints[pose_no_end,partNameToIndex(joint_a),:])
    return np.linalg.norm(joints[pose_no_start,partNameToIndex(joint_a),:]-joints[pose_no_end,partNameToIndex(joint_a),:])


def calcAvgDistance(joints,pose_no_begin,pose_no_end,joint_a,joint_b):
    total = 0
    count = 0
    for i in range(pose_no_begin,pose_no_end):
        total += distance(joints,i,joint_a,joint_b)
        count += 1
    if count > 0:
        return total / count
    else:
        return total

def distance(joints,pose_no,joint_a,joint_b):
    jointA = joints[pose_no,partNameToIndex(joint_a),:]
    jointB = joints[pose_no,partNameToIndex(joint_b),:]
    
    return np.linalg.norm(jointA-jointB)

def calcAvgAngle(joints,pose_no_begin,pose_no_end,joint_a,joint_cent,joint_b):
    total = 0
    count = 0
    for i in range(pose_no_begin,pose_no_end):
        total += angle(joints,i,joint_a,joint_cent,joint_b)
        count += 1
    if count > 0:
        return total / count
    else:
        return total


def angle(joints,pose_no,joint_a,joint_cent,joint_b):
    posCent = joints[pose_no,partNameToIndex(joint_cent),:]
    v1 = posCent - joints[pose_no,partNameToIndex(joint_a),:]
    v2 = posCent - joints[pose_no,partNameToIndex(joint_b),:]
    # return glm.acos(glm.dot(glm.normalize(v1),glm.normalize(v2)))
    v1_norm = v1 / np.linalg.norm(v1)
    v2_norm = v2 / np.linalg.norm(v2)
    return np.arccos(np.clip(np.dot(v1_norm, v2_norm), -1.0, 1.0))

def calLMA(joints):
    size = 20
    arr = np.zeros((15,size))
    frame_count = joints.shape[0]
    num_of_frames_per_element = int(frame_count / size)

    for i in range(0,size):
        arr[0,i] = calcAvgDistance(joints,i*num_of_frames_per_element,(i+1)*num_of_frames_per_element,'left_wrist','right_wrist')
        arr[1,i] = calcAvgDistance(joints,i*num_of_frames_per_element,(i+1)*num_of_frames_per_element,'left_ankle','right_ankle')
        arr[2,i] = calcAvgDistance(joints,i*num_of_frames_per_element,(i+1)*num_of_frames_per_element,'head','left_wrist')
        arr[3,i] = calcAvgDistance(joints,i*num_of_frames_per_element,(i+1)*num_of_frames_per_element,'head','right_wrist')


        arr[4,i] = calcAvgSpeed(joints,i*num_of_frames_per_element,(i+1)*num_of_frames_per_element,'left_foot')
        arr[5,i] = calcAvgSpeed(joints,i*num_of_frames_per_element,(i+1)*num_of_frames_per_element,'right_foot')

        arr[6,i] = calcAvgAngle(joints,i*num_of_frames_per_element,(i+1)*num_of_frames_per_element,'pelvis','spine3','head')
        arr[7,i] = calcAvgAngle(joints,i*num_of_frames_per_element,(i+1)*num_of_frames_per_element,'left_shoulder','left_elbow','left_wrist')
        arr[8,i] = calcAvgAngle(joints,i*num_of_frames_per_element,(i+1)*num_of_frames_per_element,'right_shoulder','right_elbow','right_wrist')
        arr[9,i] = calcAvgAngle(joints,i*num_of_frames_per_element,(i+1)*num_of_frames_per_element,'left_hip','left_knee','left_ankle')
        arr[10,i] = calcAvgAngle(joints,i*num_of_frames_per_element,(i+1)*num_of_frames_per_element,'right_hip','right_knee','right_ankle')
        arr[11,i] = calcAvgAngle(joints,i*num_of_frames_per_element,(i+1)*num_of_frames_per_element,'left_wrist','pelvis','right_wrist')
        arr[12,i] = calcAvgAngle(joints,i*num_of_frames_per_element,(i+1)*num_of_frames_per_element,'left_elbow','pelvis','right_elbow')
        arr[13,i] = calcAvgAngle(joints,i*num_of_frames_per_element,(i+1)*num_of_frames_per_element,'left_knee','pelvis','right_knee')
        arr[14,i] = calcAvgAngle(joints,i*num_of_frames_per_element,(i+1)*num_of_frames_per_element,'left_ankle','pelvis','right_ankle')
    return arr


listFile = '/sata/public/yyqi/Dataset/OCEAN/test.txt'
npy_files_path = '/sata/public/yyqi/Dataset/interx/new_joints'


with open(listFile, 'r') as f:
    fileNames = f.read().splitlines()


ocean = np.load('/sata/public/yyqi/Dataset/OCEAN/binarized_big_five.npy')

ocean_combinations = list(itertools.product([-1, 1], repeat=5))
ocean_combinations_dict = {tuple(comb): idx for idx, comb in enumerate(ocean_combinations)}

# 存储所有数据的列表
data_list = []
labels_list = []

# 读取所有npy文件并提取特征
for file in fileNames:
    npy_data = np.load(os.path.join(npy_files_path, file+'.npy'))


    features = calLMA(npy_data).flatten()
   
    gNumber = re.search(r'G(\d+)', file).group(1)
    pNumber = re.search(r'P(\d+)', file).group(1)
    oceanID = (int(gNumber)-1)*2 + int(pNumber) - 1


    #32种组合混杂在一起的效果
    # label = ocean_combinations_dict[tuple(ocean[oceanID])]
    # labels_list.append(label)


    #只处理E
    if ocean[oceanID][2] == 1:
        labels_list.append(0)
    else:
        labels_list.append(1)

    data_list.append(features)
    print(ocean[oceanID])
    NPE = np.array([
            [-0.921, 0.928, -0.894, 0, -1],
            [0, 0, 0, -1, 0],
            [0, -0.857, 0.99, -1, 0.97],
            [-0.931, 0.938, -1, 0, -0.762]
        ])

    contributions =  ocean[oceanID] * NPE


    positive_contributions = np.where(contributions > 0, contributions, -np.inf)
    negative_contributions = np.where(contributions < 0, contributions, np.inf)

    E_plus = np.max(positive_contributions, axis=1)
    E_minus = np.min(negative_contributions, axis=1)
    E_plus = np.where(E_plus == -np.inf, 0, E_plus)
    E_minus = np.where(E_minus == np.inf, 0, E_minus)

    LE = E_plus + E_minus #[bs,4]
    print(LE)
    print('/sata/public/yyqi/Dataset/OCEAN/text/'+ file + '.txt')


# 转换为numpy数组
data_array = np.array(data_list)
labels_array = np.array(labels_list)

# 使用t-SNE进行降维
tsne = TSNE(n_components=2, random_state=42)
tsne_results = tsne.fit_transform(data_array)

# 可视化t-SNE结果
plt.figure(figsize=(10, 6))
scatter = plt.scatter(tsne_results[:, 0], tsne_results[:, 1], c=labels_array, cmap='tab20', alpha=0.7)
plt.colorbar(scatter, label='OCEAN Values')
plt.title('t-SNE Visualization of npy Files Based on OCEAN Values')
plt.xlabel('t-SNE Dimension 1')
plt.ylabel('t-SNE Dimension 2')
plt.show()

print(labels_array)
