import spacy

from tqdm import tqdm
import codecs as cs
from os.path import join as pjoin
import os
import shutil

nlp = spacy.load('en_core_web_sm')
def process_text(sentence):
    sentence = sentence.replace('-', '')
    doc = nlp(sentence)
    word_list = []
    pos_list = []
    for token in doc:
        word = token.text
        if not word.isalpha():
            continue
        if (token.pos_ == 'NOUN' or token.pos_ == 'VERB') and (word != 'left'):
            word_list.append(token.lemma_)
        else:
            word_list.append(word)
        pos_list.append(token.pos_)
    return word_list, pos_list

def process_humanml3d():
    text_save_path = '/sata/public/yyqi/Dataset/OCEAN/processedText'
    directory = '/sata/public/yyqi/Dataset/OCEAN/text'
    if os.path.exists(text_save_path): # 当文件夹存在时清空文件夹
        shutil.rmtree(text_save_path, True)
        os.makedirs(text_save_path)
    else: # 当文件夹不存在时在当前路径下创建用于存放数据的文件夹
        os.mkdir(text_save_path)

    maxToken = 0
    for filename in os.listdir(directory):
        if filename.endswith('.txt'):
            file_path = os.path.join(directory, filename)
            with open(file_path, 'r', encoding='utf-8') as file:
                for line in file:
                    caption = line.strip().lower()
                    start = 0.0
                    end = 0.0

                    word_list, pose_list = process_text(caption)
                    tokens = ' '.join(['%s/%s'%(word_list[i], pose_list[i]) for i in range(len(word_list))])
                    if len(word_list) > maxToken:
                        maxToken = len(word_list)
                        print(file_path)
                    with cs.open(pjoin(text_save_path, filename), 'a+') as f:
                        f.write('%s#%s#%s#%s\n'%(caption, tokens, start, end))
    print(maxToken)

def singalProcess(filename):
    text_save_path = '/sata/public/yyqi/Dataset/OCEAN/processedText'
    directory = '/sata/public/yyqi/Dataset/OCEAN/text'
    file_path = os.path.join(directory, filename)
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            caption = line.strip().lower()
            start = 0.0
            end = 0.0

            word_list, pose_list = process_text(caption)
            tokens = ' '.join(['%s/%s'%(word_list[i], pose_list[i]) for i in range(len(word_list))])
            with cs.open(pjoin(text_save_path, filename), 'a+') as f:
                f.write('%s#%s#%s#%s\n'%(caption, tokens, start, end))


def process_kitml(corpus):
    text_save_path = './dataset/kit_mocap_dataset/texts'
    desc_all = corpus
    for i in tqdm(range(len(desc_all))):
        caption = desc_all.iloc[i]['desc']
        start = 0.0
        end = 0.0
        name = desc_all.iloc[i]['data_id']
        word_list, pose_list = process_text(caption)
        tokens = ' '.join(['%s/%s' % (word_list[i], pose_list[i]) for i in range(len(word_list))])
        with cs.open(pjoin(text_save_path, name + '.txt'), 'a+') as f:
            f.write('%s#%s#%s#%s\n' % (caption, tokens, start, end))


def process():
    """ 处理原始文本 """
    # directory = '/sata/public/yyqi/Dataset/OCEAN/text'
    # maxToken = 0
    # for filename in os.listdir(directory):
    #     if filename.endswith('.txt'):
    #         file_path = os.path.join(directory, filename)
    #         with open(file_path, 'r', encoding='utf-8') as file:
    #             nums = 0
    #             for line in file:
    #                 caption = line.strip()
    #                 if caption !='' and caption !='\n' and '(' not in caption and '[' not in caption and 'P' not in caption :
    #                     nums+=1
                        
    #                 # start = 0.0
    #                 # end = 0.0

    #                 # word_list, pose_list = process_text(caption)
    #                 # tokens = ' '.join(['%s/%s'%(word_list[i], pose_list[i]) for i in range(len(word_list))])
    #                 # if len(word_list) > maxToken:
    #                 #     maxToken = len(word_list)
    #                 #     print('max'+' '+file_path)
    #         if nums!=3:
    #             print(file_path)
    #             with cs.open('logDataError.txt', 'a+') as f:
    #                 f.write(filename+'\n')

    # print(maxToken)
    """ 统计maxVERB """
    directory = '/sata/public/yyqi/Dataset/OCEAN/processedText'
    minVerb = 100
    maxVerb = 0
    num = 0
    for filename in os.listdir(directory):
        if filename.endswith('.txt'):
            file_path = os.path.join(directory, filename)
            with open(file_path, 'r', encoding='utf-8') as f:
                num = 0
                for line in f.readlines():
                    text_dict = {}
                    line_split = line.strip().split('#')
                    caption = line_split[0]
                    tokens = line_split[1].split(' ')
                    numVerb = 0
                    for token in tokens:
                        try:
                            word, pos = token.split('/')
                            if pos == 'VERB':
                                numVerb += 1 
                        except:
                            with cs.open('logDataError.txt', 'a+') as f:
                                f.write(filename+'\n')
                    if numVerb < minVerb:
                        minVerb = numVerb
                        print('min'+' '+str(minVerb)+' '+file_path)
                    if numVerb > maxVerb:
                        maxVerb = numVerb
                        print('max'+' '+str(maxVerb)+' '+file_path)
                    if numVerb  > num:
                        num = numVerb
    #             if num > 5:
    #                 with cs.open('logActionmax.txt', 'a+') as f:
    #                     f.write(filename+'\n')

                            


if __name__ == "__main__":
    # corpus = pd.read_csv('./dataset/kit_mocap_dataset/desc_final.csv')
    # process_humanml3d(corpus)
    """ 处理成humanml3d表示形式 """
    process_humanml3d()

    """ 异常文件处理 """
    # process()


    """ 单个文件处理 """
    # filename = 'G002T003A016R009P2.txt'
    # singalProcess(filename)