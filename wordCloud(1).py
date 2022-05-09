import pandas as pd
import os
import jieba
from wordcloud import WordCloud
# from scipy.misc import imread
import imageio
import cv2
import matplotlib.pyplot as plt

def generate():
    # 读取原始训练数据
    readers = pd.read_csv(os.path.join('data', 'weibo_senti_100k.csv'), chunksize=1000, delimiter=',')
    reviews = []
    for reader in readers:
        # 使用extend方法逐个追加到reviews中
        reviews.extend(list(reader.review.values))
    with open('cloud/weibo_corpus_zh.txt', 'w', encoding='utf-8') as f:
        for review in reviews:
            f.write(review + '\n')
    print('weibo_scorpus_zh.txt创建成功')


def read_deal_text():
    with open("cloud/weibo_corpus_zh.txt", "r") as f:  # 读取我们的待处理本文
        txt = f.read()

    re_move = ["，", "。",'\n', '\xa0']  # 无效数据
    # 去除无效数据
    for i in re_move:
        txt = txt.replace(i, " ")
    word = jieba.lcut(txt)  # 使用精确分词模式进行分词后保存为word列表
    with open("cloud/txt_save.txt",'w') as file:
        for i in word:
            file.write(str(i)+' ')
    print("文本处理完成")


def img_grearte():

    mask = cv2.imread('mask.jpg')
    with open("cloud/txt_save.txt", "r") as file:
        txt = file.read()
    word = WordCloud(background_color="white", \
                     width=800, \
                     height=800,
                     font_path='simsun.ttf',#zi ti
                     mask=mask,
                     ).generate(txt)
    word.to_file('test.png')
    print("词云图片已保存")

    plt.imshow(word)  # 使用plt库显示图片
    plt.axis("off")
    plt.show()




if __name__=="__main__":
    generate()
    read_deal_text()
    img_grearte()