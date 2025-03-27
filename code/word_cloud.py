from data_process import data_process
from wordcloud import WordCloud
import matplotlib.pyplot as plt

word_fre = {}         # 创建一个空字典,用于存储每个词语的频率
adata, data_after_stop, labels = data_process()

for i in data_after_stop[labels == 0]:   # 遍历每条正常短信的分词结果
    for j in i:
        if j not in word_fre.keys():
            word_fre[j] = 1
        else:
            word_fre[j] += 1

mask = plt.imread('duihuakuan.jpg')    # 读取图片文件,作为词云的背景形状
wc = WordCloud(mask=mask, background_color='white', font_path=r'C:\Windows\Fonts\msyh.ttc')
wc.fit_words(word_fre)   # 根据word_fre字典中的词频数据生成词云
plt.imshow(wc)           # 使用 plt.imshow 显示生成的词云图像

plt.show()

