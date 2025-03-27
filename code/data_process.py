import pandas as pd
import re
import jieba

def data_process(file='message80W1.csv', random_state=26):
    # 读取CSV文件，设置第一列为索引
    data = pd.read_csv(file, header=None, index_col=0)
    data.columns = ['label', 'message']
    n = 5000

    # 随机抽取正常和垃圾短信各n条,（可选）设置随机数种子
    a = data[data['label'] == 0].sample(n, random_state=random_state)
    b = data[data['label'] == 1].sample(n, random_state=random_state)

    # 合并数据并去重
    data_new = pd.concat([a, b], axis=0)
    data_dup = data_new['message'].drop_duplicates()

    # 数据清洗：删除特定字符x
    data_qumin = data_dup.apply(lambda x: re.sub('x', '', x))
    # 或者可以用：data_qumin = data_dup.str.replace('x', '')

    # 加载自定义词典并分词
    jieba.load_userdict('newdic1.txt')
    data_cut = data_qumin.apply(jieba.lcut)

    # 加载停用词表并去除停用词
    stopWords = pd.read_csv('stopword.txt',
                            encoding='utf-8', sep='/n', header=None, engine='python')
    stopWords = ['≮', '≯', '≠', ' ', '会', '月', '日', '–'] + list(stopWords.iloc[:, 0])
    data_after_stop = data_cut.apply(lambda x: [word for word in x if word not in stopWords])

    # 获取标签并拼接文本
    labels = data_new.loc[data_after_stop.index, 'label']
    adata = data_after_stop.apply(lambda x: ' '.join(x))

    return adata, data_after_stop, labels

# 调用数据处理函数
adata, data_after_stop, labels = data_process(random_state=26)

# （可选）保存处理后的数据
adata.to_csv('adata_output.txt', index=False, sep='\t')
data_after_stop.to_csv('data_after_stop.txt', index=False, sep='\t')
