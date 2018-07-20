import pandas as pd
import numpy as np 
# def haversine_looping(df):
# disftance_list = []
# for i in range(0,len(df)):
#     disftance_list.append(df.iloc[i][‘high’]/df.iloc[i][‘open’])
#     return disftance_list
#     y_matrix = []
# for i in range(0,len(feature_matrix.loc[x_train.index.append(x_val.index)])) for j in range(0,len(labels_series[x_train.index.append(x_val.index)])):

# feature_matrix.join(df["Prompt"])
df = pd.DataFrame({'A' : ['foo', 'bar', 'foo', 'bar', 'foo', 'bar', 'foo', 'foo'],'B' : ['one', 'one', 'two', 'three','two', 'two', 'one', 'three'],'C' : np.random.randn(8),'D' : np.random.randn(8)})



#if __name__ == '__main__':
print (df)
   # main()
g1 = df.groupby('A')

g2 = g1.count()

g3 = g1.sum()

print (g1)
print (g2)
print (g3)

names = np.array(['Bob', 'Joe', 'Will', 'Bob', 'Will', 'Joe', 'Joe'])
In [177]: np.unique(names)

def Label_matrix(feature_matrix):

	grouped['C'].agg([np.sum, np.mean, np.std])

    c = matrix.loc[matrix["Language"] == "Chinese"]
    e = cf.loc[c.index][0]

    d = label_matrix.T["Chinese"]

def demean(arr):
    return arr-arr.mean()
    a = matrix.groupby("Language").transform(demean)

    先说一个还是从词的角度出发考虑的，最后的效果非常好，就是怎么样从词的向量得到句子的向量，首先选出一个词库，比如说10万个词，然后用w2v跑出所有词的向量，然后对于每一个句子，构造一个10万维的向量，向量的每一维是该维对应的词和该句子中每一个词的相似度的最大值。这种方法实际上是bag of words的一个扩展，比如说对于 我喜欢用苹果手机 这么一句话，对应的向量，会在三星，诺基亚，小米，电脑等词上也会有比较高的得分。这种做法对于bag of words的稀疏性问题效果非常好。还做过一个直接训练句子的相似度的一个query2vec模型，效果也不错，就不细说了。

作者：鲁灵犀
链接：https://www.zhihu.com/question/29978268/answer/55338644
来源：知乎
著作权归作者所有。商业转载请联系作者获得授权，非商业转载请注明出处。

#
model = word2vec.Word2Vec.load(model_path)  
vocab = model.wv.vocab  
word_vector = {}
for word in vocab:    
	word_vector[word] = model[word]
return word_vector

作者：匿名用户
链接：https://www.zhihu.com/question/264117474/answer/313508329
来源：知乎
著作权归作者所有。商业转载请联系作者获得授权，非商业转载请注明出处。