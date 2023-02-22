#Bm25
import math
import jieba
class BM25(object):
    def __init__(self, docs):#docs是一个包含所有文本的列表，每个元素是一个文本
        self.D = len(docs)  #总文本数
        self.avgdl = sum([len(doc)+0.0 for doc in docs]) / self.D   #平均文本长度
        self.docs = docs #文本库列表
        self.f = []  # 列表的每一个元素是一个dict，dict存储着一个文档中每个词的出现次数
        self.df = {} # 存储每个词及出现了该词的文档数量
        self.idf = {} # 存储每个词的idf值
        self.k1 = 1.5
        self.b = 0.75
        self.init()
 
    def init(self):
        for doc in self.docs:  #对每个文本
            tmp = {}   #定义一个字典存储词出现频次
            for word in doc:
                tmp[word] = tmp.get(word, 0) + 1  # 存储每个文档中每个词的出现次数
            self.f.append(tmp)
            for k in tmp.keys():
                self.df[k] = self.df.get(k, 0) + 1
        for k, v in self.df.items():
            self.idf[k] = math.log(self.D-v+0.5)-math.log(v+0.5) #计算idf
 
    def sim(self, doc, index):
        score = 0
        for word in doc:
            if word not in self.f[index]:
                continue
            d = len(self.docs[index])
            score += (self.idf[word]*self.f[index][word]*(self.k1+1)
                      / (self.f[index][word]+self.k1*(1-self.b+self.b*d
                                                      / self.avgdl)))
        return score
 
    def simall(self, doc):
        scores = []
        for index in range(self.D):
            score = self.sim(doc, index)
            scores.append(score)
        return scores
 
if __name__ == '__main__':
    sents1 = ["多伦县医院",  #数据库
                "四川绵阳404医院",
               "南召县人民医院"]
    sents2 = ["内蒙古锡林郭勒盟多伦县县医院","绵阳市四零四医院","邓州市人民医院"]#待匹配文本
    doc = []
    for sent in sents1:
        words = list(jieba.cut(sent))
        doc.append(words)
    print(doc)
    s = BM25(doc)
    print(s.f)
    print(s.idf)
    for k in sents2:
        print(s.simall(jieba.lcut(k))) #打印相似度匹配结果