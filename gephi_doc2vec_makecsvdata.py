from gensim.models.doc2vec import Doc2Vec
from gensim.models.doc2vec import TaggedDocument
import csv

model = Doc2Vec.load('model/doc2vec_01.model')
papers_name = model.dv.index_to_key.copy()
#print(papers_name)


#min_val = 100
#max_val = -100
#th_val = 0.95

#nodeデータの作成
with open('data/doc2vec_nodedata_01.csv', 'w', newline="") as f:
    writer = csv.writer(f)
    writer.writerow(['Id', 'Label'])
    
    for i, name in enumerate(papers_name):
        writer.writerow([i,name])


#edgeデータの作成
with open('data/doc2vec_edgedata_01.csv', 'w', newline="") as f:
    writer = csv.writer(f)
    writer.writerow(['Source', 'Target', 'Cosine_similarity'])
    
    for i in range(len(papers_name)):
        for j in range(len(papers_name)):
            if(i != j):
                cos_sim = model.dv.similarity(papers_name[i], papers_name[j])
                writer.writerow([i, j, cos_sim])
#for i in model.dv.index_to_key:
#    for j in model.dv.index_to_key:
#        sim = model.dv.similarity(i, j)
#        min_val = min(min_val, sim)
#        max_val = max(max_val, sim)
#print(max_val)
#print(min_val)
