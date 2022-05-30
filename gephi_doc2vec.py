from gensim.models.doc2vec import Doc2Vec
from gensim.models.doc2vec import TaggedDocument
import csv

train_data = []
with open('raw_data/IEEE VIS papers 1990-2020 - Main dataset.csv', encoding="utf-8") as f:
    reader = csv.reader(f)
    data_number = 100
    #print(reader)
    csvdata = [row for row in reader]
    #print(csvdata)
    train_data = [TaggedDocument(words=csvdata[i][8], tags=[csvdata[i][2]]) for i in range(data_number) ]

#print(train_data)
#print(len(train_data))
model = Doc2Vec(documents = train_data)
model.save("model/doc2vec_01.model")