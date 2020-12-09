import sys
sys.path.append('..')
from DBConnection.MongoDBConnection import MongoDB
from sklearn.decomposition import LatentDirichletAllocation
from scipy.sparse import coo_matrix
import numpy as np
import time, glob
from multiprocessing import Process, current_process


start_time = time.time()

# Gestisce le slice temporali
class TemporalSlice(object):
    def __init__(self, start, rangeYear):
        self.start = start
        self.end = start+rangeYear-1
        self.bagOfWords = self._getBagOfWords()

    # Ritorna tutte le parole utilizzate negli anni della slice temporale in un dizionario parola:indice
    def _getBagOfWords(self):
        words = []
        with MongoDB("TesiMagistrale") as db:
            for year in range(self.start, self.end+1):
                words = list(set(words + [word for data in db.query("YearBagOfWords", query={'year': year}, projection={'words': 1}) for word in data['words']]))

        words.sort()
        return {item: idx for idx, item in enumerate(words)}
    
    # Ritorna una matrice sparsa con il chunck successivo per un apprendimento incrementale
    def nextSparseMatrixChunk(self, batchSize):
        i = 0
        abstractSparseMatrix = []
        with MongoDB("TesiMagistrale") as db:
            for year in range(self.start, self.end+1):
                # Per ogni abstract di un determinato anno calcolo un vettore sparso con il conteggio dell'utilizzazione di ogni parola
                for data in db.query("ProcessedPapers", query={'year': year}, projection={'abstractWords': 1}):
                    abstractSparseVector = np.zeros(len(self.bagOfWords), dtype=np.uint8)
                    for word in data['abstractWords']:
                        abstractSparseVector[self.bagOfWords[word]] += 1

                    abstractSparseMatrix.append(abstractSparseVector)

                    i += 1
                    if i % batchSize == 0:
                        yield coo_matrix(abstractSparseMatrix)
                        abstractSparseMatrix = []

        return coo_matrix(abstractSparseMatrix)
    
    # Ritorna una matrice sparsa completa
    def getSparseMatrix(self):
        abstractSparseMatrix = []
        for chunk in self.nextSparseMatrixChunk(10000):
            abstractSparseMatrix += chunk
        return abstractSparseMatrix

# Costruisce il vettore di topics, con parola e probabilità di comparire nel topic per ognuno
def getTopics(model, feature_names, no_top_words):
    topics = []
    nTopic = 0
    normalizeComponents = model.components_ / model.components_.sum(axis=1)[:, np.newaxis]
    for topic_idx, topic in enumerate(normalizeComponents):
        topics.append([{"word": feature_names[i], "prob": normalizeComponents[nTopic][i]} for i in topic.argsort()[:-no_top_words - 1:-1]])
        nTopic += 1
    return topics

# Estrae i topics con un apprendimento incrementale con batch da 1000 papers alla volta
def extractTopics(start, stop):
    rangeYear = 3
    batchSize = 1000
    nTopics = 20
    nWords = 30

    i=0
    for year in range(start, stop, rangeYear):
        temporalSlice = TemporalSlice(year, rangeYear) # Crea la slice temporale

        lda = LatentDirichletAllocation(n_components=nTopics, random_state=0, batch_size=batchSize, learning_method='online', total_samples=2000000)
        # Cicla sui chunck restituiti dalla slice temporale e addestra l'algoritmo lda incrementalmente
        for chunk in temporalSlice.nextSparseMatrixChunk(batchSize):
            lda.partial_fit(chunk)
            i+=1
            print(f"{current_process().name} -> chunk -> {i}")

        topics = getTopics(lda, list(temporalSlice.bagOfWords), nWords)
        print(f"{current_process().name} -> {year} - {year+2}")
        
        # Carica i topics su db
        with MongoDB("TesiMagistrale") as db:
            db.insert("TemporalSlice", {'start': temporalSlice.start, 'end': temporalSlice.end, 'range': rangeYear, 'topics': topics})

    elapsed_time = time.time() - start_time
    print(elapsed_time)

if __name__ == '__main__':
    startYear = [2003, 2006]
    endYear = [2005, 2008]

    # Faccio run dei processi
    for i in range(0, len(startYear)):
        Process(target=extractTopics, args=(startYear[i], endYear[i],)).start()