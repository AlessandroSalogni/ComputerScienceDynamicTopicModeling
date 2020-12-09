import sys, os, math
sys.path.append('..')
from DBConnection.MongoDBConnection import MongoDB
import numpy as np
from scipy import spatial
import matplotlib.pyplot as plt

def getTopicsBagOfWords():
    topicsBagOfWords = []
    for temporalSlice in db.query("TemporalSlice").sort('start', 1):
        for i in range(0, len(temporalSlice['topics'])):
            topicsBagOfWords += [word['word'] for word in temporalSlice['topics'][i]]

    return {item: idx for idx, item in enumerate(set(topicsBagOfWords))}

with MongoDB("TesiMagistrale") as db:
    topicsBagOfWords = getTopicsBagOfWords()

    # Cicla su ogni slice temporale
    for temporalSlice in db.query("TemporalSlice").sort('start', 1):
        yearWordSenseDict = {}
        # Cicla su ogni topic e su ogni parola al suo interno
        for i in range(0, len(temporalSlice['topics'])):
            for selectWord in temporalSlice['topics'][i]:
                # Crea un vettore sparso per ogni parola selezionata che ne rappresenta il senso
                wordSenseSparseVector = np.zeros(len(topicsBagOfWords), dtype=np.uint8)
                for word in temporalSlice['topics'][i]:
                    if word['word'] != selectWord['word']:
                        wordSenseSparseVector[topicsBagOfWords[word['word']]] = 1

                # Se la parola non è ancora presente nel dizionario dei sensi la inserisce come chiave
                if selectWord['word'] not in yearWordSenseDict:
                    yearWordSenseDict[selectWord['word']] = []

                # Inserisce il senso per la parola
                yearWordSenseDict[selectWord['word']].append((i+1, wordSenseSparseVector))

        print(f"\n\n--------- {temporalSlice['start']}-{temporalSlice['end']} ---------\n")
        # Cicla su tutte le parole nel dizionario
        for word, senses in yearWordSenseDict.items():
            if len(senses) > 1:
                print(f"{word}:")

            i = 0
            # Cicla su tutti i sensi della parola a coppie e ne calcola la cosine similarity
            for topic1, sense1 in senses:
                i += 1
                for topic2, sense2 in senses[i:]:
                    print(f"{topic1}-{topic2} -> {1 - spatial.distance.cosine(sense1, sense2)}")