import sys
sys.path.append('..')
from DBConnection.MongoDBConnection import MongoDB
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.tokenize import RegexpTokenizer
from multiprocessing import Process, current_process
import re

class AbstractPreprocessing(object):
    def __init__(self):
        self.lemmatizer = WordNetLemmatizer()
        self.tokenizer = RegexpTokenizer(r"\w+")  # Permette di trasformare un testo in tokens rimuovendo tutta la punteggiatura
        with open("../../Resources/stopwords.txt", 'r', encoding="utf8") as file:
            self.stopwords = file.readline().split(",") # Crea una lista di stopwords andole a recuperare dal file che le contiene

    # Resituisce la lista di tokens dell'abstract (rende minuscolo l'abstract, lo tokenizza e per ogni token, che non appartiene alle stopwords o non è numerico, lo lemmatizza)
    def getAbstractWords(self, abstract):
        return [self.lemmatizer.lemmatize(word) for word in self.tokenizer.tokenize(abstract.lower()) if word not in self.stopwords and not word.isnumeric()]

def preprocessing(startId, endId):
    abstractPreprocessing = AbstractPreprocessing()
    query = {'$and': [{'_id': {'$gt': startId}}, {'_id': {'$lt': endId}}]} if endId is not None else {'_id': {'$gt': startId}} # Query a seconda del processo (l'ultimo processo non ha un id di fine)
    papers = []

    with MongoDB("TesiMagistrale") as db:
        i = 0
        for data in db.query("Papers", query, projection = {'_id': 1, 'year': 1, 'paperAbstract': 1}):
            words = abstractPreprocessing.getAbstractWords(data['paperAbstract']) # Fa la processazione dell'abstract
            papers.append({'_id': data['_id'], 'year': data['year'], 'abstractWords': words}) # Crea e appende il documento da caricare su DB

            i += 1
            if i % 15000 == 0: # Ogni 15000 abstract carica i documenti su DB
                db.insert("ProcessedPapers", papers, many = True)
                print(f"{current_process().name} -> {i} -> {data['_id']}")
                papers = []

        if papers: # Carica i documenti rimanenti
            db.insert("ProcessedPapers", papers, many = True)

if __name__ == '__main__':
    # Creo un array con gli id iniziali di ogni processo
    # ids = ['0000011aa918c023748585e443007bd6532', '55434187640b3d39a78c5e4875be32f7cf3f9224', 'ab277b675a603de0d53d800152ed16cb9eb08de9']
    startIds = ['5543379c39eefe484d8fdba4999b328508a01a72', 'a46187cc7cea80e4ba05979b7ce2a8737269877e', 'f9630a3bf302ea5dfa0312d2a1de3f181fe41b11']
    endIds = ['55434187640b3d39a78c5e4875be32f7cf3f9224', 'ab277b675a603de0d53d800152ed16cb9eb08de9', None]

    # Faccio run dei processi
    for i in range(0, len(startIds)):
        Process(target=preprocessing, args=(startIds[i], endIds[i],)).start()