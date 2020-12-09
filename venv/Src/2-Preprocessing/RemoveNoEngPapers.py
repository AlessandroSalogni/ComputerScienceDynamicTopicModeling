import sys
sys.path.append('..')
from DBConnection.MongoDBConnection import MongoDB
from langdetect import detect
from multiprocessing import Process, current_process

# Individua la lingua di un paper a partire dall'abstract e se non è inglese lo elimina dal DB
def detectLangPapers(startId, limit):
    with MongoDB("TesiMagistrale") as db:
        removeId = [] # Contiene gli id dei paper da eliminare
        i = 0

        # Cicla sul cursore con i papers, da startId fino a limit
        for data in db.query("Papers", query = {'_id': {'$gt': startId}}, projection = {'paperAbstract': 1}, limit = limit):
            try: # Tenta di identificare la lingua e se non è inglese aggiunge il paper tra quelli da rimuovere
                if detect(data['paperAbstract']) != "en":
                    removeId.append(data['_id'])
            except: # Se non è stato possibile identificare la lingua (es: numero) aggiunge il paper tra quelli da rimuovere
                removeId.append(data['_id'])

            i += 1
            if i % 5000 == 0: # Ogni 5000 paper analizzati faccio una stampa per vedere il progersso
                print(f"{current_process().name} -> {i}")
            if i % 100000 == 0 or i == limit: # Ogni 100000 paper analizzati o al completamento del curosre cancello dal DB o paper presenti in removeId
                db.delete("Papers", {'_id': {'$in': removeId}}, many=True)
                print(f"{current_process().name} -> {data['_id']}")
                removeId = []


if __name__ == '__main__':
    limit = 800000 # Numero di documenti analizzati per processo
    nProcess = 3 # Numero di processi

    # Creo un array con l'id iniziale di ogni processo
    ids = ['c008657db37d0d3f0347139d871a00a9a26b03cc'] # Id del primo documento da trattare (quelli con id inferiore sono già stati trattati)
    with MongoDB("TesiMagistrale") as db:
        for i in range(0, nProcess - 1):
            ids.append(db.query("Papers", query={'_id': {'$gt': ids[len(ids) - 1]}}, skip=limit, limit=1)[0]['_id'])

    # Faccio run dei processi
    for id in ids:
        Process(target=detectLangPapers, args=(id,limit,)).start()
