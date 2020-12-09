import sys
sys.path.append('..')
from DBConnection.MongoDBConnection import MongoDB

with MongoDB("TesiMagistrale") as db:
    for year in range(1978, 1979): # Ciclo per gli anni presi in considerazione
        # Creo una bag of words con tutte le parole usate nei documenti dell'anno selezionato. Le ripetizioni vengono eliminate
        words = list(set([word for data in db.query("ProcessedPapers", query={'year': year}, projection={'abstractWords': 1}) for word in data['abstractWords']]))
        db.insert("YearBagOfWords", {'year': year, 'words': words}) # Carico le parole su DB
        print(f"{year} -> {len(words)}")