import sys
sys.path.append('..')
from DBConnection.MongoDBConnection import MongoDB
import json, time, glob

start_time = time.time()

# Cicla su tutti i file di dati presenti nella cartella
for filePath in glob.glob("../../Resources/Data/*"):
    computerSciencePapers = []

    # Per ogni file selezionato filto tutti i papers in ambito di Computer Science
    with open(filePath, 'r', encoding="utf8") as file:
        for line in file:
            jsonLine = json.loads(line)
            if "Computer Science" in jsonLine["fieldsOfStudy"] and jsonLine["paperAbstract"]:
                jsonLine['_id'] = jsonLine.pop('id')
                computerSciencePapers.append(jsonLine)

    # Carico su DB i papers selezionati dal file
    with MongoDB("TesiMagistrale") as db:
        db.insert("Papers", computerSciencePapers, True)

    print(filePath)

elapsed_time = time.time() - start_time
print(elapsed_time)
