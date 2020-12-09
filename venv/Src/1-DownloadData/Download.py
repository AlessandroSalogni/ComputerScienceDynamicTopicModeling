import requests, gzip, shutil, os

pathManifets = "../../Resources/"
pathData = pathManifets + "Data/"

download = False
with open(pathManifets + 'manifest.txt', 'r') as manifest:
    # Cicla su tutti i nomi dei file di risorse presenti nel manifest e li scarica a anche
    for filename in manifest:
        if '134' in filename:
            download = True
        # if '134' in filename:
        #     download = False

        if download:
            # Nome del file senza .gz
            filename = filename[:-4]

            # Scarica il file selezionato
            url = 'https://s3-us-west-2.amazonaws.com/ai2-s2-research-public/open-corpus/2020-04-10/' + filename + ".gz"
            r = requests.get(url, allow_redirects=True)
            open(pathData + filename + ".gz", 'wb').write(r.content)

            # Fa l'unzip del file scaricato
            with gzip.open(pathData + filename + ".gz", 'rb') as fIn:
                with open(pathData + filename, 'wb') as fOut:
                    shutil.copyfileobj(fIn, fOut)

            # Elimina il file zippato
            os.remove(pathData + filename + ".gz")
            print(filename)