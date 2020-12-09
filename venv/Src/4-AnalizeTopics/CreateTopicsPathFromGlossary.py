import sys, os, math, time
sys.path.append('..')
from DBConnection.MongoDBConnection import MongoDB
from graphviz import Digraph

with MongoDB("TesiMagistrale") as db:
    filename = 'Security'
    with open(f"../../Resources/Glossary/{filename}.txt", 'r', encoding="utf8") as file:
        glossary= file.readline().rstrip().split(",")

    # Crea il grafo dei topic path
    graph = Digraph()
    reset = False
    prevNodes = []
    newNodes = []
    # Cicla su tutti i topic e cerca quelli che contengono le parole indicate sul file
    for temporalSlice in db.query("TemporalSlice").sort('start', 1):
        # print(f"{temporalSlice['start']}-{temporalSlice['end']}")
        for i in range(len(temporalSlice['topics'])):
            totWordsInGlossaryFirst = sum([1 for word in temporalSlice['topics'][i] if word['word'] in glossary])

            # Se sono presenti tutte le parole aggiunge il topic al grafo
            if totWordsInGlossaryFirst == len(glossary):
                nodeLabel = f"{temporalSlice['start']}-{temporalSlice['end']} topic {i + 1}"
                graph.node(nodeLabel, nodeLabel)

                for prevNode in prevNodes:
                    graph.edge(prevNode, nodeLabel)

                reset = True
                newNodes.append(nodeLabel)
                # print(f"topic {i+1}")
        # print()

        if reset:
            prevNodes = newNodes
            newNodes = []
            reset = False

    graph.render(f"../../Resources/Glossary/{filename}.dot")