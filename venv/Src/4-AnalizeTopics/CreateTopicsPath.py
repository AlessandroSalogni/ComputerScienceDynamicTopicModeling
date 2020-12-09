import sys, os, math, time
os.environ["PATH"] += os.pathsep + 'C:/Program Files/Graphviz 2.44.1/bin/'
sys.path.append('..')
from DBConnection.MongoDBConnection import MongoDB
from scipy import spatial
from collections import Counter
from wordcloud import WordCloud
from fpdf import FPDF
from nltk.corpus import wordnet as wn
from graphviz import Digraph
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import statistics
import psutil

class Topic():
    def __init__(self, words):
        self.words = words

    def getTopicSparseVector(self, topicsBagOfWords):
        wordSenseSparseVector = np.zeros(len(topicsBagOfWords), dtype=np.uint8)

        for word in self.words:
            wordSenseSparseVector[topicsBagOfWords[word['word']]] = 1

        return wordSenseSparseVector

class TemporalSliceLinks():
    def __init__(self, temporalSlice1, temporalSlice2):
        self.temporalSlice1 = temporalSlice1
        self.temporalSlice2 = temporalSlice2

    def getLinks(self, topicsBagOfWords, threshold):
        links = []
        for i in range(0, len(self.temporalSlice1['topics'])):
            for j in range(0, len(self.temporalSlice2['topics'])):
                if 1 - spatial.distance.cosine(
                        Topic(self.temporalSlice1['topics'][i]).getTopicSparseVector(topicsBagOfWords),
                        Topic(self.temporalSlice2['topics'][j]).getTopicSparseVector(topicsBagOfWords)) > threshold:
                    links.append((i + 1, j + 1))

        return links

    def getLinksSimilaritySum(self,  topicsBagOfWords, threshold):
        linksSimilaritySum = 0
        for i in range(0, len(self.temporalSlice1['topics'])):
            for j in range(0, len(self.temporalSlice2['topics'])):
                similarity = 1 - spatial.distance.cosine(
                        Topic(self.temporalSlice1['topics'][i]).getTopicSparseVector(topicsBagOfWords),
                        Topic(self.temporalSlice2['topics'][j]).getTopicSparseVector(topicsBagOfWords))

                if similarity > threshold:
                    linksSimilaritySum += similarity

        return linksSimilaritySum

    def getBestThreshold(self, topicsBagOfWords):
        maxScore = -1
        bestThreshold = 0
        for threshold in np.arange(0.35, 0.8, 0.05):
            links = self.getLinks(topicsBagOfWords, threshold)

            if len(links) > 0:
                score = len(links) / Counter([link[1] for link in links]).most_common(1)[0][1]

                if score > maxScore:
                    maxScore = score
                    bestThreshold = threshold

        return bestThreshold

class TopicWordsColor():
    def __init__(self, currentTopic, prevTopic = [], nextTopic = []):
        self.currentTopic = [word['word'] for word in currentTopic]
        self.prevTopic = [word['word'] for word in prevTopic]
        self.nextTopic = [word['word'] for word in nextTopic]

    def getColorToWordsDict(self):
        appearWords = list(set(self.currentTopic) - set(self.prevTopic))
        disappearWords = list(set(self.currentTopic) - set(self.nextTopic))
        onlyCurrentWords = list(set(appearWords) & set(disappearWords))

        return {'red': disappearWords, 'green': appearWords, 'orange': onlyCurrentWords}

    def getColorToWordsList(self, topicsBagOfWords):
        colorVector = np.full(len(topicsBagOfWords), 'white', dtype=np.object)
        wordsColored = []
        for color, words in self.getColorToWordsDict().items():
            for word in words:
                wordsColored.append(word)
                colorVector[topicsBagOfWords[word]] = "green" if color == "red" else color

        for wordNotColored in list(set(self.currentTopic) - set(wordsColored)):
            colorVector[topicsBagOfWords[wordNotColored]] = 'green'

        return colorVector

class TopicPath():
    def __init__(self, start, path):
        self.start = start
        self.path = path

    def _getTopicPathBagOfWords(self, order="alphabetical"):
        if order == "alphabetical":
            topicPathBagOfWords = sorted(set([word['word'] for j in range(len(self.path)) for word in self._getTopic(j)]))
        elif order == "reverse":
            topicPathBagOfWords = []
            for j in range(len(self.path)):
                wordsInTopic = [word['word'] for word in self._getTopic(len(self.path)-(j+1)) if word['word'] not in topicPathBagOfWords]
                topicPathBagOfWords += wordsInTopic
        elif order == "relevance":
            wordsCounter = Counter([word['word'] for j in range(len(self.path)) for word in self._getTopic(j)])
            topicPathBagOfWords = sorted(wordsCounter, key=wordsCounter.get, reverse=True)

        return {item: idx for idx, item in enumerate(topicPathBagOfWords)}

    def _getTopic(self, idx):
        return list(db.query("TemporalSlice", query={'start': self.start + idx*3}))[0]['topics'][self.path[idx] - 1]

    def _getBestSimilarity(self, word1, word2):
        bestSimilarity = 0
        for synset1 in wn.synsets(word1):
            for synset2 in wn.synsets(word2):
                similarity = synset1.wup_similarity(synset2)

                if similarity != None and similarity > bestSimilarity:
                    bestSimilarity = similarity

        return bestSimilarity

    def getGraph(self, minLength):
        if len(self.path) < minLength:
            return

        for i in range(len(self.path)):
            nodeLabel = f"{self.start + i*3}-{self.start + i*3 +2} topic {self.path[i]}"
            if nodeLabel not in existingNodeDict:
                existingNodeDict[nodeLabel] = []
                graph.node(nodeLabel, nodeLabel)
            if i < len(self.path)-1:
                nodeLabelEdge = f"{self.start + (i+1)*3}-{self.start + (i+1)*3 +2} topic {self.path[i+1]}"
                if nodeLabelEdge not in existingNodeDict[nodeLabel]:
                    existingNodeDict[nodeLabel].append(nodeLabelEdge)
                    graph.edge(nodeLabel, nodeLabelEdge)

    def getSynsetTopicSimilarity(self, minLength):
        if len(self.path) < minLength:
            return -1

        topicPairSimilarityList = []
        for j in range(len(self.path) - 1):
            currentTopic = [word['word'] for word in self._getTopic(j)]
            nextTopic = [word['word'] for word in self._getTopic(j+1)]

            currentTopicWithoutNext = list(set(currentTopic) - set(nextTopic))
            nextTopicWithoutCurrent = list(set(nextTopic) - set(currentTopic))

            for currentWord in currentTopicWithoutNext:
                for nextWord in nextTopicWithoutCurrent:
                    topicPairSimilarityList.append(self._getBestSimilarity(currentWord, nextWord))

        return statistics.mean(topicPairSimilarityList)

    def createTableVisualization(self, minLength, order="alphabetical"):
        if len(self.path) < minLength:
            return

        topicPathBagOfWords = self._getTopicPathBagOfWords(order)

        topicsSparseVector = []
        topicsColorVector = []
        for j in range(len(self.path)):
            currentTopic = self._getTopic(j)
            prevTopic = self._getTopic(j-1) if j > 0 else []
            nextTopic = self._getTopic(j+1) if j < len(self.path) - 1 else []

            topicsSparseVector.append(Topic(currentTopic).getTopicSparseVector(topicPathBagOfWords))
            topicsColorVector.append(TopicWordsColor(currentTopic, prevTopic, nextTopic).getColorToWordsList(topicPathBagOfWords))

        fig = go.Figure(data=[go.Table(
            header=dict(values=[""] + [f"{self.start + j*3}-{self.start + j*3+2} topic {self.path[j]}" for j in range(len(self.path))],
                        fill_color='paleturquoise', align='center', font_size=12),
            cells=dict(values=[list(topicPathBagOfWords.keys())] + [['' for value in sparseVector] for sparseVector in topicsSparseVector],
                       fill_color=[['lavender'] * len(list(topicPathBagOfWords.keys()))] + topicsColorVector, align='center', font_size=12))
        ])
        fig.update_layout(width=1000, height=2100)
        fig.show()

    def createWordCloudVisualization(self, pdf, minLength):
        if len(self.path) < minLength:
            return

        for j in range(len(self.path)):
            currentTopic = self._getTopic(j)
            prevTopic = self._getTopic(j - 1) if j > 0 else []
            nextTopic = self._getTopic(j + 1) if j < len(self.path) - 1 else []

            frequencies = {word['word']: word['prob'] for word in currentTopic}

            wordcloud = WordCloud(width=800, height=800, background_color ='white', min_font_size = 8, color_func = SimpleGroupedColorFunc(TopicWordsColor(currentTopic, prevTopic, nextTopic).getColorToWordsDict(), 'blue'))
            wordcloud.generate_from_frequencies(frequencies=frequencies)

            plt.figure(figsize=(10, 10), facecolor=None)
            plt.imshow(wordcloud)
            plt.axis('off')
            plt.title(f"{self.start + j*3}-{self.start + j*3 + 2} topic {self.path[j]}", fontsize=30, pad=30)
            plt.savefig(f"../../Resources/WordClouds/topic_{self.path[j]}_{self.start + j*3}_{self.start + j*3 + 2}.png")
            plt.close()

            if j % 12 == 0:
                pdf.add_page()
            if j == 0:
                pdf.cell(0, txt=f"{self.start + j*3} - {self.start + (len(self.path)-1)*3 + 2}", align='C');

            pdf.image(f"../../Resources/WordClouds/topic_{self.path[j]}_{self.start + j*3}_{self.start + j*3 + 2}.png",
                x=70 * (j % 3), y=(math.floor((j % 12) / 3) * 70) + 15, w=70)
            os.remove(f"../../Resources/WordClouds/topic_{self.path[j]}_{self.start + j*3}_{self.start + j*3 + 2}.png")

    def createWordCloudFocusVisualization(self, pdf, minLength):
        if len(self.path) < minLength:
            return

        topicPathBagOfWords = self._getTopicPathBagOfWords()
        topicPathBagOfWordsKeys = list(topicPathBagOfWords.keys())

        wordsMatrixPresence = np.zeros((len(topicPathBagOfWords), len(self.path)), dtype=np.single)
        wordsMatrixProb = np.zeros((len(topicPathBagOfWords), len(self.path)), dtype=np.single)
        for j in range(len(self.path)):
            for word in self._getTopic(j):
                wordsMatrixPresence[topicPathBagOfWords[word['word']]][j] = 1
                wordsMatrixProb[topicPathBagOfWords[word['word']]][j] = word['prob']

        wordsAppearPercentage = wordsMatrixPresence.sum(axis=1)*100/len(self.path)
        wordsMeanProb = wordsMatrixProb.sum(axis=1)/len(self.path)

        coreWords = {topicPathBagOfWordsKeys[j] : wordsMeanProb[j] for j in range(len(topicPathBagOfWords)) if wordsAppearPercentage[j] >= 70}
        notCoreWords = {topicPathBagOfWordsKeys[j] : wordsMeanProb[j] for j in range(len(topicPathBagOfWords)) if wordsAppearPercentage[j] <= 30}

        yearNotCoreWordsDict = {}
        for j in range(len(self.path)):
            if f"{self.start + j*3}" not in yearNotCoreWordsDict:
                yearNotCoreWordsDict[f"{self.start + j*3}"] = {}

            for key in notCoreWords.keys():
                if wordsMatrixPresence[topicPathBagOfWords[key]][j] == 1:
                    yearNotCoreWordsDict[f"{self.start + j * 3}"][key] = notCoreWords[key]

        rand = np.random.rand()
        wordcloud = WordCloud(width=800, height=800, background_color ='white', min_font_size = 7)
        wordcloud.generate_from_frequencies(frequencies=coreWords)

        plt.figure(figsize=(10, 10), facecolor=None)
        plt.imshow(wordcloud)
        plt.axis('off')
        plt.title(f"Core words", fontsize=30, pad=30)
        plt.savefig(f"../../Resources/WordClouds/core_words_{rand}.png")
        plt.close()

        pdf.add_page()
        pdf.cell(0, txt=f"{self.start} - {self.start + (len(self.path) - 1) * 3 + 2}", align='C');
        pdf.image(f"../../Resources/WordClouds/core_words_{rand}.png", x=70, y=15, w=70)
        os.remove(f"../../Resources/WordClouds/core_words_{rand}.png")

        # pdf.set_font("Times", size=12)
        # pdf.cell(0, ln=1, txt=f"Evolution Words", align='C');

        for j in range(len(self.path)):
            if len(yearNotCoreWordsDict[f"{self.start + j*3}"]) == 0:
                continue

            rand = np.random.rand()
            wordcloud = WordCloud(width=800, height=800, background_color ='white', min_font_size = 8)
            wordcloud.generate_from_frequencies(frequencies=yearNotCoreWordsDict[f"{self.start + j*3}"])

            plt.figure(figsize=(10, 10), facecolor=None)
            plt.imshow(wordcloud)
            plt.axis('off')
            plt.title(f"{self.start + j*3}-{self.start + j*3 + 2} topic {self.path[j]}", fontsize=30, pad=30)
            plt.savefig(f"../../Resources/WordClouds/topic_{self.path[j]}_{self.start + j*3}_{self.start + j*3 + 2}_{rand}.png")
            plt.close()

            pdf.image(f"../../Resources/WordClouds/topic_{self.path[j]}_{self.start + j*3}_{self.start + j*3 + 2}_{rand}.png",
                x=42 * (j % 5), y=(math.floor(j / 5) * 50) + 90, w=42)
            os.remove(f"../../Resources/WordClouds/topic_{self.path[j]}_{self.start + j*3}_{self.start + j*3 + 2}_{rand}.png")

    def createWordLineVisualization(self, pdf, minLength, order="alphabetical"):
        if len(self.path) < minLength:
            return

        topicPathBagOfWords = self._getTopicPathBagOfWords(order)

        wordsMatrix = np.zeros((len(topicPathBagOfWords), len(self.path)), dtype=np.single)
        for j in range(len(self.path)):
            for word in self._getTopic(j):
                wordsMatrix[topicPathBagOfWords[word['word']]][j] = word['prob']

        years = [self.start + j*3 for j in range(len(self.path))]
        plt.rc('axes', titlesize=15)
        plt.rc('xtick', labelsize=12)
        for j in range(len(wordsMatrix)):
            word = list(topicPathBagOfWords.keys())[list(topicPathBagOfWords.values()).index(j)]
            rand = np.random.rand()

            plt.figure(figsize=(12, 2))
            plt.scatter(years, wordsMatrix[j], color=['g' if value != 0 else 'r' for value in wordsMatrix[j]], marker="o")
            plt.plot(years, wordsMatrix[j], linestyle=":")
            plt.title(word)
            plt.ylim(0, 0.163)
            plt.yticks([])
            plt.xticks(ticks=years, labels=[f"{str(year)[-2:]}-{str(year+2)[-2:]}" for year in years])
            plt.savefig(f"../../Resources/TopicPath/word_{word}_{rand}.png")
            plt.close()

            if j % 26 == 0:
                pdf.add_page()
            if j == 0:
                pdf.cell(0, txt=f"{self.start} - {self.start + (len(self.path)-1)*3 + 2}", align='C');

            pdf.image(f"../../Resources/TopicPath/word_{word}_{rand}.png", x=105 * (j % 2), y=(math.floor((j % 26) / 2) * 20) + 15, w=105)
            os.remove(f"../../Resources/TopicPath/word_{word}_{rand}.png")

    def createParallelCoordinatesVisualization(self, minLength, order="alphabetical"):
        if len(self.path) < minLength:
            return

        topicPathBagOfWords = self._getTopicPathBagOfWords(order)

        wordsMatrix = np.zeros((len(topicPathBagOfWords), len(self.path)), dtype=np.single)
        for j in range(len(self.path)):
            for word in self._getTopic(j):
                wordsMatrix[topicPathBagOfWords[word['word']]][j] = word['prob']

        years = [self.start + j*3 for j in range(len(self.path))]
        goDictList = [dict(range=[1, len(topicPathBagOfWords)],
                         tickvals=[i+1 for i in range(len(topicPathBagOfWords))],
                         ticktext=list(topicPathBagOfWords.keys()),
                         constraintrange=[1, 2],
                         label='Words', values = [i+1 for i in range(len(topicPathBagOfWords))])]
        for j in range(len(self.path)):
            goDictList.append(dict(range = [0, 0.17],
                                   tickvals = [i for i in np.arange(0, 0.18, 0.01)],
                                   label = f"{str(self.start + j*3)[-2:]}-{str(self.start + j*3 +2)[-2:]}",
                                   values = wordsMatrix[:,j]))
        fig = go.Figure(data=
            go.Parcoords(
                line_color='blue',
                dimensions=goDictList
            )
        )
        fig.update_layout(
            autosize=False,
            width=1350,
            height=13*len(topicPathBagOfWords)
        )
        fig.show()

class SimpleGroupedColorFunc(object):
    def __init__(self, color_to_words, default_color):
        self.word_to_color = {word: color for (color, words) in color_to_words.items() for word in words}
        self.default_color = default_color

    def __call__(self, word, **kwargs):
        return self.word_to_color.get(word, self.default_color)


def getTopicsBagOfWords():
    topicsBagOfWords = []
    for temporalSlice in db.query("TemporalSlice").sort('start', 1):
        for i in range(0, len(temporalSlice['topics'])):
            topicsBagOfWords += [word['word'] for word in temporalSlice['topics'][i]]

    return {item: idx for idx, item in enumerate(set(topicsBagOfWords))}

def findTopicPath(start, current, path):
    compatibleLinks = 0
    if current < 2018:
        for link in topicLinksDict[current]:
            if link[0] == path[-1]:
                if current not in topicLinkVisitedDict:
                    topicLinkVisitedDict[current] = []

                compatibleLinks += 1
                topicLinkVisitedDict[current].append(link)

                newPath = [x for x in path]
                newPath.append(link[1])
                findTopicPath(start, current+3, newPath)

    if compatibleLinks == 0:
        topicPathList.append(TopicPath(start, path))

def createVisualization(visualization, minPathLength):
    if visualization in ['WordCloud', 'WordCloudFocus', 'WordLine']:
        pdf = FPDF()
        pdf.set_font('Times', size=14)

    for i in range(len(topicPathList)):
        if visualization == 'WordCloud':
            topicPathList[i].createWordCloudVisualization(pdf, minPathLength)
        elif visualization == 'WordCloudFocus':
            if i == 7: pdf.add_page()
            topicPathList[i].createWordCloudFocusVisualization(pdf, minPathLength)
        elif visualization == 'Table':
            if i == 37:
                topicPathList[i].createTableVisualization(minPathLength, "alphabetical")
        elif visualization == 'WordLine':
            topicPathList[i].createWordLineVisualization(pdf, minPathLength)
        elif visualization == 'ParallelCoordinates':
            topicPathList[i].createParallelCoordinatesVisualization(minPathLength)

    if visualization in ['WordCloud', 'WordCloudFocus', 'WordLine']:
        pdf.output(f"../../Resources/TopicPath/topics_path_{visualization}.pdf", "F")

qualitativeAnalysis = True
visualization = True
graph = False
ranking = False

with MongoDB("TesiMagistrale") as db:
    topicsBagOfWords = getTopicsBagOfWords()  
    temporalSlices = [temporalSlice for temporalSlice in db.query("TemporalSlice").sort('start', 1)]

    # Genara tutti i link tra coppie di slice temporali e la similarità tra le coppie
    topicLinksDict = {}
    temporalSliceLinksSimilaritySumDict = {}
    for i in range(len(temporalSlices) - 1):
        temporalSliceLinks = TemporalSliceLinks(temporalSlices[i], temporalSlices[i+1])
        topicLinksDict[temporalSlices[i]['start']] = temporalSliceLinks.getLinks(topicsBagOfWords, temporalSliceLinks.getBestThreshold(topicsBagOfWords))
        temporalSliceLinksSimilaritySumDict[temporalSlices[i]['start']] = temporalSliceLinks.getLinksSimilaritySum(topicsBagOfWords, 0.7)

    if not qualitativeAnalysis:
        # Stampa la similarità tra le varie coppie di slice temporali in un grafico
        plt.figure(figsize=(12, 9))
        plt.rc('xtick', labelsize=16)
        plt.rc('ytick', labelsize=16)
        axes = plt.gca()
        axes.set_ylim([0, 14])
        plt.plot([f"{str(year)[-2:]}-{str(year+3)[-2:]}" for year in range(1979, 2018, 3)], list(temporalSliceLinksSimilaritySumDict.values()))
        plt.show()
    else:
        # Genera i topic path a partire dai link
        topicLinkVisitedDict = {}
        topicPathList = []
        for year, links in topicLinksDict.items():
            for link in links:
                if year not in topicLinkVisitedDict:
                    topicLinkVisitedDict[year] = []
                if link not in topicLinkVisitedDict[year]:
                    topicLinkVisitedDict[year].append(link)
                    findTopicPath(year, year+3, [link[0], link[1]])

        # Fa il ranking dei topic path
        if ranking:
            topicPathSimilarityList = []
            for path in topicPathList:
                topicPathSimilarityList.append(path.getSynsetTopicSimilarity(3))
                print(f"{path.start}-{path.start + len(path.path)*3 -1} -> {path.path} -> {path.getSynsetTopicSimilarity(3)}")

        # Crea il grafo dei topic path
        if graph:
            graph = Digraph()
            existingNodeDict = {}
            for path in topicPathList:
                path.getGraph(2)

            graph.render('../../Resources/TopicPath/topicPathGraphAll.dot')

        # Crea le visualizzazioni dei topic path
        if visualization:
            createVisualization('Table', 3)