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
                    nodeLabel = f"{self.temporalSlice1['start']}-{self.temporalSlice1['end']} topic {i+1}"
                    if nodeLabel not in existingNodeDict:
                        existingNodeDict[nodeLabel] = []
                        graph.node(nodeLabel, nodeLabel)
                    nodeLabelEdge = f"{self.temporalSlice2['start']}-{self.temporalSlice2['end']} topic {j+1}"
                    if nodeLabelEdge not in existingNodeDict[nodeLabel]:
                        existingNodeDict[nodeLabel].append(nodeLabelEdge)
                        graph.edge(nodeLabel, nodeLabelEdge)

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
    def __init__(self, path):
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
        return list(db.query("TemporalSlice", query={'start': self.path[idx][0]}))[0]['topics'][self.path[idx][1] - 1]

    def _getBestSimilarity(self, word1, word2):
        bestSimilarity = 0
        for synset1 in wn.synsets(word1):
            for synset2 in wn.synsets(word2):
                similarity = synset1.wup_similarity(synset2)

                if similarity != None and similarity > bestSimilarity:
                    bestSimilarity = similarity

        return bestSimilarity

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
            header=dict(values=["Words"] + [f"{self.path[j][0]}-{self.path[j][0]+2} topic {self.path[j][1]}" for j in range(len(self.path))],
                        fill_color='paleturquoise', align='center'),
            cells=dict(values=[list(topicPathBagOfWords.keys())] + [['' for value in sparseVector] for sparseVector in topicsSparseVector],
                       fill_color=[['lavender'] * len(list(topicPathBagOfWords.keys()))] + topicsColorVector, align='center'))
        ])
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
            plt.title(f"{self.path[j][0]}-{self.path[j][0] + 2} topic {self.path[j][1]}", fontsize=30, pad=30)
            plt.savefig(f"../../Resources/WordClouds/topic_{self.path[j][1]}_{self.path[j][0]}_{self.path[j][0] + 2}.png")
            plt.close()

            if j % 12 == 0:
                pdf.add_page()
            if j == 0:
                pdf.cell(0, txt=f"{self.path[j][0]} - {self.path[len(self.path)-1][0] + 2}", align='C');

            pdf.image(f"../../Resources/WordClouds/topic_{self.path[j][1]}_{self.path[j][0]}_{self.path[j][0] + 2}.png",
                x=70 * (j % 3), y=(math.floor((j % 12) / 3) * 70) + 15, w=70)
            os.remove(f"../../Resources/WordClouds/topic_{self.path[j][1]}_{self.path[j][0]}_{self.path[j][0] + 2}.png")

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
            if f"{self.path[j][0]}" not in yearNotCoreWordsDict:
                yearNotCoreWordsDict[f"{self.path[j][0]}"] = {}

            for key in notCoreWords.keys():
                if wordsMatrixPresence[topicPathBagOfWords[key]][j] == 1:
                    yearNotCoreWordsDict[f"{self.path[j][0]}"][key] = notCoreWords[key]

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
        pdf.cell(0, txt=f"{self.path[0][0]} - {self.path[len(self.path) - 1][0] + 2}", align='C');
        pdf.image(f"../../Resources/WordClouds/core_words_{rand}.png", x=70, y=15, w=70)
        os.remove(f"../../Resources/WordClouds/core_words_{rand}.png")

        # pdf.set_font("Times", size=12)
        # pdf.cell(0, ln=1, txt=f"Evolution Words", align='C');

        for j in range(len(self.path)):
            if len(yearNotCoreWordsDict[f"{self.path[j][0]}"]) == 0:
                continue

            rand = np.random.rand()
            wordcloud = WordCloud(width=800, height=800, background_color ='white', min_font_size = 8)
            wordcloud.generate_from_frequencies(frequencies=yearNotCoreWordsDict[f"{self.path[j][0]}"])

            plt.figure(figsize=(10, 10), facecolor=None)
            plt.imshow(wordcloud)
            plt.axis('off')
            plt.title(f"{self.path[j][0]}-{self.path[j][0] + 2} topic {self.path[j][1]}", fontsize=30, pad=30)
            plt.savefig(f"../../Resources/WordClouds/topic_{self.path[j][1]}_{self.path[j][0]}_{self.path[j][0] + 2}_{rand}.png")
            plt.close()

            pdf.image(f"../../Resources/WordClouds/topic_{self.path[j][1]}_{self.path[j][0]}_{self.path[j][0] + 2}_{rand}.png",
                x=42 * (j % 5), y=(math.floor(j / 5) * 50) + 90, w=42)
            os.remove(f"../../Resources/WordClouds/topic_{self.path[j][1]}_{self.path[j][0]}_{self.path[j][0] + 2}_{rand}.png")

    def createWordLineVisualization(self, pdf, minLength, order="alphabetical"):
        if len(self.path) < minLength:
            return

        topicPathBagOfWords = self._getTopicPathBagOfWords(order)

        wordsMatrix = np.zeros((len(topicPathBagOfWords), len(self.path)), dtype=np.single)
        for j in range(len(self.path)):
            for word in self._getTopic(j):
                wordsMatrix[topicPathBagOfWords[word['word']]][j] = word['prob']

        years = [self.path[j][0] for j in range(len(self.path))]
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
                pdf.cell(0, txt=f"{self.path[0][0]} - {self.path[len(self.path)-1][0] + 2}", align='C');

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

        years = [self.path[j][0] for j in range(len(self.path))]
        goDictList = [dict(range=[1, len(topicPathBagOfWords)],
                         tickvals=[i+1 for i in range(len(topicPathBagOfWords))],
                         ticktext=list(topicPathBagOfWords.keys()),
                         constraintrange=[1, 2],
                         label='Words', values = [i+1 for i in range(len(topicPathBagOfWords))])]
        for j in range(len(self.path)):
            goDictList.append(dict(range = [0, 0.17],
                                   tickvals = [i for i in np.arange(0, 0.18, 0.01)],
                                   label = f"{str(self.path[j][0])[-2:]}-{str(self.path[j][0] +2)[-2:]}",
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

def createVisualization(visualization, minPathLength):
    if visualization in ['WordCloud', 'WordCloudFocus', 'WordLine']:
        pdf = FPDF()
        pdf.set_font('Times', size=14)

    for i in range(len(topicPathList)):
        if visualization == 'WordCloud':
            topicPathList[i].createWordCloudVisualization(pdf, minPathLength)
        elif visualization == 'WordCloudFocus':
            if i == 1:
                pdf.add_page()
            topicPathList[i].createWordCloudFocusVisualization(pdf, minPathLength)
        elif visualization == 'Table':
            topicPathList[i].createTableVisualization(minPathLength, "alphabetical")
        elif visualization == 'WordLine':
            topicPathList[i].createWordLineVisualization(pdf, minPathLength)
        elif visualization == 'ParallelCoordinates':
            topicPathList[i].createParallelCoordinatesVisualization(minPathLength)

    if visualization in ['WordCloud', 'WordCloudFocus', 'WordLine']:
        pdf.output(f"../../Resources/TopicPath/topics_path_{visualization}_with_jumps.pdf", "F")


topicPathList = [
    TopicPath([(1979, 4), (1982, 1), (1985, 4), (1991, 14), (1994, 16), (1997, 13), (2000, 15), (2003, 9), (2006, 18), (2009, 17), (2012, 18), (2015, 7), (2018, 17)]),
    TopicPath([(1979, 4), (1997, 5), (2000, 19), (2012, 19), (2015, 7), (2018, 17)]),
    TopicPath([(1979, 1), (1982, 14), (1985, 3), (1988, 12), (1991, 17)]),
    TopicPath([(1979, 8), (1982, 12), (1985, 20), (1988, 8), (1994, 19), (1997, 4), (2003, 5), (2006, 14), (2009, 20), (2012, 2), (2015, 8)]),
    TopicPath([(1979, 14), (1985, 9), (1988, 7), (1991, 2), (1994, 9), (1997, 17), (2000, 7), (2003, 1), (2006, 19), (2009, 9), (2012, 4), (2018, 9)]),
    TopicPath([(1979, 18), (1982, 6), (1985, 18), (1988, 15), (1991, 16), (1994, 3), (1997, 19), (2000, 4), (2009, 1), (2012, 5), (2015, 19), (2018, 7)]),
    TopicPath([(1982, 2), (1985, 17), (1988, 9), (1991, 15), (1994, 5), (1997, 12), (2000, 17), (2006, 4), (2009, 5), (2015, 16), (2018, 5)]),
    TopicPath([(1982, 2), (1985, 17), (1988, 9), (1991, 15), (1994, 5), (1997, 12), (2006, 10), (2009, 4), (2012, 20), (2015, 16), (2018, 5)]),
    TopicPath([(1982, 7), (1988, 19), (1991, 5), (1994, 20), (1997, 10), (2000, 1), (2003, 10), (2009, 6), (2015, 1)]),
    TopicPath([(1985, 18), (1988, 5), (1994, 15), (1997, 7), (2000, 2), (2003, 20), (2006, 13), (2009, 8)]),
    TopicPath([(1985, 5), (1988, 18), (1994, 12), (1997, 15), (2000, 3), (2003, 4), (2006, 7)]),
    TopicPath([(1988, 4), (1991, 7), (1994, 7), (1997, 20), (2000, 9), (2003, 19), (2015, 13), (2018, 1)]),
    TopicPath([(1988, 10), (1991, 13), (1994, 2), (1997, 9), (2000, 13), (2003, 16), (2006, 5), (2009, 14), (2012, 1)]),
    TopicPath([(1988, 20), (2000, 11), (2003, 7), (2006, 12), (2009, 19), (2015, 17), (2018, 20)]),
    TopicPath([(1991, 4), (1994, 18), (1997, 16)]),
    TopicPath([(1994, 4), (1997, 18), (2000, 16), (2006, 3)]),
    TopicPath([(1994, 4), (1997, 18), (2003, 6), (2006, 8), (2009, 3), (2015, 15), (2018, 11)]),
    TopicPath([(1994, 14), (1997, 11), (2000, 14), (2003, 8), (2006, 11), (2009, 11), (2012, 12)]),
    TopicPath([(1997, 2), (2000, 8), (2003, 12), (2006, 15), (2009, 10), (2012, 10), (2015, 14), (2018, 12)]),
    TopicPath([(2009, 13), (2015, 11), (2018, 10)]),
    TopicPath([(2009, 16), (2012, 14)]),
    TopicPath([(2015, 4), (2018, 4)])
]

visualization = True

with MongoDB("TesiMagistrale") as db:
    if not visualization:
        topicsBagOfWords = getTopicsBagOfWords()
        temporalSlices = [temporalSlice for temporalSlice in db.query("TemporalSlice").sort('start', 1)]

        # Crea il grafo dei topic path
        graph = Digraph()
        existingNodeDict = {}

        # Genara tutti i link tra coppie di slice temporali e la similarità tra le coppie
        for i in range(len(temporalSlices) - 1):
            for j in range(i+1, len(temporalSlices)):
                TemporalSliceLinks(temporalSlices[i], temporalSlices[j]).getLinks(topicsBagOfWords, 0.45)

        graph.render('../../Resources/TopicPath/topicPathGraphAllPair0_45.dot')
    else:
        createVisualization('Table', 3)
