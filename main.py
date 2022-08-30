class EvalFeatures:
    def getJSDivergence(self, distA, distB):
        probA, probB, jsd = 0, 0, 0
        complete = []
        complete.extend(distA.vocabWords)
        for c in range(len(distB.vocabWords)):
            if (not distB.vocabWords[c] in complete):
                complete.append(distB.vocabWords[c])
        for i in range(len(complete)):
            try:
                indA = distA.vocabWords.index(complete[i])
                probA = distA.vocabFreq[indA] / (distA.numTokens)
            except ValueError:
                probA = 0
            try:
                indB = distB.vocabWords.index(complete[i])
                probB = distA.vocabFreq[indB] / (distB.numTokens)
            except ValueError:
                probB = 0
            if (probA == 0):
                part1 = 0
            else:
                part1 = probA * math.log(probA / ((probA / 2) + (probB / 2)))
            if (probB == 0):
                part2 = 0
            else:
                part2 = probB * math.log(probB / ((probB / 2) + (probA / 2)))
            jsd += (part1 + part2) / 2
        return jsd

    def getSmoothedJSDivergence(self, distA, distB, gamma, bins):
        countA, probA, countB, probB, jsd = 0.0, 0, 0, 0, 0
        complete = []
        complete.extend(distA.vocabWords)
        for c in range(len(distB.vocabWords)):
            if (not distB.vocabWords[c] in complete):
                complete.append(distB.vocabWords[c])
        for i in range(len(complete)):
            try:
                indA = distA.vocabWords.index(complete[i])
                countA = distA.vocabFreq[indA] + gamma
            except ValueError:
                countA = gamma
            probA = countA / ((distA.numTokens) + (gamma * bins))
            try:
                indB = distB.vocabWords.index(complete[i])
                countB = distB.vocabFreq[indB] + gamma
            except ValueError:
                countB = gamma
            probB = countB / ((distB.numTokens) + (gamma * bins))
            if (probA == 0):
                part1 = 0
            else:
                part1 = probA * math.log(probA / ((probA / 2) + (probB / 2)))
            if (probB == 0):
                part2 = 0
            else:
                part2 = probB * math.log(probB / ((probB / 2) + (probA / 2)))
            jsd += (part1 + part2) / 2
        return jsd

    def getKLdivergenceSmoothed(self, distA, distB, gamma, bins):
        divDistADistB = 0
        divDistBDistA = 0
        complete = []
        complete.extend(distA.vocabWords)
        for c in range(len(distB.vocabWords)):
            if (not distB.vocabWords[c] in complete):
                complete.append(distB.vocabWords[c])
        for i in range(len(complete)):
            try:
                indA = distA.vocabWords.index(complete[i])
                countA = distA.vocabFreq[indA] + gamma
            except ValueError:
                countA = gamma
            try:
                indB = distB.vocabWords.index(complete[i])
                countB = distB.vocabFreq[indB] + gamma
            except ValueError:
                countB = gamma
            probA = countA / ((distA.numTokens) + (gamma * bins))
            probB = countB / ((distB.numTokens) + (gamma * bins))
            probA_B = probA / probB
            probB_A = probB / probA
            divDistADistB += probA * math.log(probA_B)
            divDistBDistA += probB * math.log(probB_A)
        if ((divDistADistB < 0) or (divDistBDistA < 0)):
            print(" negative div = ", divDistADistB, ",", divDistBDistA)
        KLdiv = [0, 0]
        KLdiv[0] = divDistADistB
        KLdiv[1] = divDistBDistA
        return KLdiv

    def getPercentTokensThatIsSignTerms(self, topicWordList, dist):
        count = 0
        for i in range(len(topicWordList)):
            signTerm = topicWordList[i]
            try:
                present = dist.vocabWords.index(signTerm)
                count += dist.vocabFreq[present]
            except:
                continue
        if (count == 0):
            return 0
        percentTokens = count / dist.numTokens
        return (percentTokens)

    def getPercentTopicWordsCoveredByGivenDist(self, topicWordList, dist):
        count = 0
        for i in range(len(topicWordList)):
            signTerm = topicWordList[i]
            try:
                present = dist.vocabWords.index(signTerm)
                count = count + 1
            except:
                continue
        if (count == 0):
            return 0
        percentCovered = count / len(topicWordList)
        return percentCovered

    def getUnigramProbability(self, emissionDist, newDist):
        probOfNewDist = 0.0
        for i in range(len(newDist.vocabWords)):
            word = newDist.vocabWords[i]
            try:
                emitIndex = emissionDist.vocabWords.index(word)
                wordFreqInEmissionDist = emissionDist.vocabFreq[emitIndex]
            except:
                continue
            wordEmissionProbability = wordFreqInEmissionDist / emissionDist.numTokens
            wordFreqInNewDist = newDist.vocabFreq[i]
            probOfNewDist += wordFreqInNewDist * math.log10(wordEmissionProbability)
        return probOfNewDist

    def getMultinomialProbability(self, emissionDist, newDist):
        unigramProb = self.getUnigramProbability(emissionDist, newDist)
        denomMultCoeff = 0
        numMultCoeff = 0
        for i in range(len(newDist.vocabWords)):
            wordFreqInNewDist = newDist.vocabFreq[i]
            fact = 1
            for j in range(1, wordFreqInNewDist):
                fact *= j
            denomMultCoeff += math.log10(fact)
        for j in range(1, newDist.numTokens):
            numMultCoeff += math.log10(j)
        multinomialCoeff = numMultCoeff - denomMultCoeff
        return (multinomialCoeff + unigramProb)
class vocabDist:
    def __init__(self, words, freq, tokens):
        self.vocabWords = words
        self.vocabFreq = freq
        self.numTokens = tokens

    def printStats(self):
        print("vocabulary size = ", len(self.vocabWords))
        print("total tokens = " + self.numTokens)
class CorpusBasedUtilities:
    def readStopWords(self, filepath):
        if (not os.path.exists(filepath)):
            print("Stopword file not found: " + filepath)
        br = open(filepath, 'r')
        line = br.readline()
        while (line != ""):
            if (line.strip() == ""):
                continue
            self.stopWords.append(line.strip().lower())
            line = br.readline()
        br.close()

    def readBackgroundIdf(self, filepath):
        if (not os.path.exists(filepath)):
            print("Idf file not found: " + filepath)
        br = open(filepath, 'r')
        self.totalDocs = int(br.readline().strip())
        line = br.readline()
        while (line != ""):
            if (line.strip() == ""):
                continue
            toks = line.strip().split(" ")
            wd = toks[0]
            idf = float(toks[1])
            self.wordToIdf.update({wd: idf})
            line = br.readline()
        br.close()

    def readBackgroundCorpusCounts(self, filepath, stem):
        if (not os.path.exists(filepath)):
            print("Idf file not found: " + filepath)
        br = open(filepath, 'r')
        totalToks = 0
        words = []
        freqs = []
        line = br.readline()
        while (line != ""):
            if (line.strip() == ""):
                continue
            toks = line.strip().split(" ")
            wd = toks[0]
            if (stem):
                wd = PorterStemmer().stem(wd)
            count = int(toks[1])
            totalToks += count
            # //update frequency if word already in list else add as new
            try:
                indInVocab = words.index(wd)
                cur = freqs[indInVocab]
                freqs[indInVocab] = cur + count
            except:
                words.append(wd)
                freqs.append(count)
            line = br.readline()
        # //create aggregated vocabulary dist. Only one entry per stem with frequencies totalled over all words adding to the same stem.
        global backgroundDist
        backgroundDist = vocabDist(words, freqs, totalToks)
        br.close()

    def __init__(self, conf):
        self.stopWords = []  # //this list will be empty throughout if stopword=N option is specified
        self.wordToIdf = {}
        self.totalDocs = 0
        self.stemOption = conf.performStemming  # //if true, then all vocabulary distributions--input, summary, background and idf values file are stemmed.
        # //print("stemopt= "+stemOption)
        if (conf.removeStopWords):
            print("Reading stopwords from " + conf.stopFile)
            self.readStopWords(conf.stopFile)
        if (conf.topic):
            print("Reading background corpus frequency counts " + conf.bgCountFile)
            self.readBackgroundCorpusCounts(conf.bgCountFile, conf.performStemming)
        if (conf.cosine):
            if (conf.performStemming):
                print("Reading idf values" + conf.bgIdfStemmedFile)
                self.readBackgroundIdf(conf.bgIdfStemmedFile)
            else:
                print("Reading idf values" + conf.bgIdfUnstemmedFile)
                self.readBackgroundIdf(conf.bgIdfUnstemmedFile)

    def computeVocabulary(self, path):
        if (not os.path.exists(path)):
            print("Cannot compute vocabulary for : non-existent file path : " + path)
        counts = {}
        if (os.path.isdir(path)):
            files = [f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]
            for f in range(len(files)):
                with open(path + "\\" + files[f], 'r') as brf:
                    for fline in brf:
                        if (fline.strip() == ""):
                            continue
                        fline = re.sub("[^A-Za-z0-9 ]", " ", fline)
                        toks = fline.strip().split(" ")
                        for t in range(len(toks)):
                            self.updateCounts(toks[t].lower(), counts)
                brf.close()
        else:
            bx = open(path, 'r', encoding='utf8')
            ll = bx.readline()
            while (ll != ""):
                if (ll.strip() == ""):
                    continue
                ll = ll.replace("[^a-zA-Z0-9 ]", " ")
                tks = ll.strip().split("[ ]+")
                for tk in range(len(tks)):
                    self.updateCounts(tks[tk].lower(), counts)
                ll = bx.readline()
            bx.close()
        words = []
        freq = []
        totalToks = 0
        em = counts.keys()
        for m in em:
            wd = str(m)
            fr = counts[wd]
            words.append(wd)
            freq.append(fr)
            totalToks += fr
        return vocabDist(words, freq, totalToks)

    def updateCounts(self, word, currCounts):
        if (word in self.stopWords):
            return
        if (self.stemOption):
            word = PorterStemmer().stem(word)
        if (word in currCounts.keys()):
            cur = currCounts[word]
            currCounts.update({word: cur + 1})
        else:
            currCounts.update({word: 1})

    def computeCosine(self, vecA, vecB):
        dotProd = 0.0
        vecAMag = 0.0
        vecBMag = 0.0
        cosSim = 0.0
        for n in range(len(vecA)):
            dotProd += vecA[n] * vecB[n]
            vecAMag += vecA[n] * vecA[n]
            vecBMag += vecB[n] * vecB[n]
        if (dotProd == 0.0):
            cosSim = 0.0
        else:
            cosSim = dotProd / (math.sqrt(vecAMag) * math.sqrt(vecBMag))
        return cosSim

    def computeTfCosineGivenVocabDists(self, distA, distB):
        complete = []
        complete.append(distA.vocabWords)
        for p in range(len(distB.vocabWords)):
            if (not distB.vocabWords[p] in complete):
                complete.append(distB.vocabWords[p])
        vecA = [0] * len(complete)
        vecB = [0] * len(complete)
        for i in range(len(complete)):
            indA = distA.vocabWords.index(complete[i])
            if (indA != -1):
                vecA[i] = distA.vocabFreq[indA]
            else:
                vecA[i] = 0
            indB = distB.vocabWords.index(complete[i])
            if (indB != -1):
                vecB[i] = distB.vocabFreq[indB]
            else:
                vecB[i] = 0
        return self.computeCosine(vecA, vecB)

    def multiplyIdfAndGetCosine(self, stringsInVector, tfVecA, tfVecB):
        tfidf = [[0] * len(stringsInVector)] * 2
        for i in range(len(stringsInVector)):
            if (stringsInVector[i] in self.wordToIdf.keys()):
                idf = self.wordToIdf[stringsInVector[i]]
            else:
                idf = math.log(self.totalDocs)
            tfidf[0][i] = tfVecA[i] * idf
            tfidf[1][i] = tfVecB[i] * idf
        return self.computeCosine(tfidf[0], tfidf[1])

    def computeTfIdfCosineGivenVocabDists(self, distA, distB):
        complete = []
        complete.extend(distA.vocabWords)
        for p in range(len(distB.vocabWords)):
            if (not distB.vocabWords[p] in complete):
                complete.append(distB.vocabWords[p])
        vecA = [0] * len(complete)
        vecB = [0] * len(complete)
        for i in range(len(complete)):
            try:
                indA = distA.vocabWords.index(complete[i])
                vecA[i] = distA.vocabFreq[indA]
            except:
                vecA[i] = 0
            try:
                indB = distB.vocabWords.index(complete[i])
                vecB[i] = distB.vocabFreq[indB]
            except:
                vecB[i] = 0
        return self.multiplyIdfAndGetCosine(complete, vecA, vecB)

    def computeLogLikelihoodRatio(self, dist):
        chisqValues = []
        for i in range(len(dist.vocabWords)):
            wfreq = dist.vocabFreq[i]
            try:
                bgIndex = backgroundDist.vocabWords.index(dist.vocabWords[i])
                bgfreq = backgroundDist.vocabFreq[bgIndex]
            except:
                bgfreq = 0.0
            o11 = wfreq
            o12 = bgfreq
            o21 = dist.numTokens - wfreq
            o22 = backgroundDist.numTokens - bgfreq
            N = o11 + o12 + o21 + o22
            p = (o11 + o12) / N
            p1 = o11 / (o11 + o21)
            p2 = o12 / (o12 + o22)
            if (p == 0):
                t1 = 0.0
            else:
                t1 = math.log10(p) * (o11 + o12)
            if (p == 1):
                t2 = 0.0
            else:
                t2 = (o21 + o22) * math.log10(1 - p)
            if (p1 == 0):
                t3 = 0.0
            else:
                t3 = o11 * math.log10(p1)
            if (p1 == 1):
                t4 = 0.0
            else:
                t4 = o21 * math.log10(1 - p1)
            if (p2 == 0):
                t5 = 0.0
            else:
                t5 = o12 * math.log10(p2)
            if (p2 == 1):
                t6 = 0.0
            else:
                t6 = o22 * math.log10(1 - p2)
            loglik = -2.0 * ((t1 + t2) - (t3 + t4 + t5 + t6))
            chisqValues.append(loglik)
        return chisqValues

    def getTopicSignatures(self, dist, criticalValue):
        loglikRatios = self.computeLogLikelihoodRatio(dist)
        topicWords = []
        for x in range(len(dist.vocabWords)):
            if (loglikRatios[x] > criticalValue):
                topicWords.append(dist.vocabWords[x])
        return topicWords

    def clearAll(self):
        self.wordToIdf.clear()
        self.stopWords.clear()
class AverageScores:
    def getAverageValues(self, diffInpFeatures):
        df = "{0:.4f}".format
        avgValues = len(diffInpFeatures.getFirst())
        for inp in range(len(diffInpFeatures)):
            featValues = diffInpFeatures.get[inp]
            for feat in range(len(featValues)):
                avgValues[feat] += featValues[feat]
        numInputs = len(diffInpFeatures)
        ret = ""
        for a in range(len(avgValues)):
            ret += df(avgValues[a] / numInputs) + " "
        return ret.strip()
class ConfigOptions:
    def __init__(self):
        self.performStemming = True
        self.removeStopWords = True
        self.stopFile = ""
        self.bgCountFile = ""
        self.bgIdfUnstemmedFile = ""
        self.bgIdfStemmedFile = ""
        self.divergence = True
        self.cosine = True
        self.topic = True
        self.summProb = True
        self.topicCutoff = 10.0
class InputBasedEvaluation:
    def generateFeatures(self, inputDist, summaryDist, listOfFeatures, feat, cbu, topicCutoff):
        myFormatter = "{0:.4f}".format
        featString = ""
        if ("KLInputSummary" in listOfFeatures):
            klarray = feat.getKLdivergenceSmoothed(inputDist, summaryDist, 0.005, 1.5 * len(inputDist.vocabWords))
            featString += myFormatter(klarray[0]) + " " + myFormatter(klarray[1]) + " "
            featString += myFormatter(feat.getJSDivergence(inputDist, summaryDist)) + " "
            featString += myFormatter(
                feat.getSmoothedJSDivergence(inputDist, summaryDist, 0.005, 1.5 * len(inputDist.vocabWords))) + " "
        if ("cosineAllWords" in listOfFeatures):
            featString += myFormatter(cbu.computeTfIdfCosineGivenVocabDists(inputDist, summaryDist)) + " "
        if ("percentTopicTokens" in listOfFeatures):
            inputTopicWords = cbu.getTopicSignatures(inputDist, topicCutoff)
            percentTokensTopicWords = myFormatter(
                feat.getPercentTokensThatIsSignTerms(inputTopicWords, summaryDist))
            fractionTopicWordsCovered = myFormatter(
                feat.getPercentTopicWordsCoveredByGivenDist(inputTopicWords, summaryDist))
            featString += percentTokensTopicWords + " " + fractionTopicWordsCovered + " "
            topicWordFrequencies = []
            totalCount = 0
            for tp in range(len(inputTopicWords)):
                indtopic = inputDist.vocabWords.index(inputTopicWords[tp])
                freq = inputDist.vocabFreq[indtopic]
                topicWordFrequencies.append(freq)
                totalCount += freq
            inputTopicDist = vocabDist(inputTopicWords, topicWordFrequencies, totalCount)
            topicOverlap = myFormatter(cbu.computeTfIdfCosineGivenVocabDists(inputTopicDist, summaryDist))
            featString += topicOverlap + " "
        if ("unigramProb" in listOfFeatures):
            uniprob = feat.getUnigramProbability(inputDist, summaryDist)
            multprob = feat.getMultinomialProbability(inputDist, summaryDist)
            featString += myFormatter(uniprob) + " " + myFormatter(multprob) + " "
        return featString.strip()

    def readAndStoreConfigOptions(self):
        cf = ConfigOptions()
        cf.performStemming = True
        cf.removeStopWords = True
        cf.divergence = True
        cf.cosine = True
        cf.topic = True
        cf.summProb = True
        cf.stopFile = 'smart_common_words.txt'
        cf.bgCountFile = 'bgFreqCounts.unstemmed.txt'
        cf.bgIdfUnstemmedFile = 'bgIdfValues.unstemmed.txt'
        cf.bgIdfStemmedFile = 'bgIdfValues.stemmed.txt'
        cf.topicCutoff = 0.75
        return cf


import PyPDF2
import os
import re
import math
from nltk.stem import PorterStemmer
originalText = []
processedText = []
path = 'papers'
directory = 'output'
mylist = os.listdir(path)
for cl in mylist:
    pdfFileObj = open(f'{path}/{cl}', 'rb')
    pdfReader = PyPDF2.PdfFileReader(pdfFileObj)
    # print(pdfReader.numPages)
    txtFileObj= open(f'{directory}/{cl}.txt', 'w+', encoding='utf8')
    for i in range(pdfReader.numPages):
        pageObj = pdfReader.getPage(i)
        temp=pageObj.extractText()
        originalText.append(temp)
        txtFileObj.writelines(temp)
    pdfFileObj.close()
    txtFileObj.close()
textfile = open('output/1.pdf.txt', 'r',encoding='utf8')
filetext = textfile.read()
textfile.close()
x = re.search("1 I NTRODUCTION", filetext)
if(x!=None):
    print("The first white-space character is located in position:", x.start())
y= re.search("2 L ITERATURE REVIEW", filetext)
if(y!=None):
    print(filetext[x.start():y.start()])
    # ieval = InputBasedEvaluation()
    # opt = ieval.readAndStoreConfigOptions()
    # featuresToCompute = []
    # featuresToCompute.append("KLInputSummary")
    # featuresToCompute.append("KLSummaryInput")
    # featuresToCompute.append("unsmoothedJSD")
    # featuresToCompute.append("smoothedJSD")
    # featuresToCompute.append("cosineAllWords")
    # featuresToCompute.append("percentTopicTokens")
    # featuresToCompute.append("fractionTopicWords")
    # featuresToCompute.append("topicWordOverlap")
    # featuresToCompute.append("unigramProb")
    # featuresToCompute.append("multinomialProb")
    # cbu = CorpusBasedUtilities(opt)
    # feat = EvalFeatures()
    # for f in range(len(featuresToCompute)):
    #     print(featuresToCompute[f])
    #     inputVocabDist = cbu.computeVocabulary(f'{directory}/{cl}.txt')
    #     summaryVocabDist = cbu.computeVocabulary(f'{directory}/{cl}.txt')
    #     features = ieval.generateFeatures(inputVocabDist, summaryVocabDist, featuresToCompute, feat, cbu, opt.topicCutoff)
    # numFeatures = len(featuresToCompute)