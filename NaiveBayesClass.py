import random
import os
import math

#
# Count the lines of a text file
def countLines(filePath):
  count = 0
  with open(filePath, 'r', encoding='utf8') as f:
    for count, line in enumerate(f):
        pass
  return count + 1

#
# Sample randomly a given number of lines (numLines) from a text file
def sampleLines(filePath, fromLine, toLine, numLines):
  res = []
  rndLines = random.sample(range(fromLine, toLine), numLines)
  rndLines.sort()
  
  rndIdx = 0
  count = 0
  with open(filePath, 'r', encoding='utf8') as f:
    for line in f:
      if rndIdx >= len(rndLines):
        break
      
      line = line.replace("\n", '')
      if line == '':
        continue
      count = count + 1
      
      if count == rndLines[rndIdx]:
        res.append(line)
        rndIdx = rndIdx + 1

  return res

#
# Sample randomly a given number of lines (samplingTotalCount) from the files present in the given directory
def sampleFiles0(dirPath, fileCount, samplingTotalCount):
  SKIP_LINES = 100
  
  sampleCount = math.floor(samplingTotalCount / fileCount) 
  sampleLastCount = samplingTotalCount - sampleCount * (fileCount - 1)

  res = []
  idx = 1
  for path in os.listdir(dirPath):
    filePath = os.path.join(dirPath, path)
    if os.path.isfile(filePath):
      numLines = countLines(filePath)
      if idx == fileCount:
        res.extend(sampleLines(filePath, SKIP_LINES + 1, numLines - SKIP_LINES, sampleLastCount))
      else:
        res.extend(sampleLines(filePath, SKIP_LINES + 1, numLines - SKIP_LINES, sampleCount))
      idx = idx + 1

  return res

#
# Sample randomly a training and test set from the files present in the given directory
def sampleFiles(dirPath, trainingCount, testCount):
  fileCount = 0
  for path in os.listdir(dirPath):
    filePath = os.path.join(dirPath, path)
    if os.path.isfile(filePath):
      fileCount = fileCount + 1
      
  training = sampleFiles0(dirPath, fileCount, trainingCount)
  test = sampleFiles0(dirPath, fileCount, testCount)
  return (training, test)

#
# Clean a word by removing unnecessary characters
def getCleanWord(word):
  res = word.lower()
  res = res.replace('"', '')
  res = res.replace('“', '')
  res = res.replace('”', '')
  res = res.replace('!', '')
  res = res.replace('?', '')
  res = res.replace('-', '')
  res = res.replace('_', '')
  res = res.replace('.', '')
  res = res.replace(',', '')
  res = res.replace(';', '')
  res = res.replace(':', '')
  res = res.replace("'s", '')
  #res = res.replace('a', '')
  #res = res.replace('an', '')
  #res = res.replace('in', '')
  #res = res.replace('as', '')
  #res = res.replace('to', '')
  #res = res.replace('by', '')
  #res = res.replace('of', '')
  return res

#
# Turn a list of sentences in a list of words
def getWords(arr):
  words = [getCleanWord(word) for line in arr for word in line.split()]
  words = [word for word in words if word] # remove empty strings
  return words

#
# Return the # of all words and the # of all unique words of two training set 
def getNumWords(training1, training2):
  trainingWords = getWords(training1)
  trainingWords.extend(getWords(training2))
  
  # get unique
  list_set = set(trainingWords)
  unique_list = (list(list_set))
  return (len(trainingWords), len(unique_list))

#
# Get the probability of testStr of being part of the trainingArr by Bayes' Theorem
def getNaiveBayesClassProbability(trainingArr, numAllWords, numUniqueWords, testStr):
  trainingWords = getWords(trainingArr)
  numTrainingWords = len(trainingWords) # [len(sentence.split()) for sentence in trainingArr]
  probRes = 1
  for word in testStr.split(' '):
    word = getCleanWord(word)
    if word == '':
      continue
    
    numWordInTraining = trainingWords.count(word)
    prob = (numWordInTraining + 1) / (numTrainingWords + numUniqueWords)
    
    # multiply by P(C_1) or P(C_2), but since they are both 1, we can skip it
    # prob = prob * numTrainingWords / numAllWords
    
    # product of the probabilities of each word
    probRes = probRes * prob
      
  return probRes

#
# MAIN
#

TRAINIG_SAMPLE_LINES = 1000
TEST_SAMPLE_LINES = 10

# --- prepare training data

(trainingJA, testJA) = sampleFiles('./data/jane_austen/', TRAINIG_SAMPLE_LINES, TEST_SAMPLE_LINES)
(trainingCD, testCD) = sampleFiles('./data/charles_dickens/', TRAINIG_SAMPLE_LINES, TEST_SAMPLE_LINES)
(numAllWords, numUniqueWords) = getNumWords(trainingJA, trainingCD)

# --- test the prediction

countAllGood = 0
tp = 0
tn = 0
for testStr in testJA:
  probJA = getNaiveBayesClassProbability(trainingJA, numAllWords, numUniqueWords, testStr)
  probCD = getNaiveBayesClassProbability(trainingCD, numAllWords, numUniqueWords, testStr)
  res = probJA > probCD
  print(str(res) + ' - ' + testStr + ': ' + str(probJA) + ' - ' + str(probCD))
  if res:
    countAllGood = countAllGood + 1
    tp = tp + 1

for testStr in testCD:
  probJA = getNaiveBayesClassProbability(trainingJA, numAllWords, numUniqueWords, testStr)
  probCD = getNaiveBayesClassProbability(trainingCD, numAllWords, numUniqueWords, testStr)
  res = probCD > probJA 
  print(str(res) + ' - ' + testStr + ': ' + str(probJA) + ' - ' + str(probCD))
  if res:
    countAllGood = countAllGood + 1
    tn = tn + 1

# --- show the final results

print('')
print(str(countAllGood) + ' / ' + str(TEST_SAMPLE_LINES * 2))
print('Sensitivity: ' + str(round(tp / len(testJA), 2)))
print('Specificity: ' + str(round(tn / len(testCD), 2)))
print('Accuracy: ' + str(round((tp + tn) / (len(testJA) + len(testCD)), 2)))

