from nltk.tokenize import word_tokenize
import nltk.classify.util as util
from nltk.classify import NaiveBayesClassifier
from nltk.metrics import BigramAssocMeasures
from nltk.collocations import BigramCollocationFinder as BCF
import itertools
import pickle

def features(words):
	words = word_tokenize(words)

	scoreF = BigramAssocMeasures.chi_sq

	#bigram count
	n = 150

	bigrams = BCF.from_words(words).nbest(scoreF, n)

	return dict([word,True] for word in itertools.chain(words, bigrams))

def training():
	neu_sen = open("neutral.txt", 'r', encoding='latin-1').read()
	pos_sen = open("positive.txt", 'r', encoding = 'latin-1').read()
	neg_sen = open("negative.txt", 'r', encoding = 'latin-1').read()

	emoji = open("emoji.txt",'r', encoding = 'latin-1').read()
	pos_emoji = []
	neg_emoji = []
	for i in emoji.split('\n'):
		exp = ''
		if i[len(i)-2] == '-':
			for j in range(len(i) - 2):
				exp += i[j]
			neg_emoji.append(( {exp : True}, 'negative'))
		else:
			for j in range(len(i)-1):
				exp += i[j]
			pos_emoji.append(( {exp : True}, 'positive'))

	ntrev = [(features(words), 'neutral') for words in neu_sen.split('\n')]
	prev = [(features(words), 'positive') for words in pos_sen.split('\n')]
	nrev = [(features(words), 'negative') for words in neg_sen.split('\n')]
	
	neu_set = ntrev
	pos_set = prev + pos_emoji
	neg_set = nrev + neg_emoji

	real_classifier = NaiveBayesClassifier.train(neu_set+pos_set+neg_set)

	# SAVE IN FILE TO AVOID TRAIINING THE DATA AGAIN
	save_doc = open("classifier.pickle", 'wb')
	pickle.dump(real_classifier, save_doc)
	save_doc.close()

	# TO TEST ACCURACY OF CLASSIFIER UNCCOMMENT THE CODE BELOW
	ACCURACY : 78.1695423855964

	ncutoff = int(len(nrev)*3/4)
	pcutoff = int(len(prev)*3/4)
	train_set = nrev[:ncutoff] + prev[:pcutoff] + pos_emoji + neg_emoji
	test_set = nrev[ncutoff:] + prev[pcutoff:]
	test_classifier = NaiveBayesClassifier.train(train_set)

	print ("Accuracy is : ", util.accuracy(test_classifier, test_set) * 100)
