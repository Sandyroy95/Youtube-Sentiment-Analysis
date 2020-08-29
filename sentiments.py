import Sentiment.training_classifier as tcl
from nltk.tokenize import word_tokenize	
import os.path
import pickle
from statistics import mode
from nltk.classify import ClassifierI
from nltk.metrics import BigramAssocMeasures
from nltk.collocations import BigramCollocationFinder as BCF
import itertools
import plotly.graph_objects as go
import Sentiment.get_tweets as tw

def features(words):
	temp = word_tokenize(words)

	words = [temp[0]]
	for i in range(1, len(temp)):
		if temp[i] != temp[i - 1]:
			words.append(temp[i])

	scoreF = BigramAssocMeasures.chi_sq

	#bigram count
	n = 150

	bigrams = BCF.from_words(words).nbest(scoreF, n)

	return dict([word,True] for word in itertools.chain(words, bigrams))

class VoteClassifier(ClassifierI):
	def __init__(self, *classifiers):
		self.__classifiers = classifiers

	def classify(self, comments):
		votes = []
		for c in self.__classifiers:
			v = c.classify(comments)
			votes.append(v)
		con = mode(votes)

		choice_votes = votes.count(mode(votes))
		conf = (1.0 * choice_votes) / len(votes)

		return con, conf

def sentiment(comments, query, count):
	if not os.path.isfile('classifier.pickle'):
		tcl.training()

	fl = open('classifier.pickle','rb')
	classifier = pickle.load(fl)
	fl.close()

	pos = 0
	neg = 0
	neu = 0
	posCmt = []
	negCmt = []
	posCount = 0
	negCount = 0

	for words in comments:
		comment = features(words)
		sentiment_value, confidence = VoteClassifier(classifier).classify(comment)

		if sentiment_value == 'positive':# and confidence * 100 >= 60:
			pos += 1
			if posCount < 5:
				posCmt.append(words)
				posCount += 1
		elif sentiment_value == 'negative':
			neg += 1
			if negCount < 5:
				negCmt.append(words)
				negCount += 1
		else:
			neu += 1

	print("\nFirst 5 positive youtube comments:")
	for p in posCmt:
		print(p)
		print("---")

	print("\nFirst 5 negative youtube comments: ")
	for n in negCmt:
		print(n)
		print("---")

	tweets = tw.TwitterClient.tweetsSentiment(query, count)
	tweetPos = tweets['pos']
	tweetsNeg = tweets['neg']
	tweetsNeu = tweets['neu']

	# print("youtube positive: ", pos)
	# print("youtube negative: ", neg)
	# print("youtube neutral: ", neu)
	# print("tweets positive: ", tweetPos)
	# print("tweets negative: ", tweetsNeg)
	# print("tweets neutral: ", tweetsNeu)

	print ("Positive sentiment : ", ((pos + tweetPos) * 100.0 /(len(comments) + tweetsNeu+tweetPos+tweetsNeg) ))
	print ("Negative sentiment : ", ((neg + tweetsNeg) * 100.0 /(len(comments) + tweetsNeu+tweetPos+tweetsNeg) ))
	print("Neutral sentiment : ", ((neu + tweetsNeu) * 100.0 /(len(comments) + tweetsNeu+tweetPos+tweetsNeg) ))

	sent = ['Positive', 'Negative', 'Neutral']
	fig = go.Figure([go.Bar(x=sent, y=[((pos + tweetPos) * 100.0 /(len(comments) + tweetsNeu+tweetPos+tweetsNeg) ), ((neg + tweetPos) * 100.0 /(len(comments) + tweetsNeu+tweetPos+tweetsNeg) ), ((neu + tweetPos) * 100.0 /(len(comments) + tweetsNeu+tweetPos+tweetsNeg) )])])
	fig.show()
