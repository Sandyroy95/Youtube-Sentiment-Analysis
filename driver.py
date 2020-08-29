import comment_extract as CE
import sentiments as SYT

def main():
	# Sample VideoId = '6lAUk0uQWEc'
	videoId = input("Enter VideoId : ")
	# Sample Tweets Keyword = 'Death Stranding'
	twitterQuery = input("Enter Twitter Hashtag/Keyword: ")
	# Fetch the number of comments
	# if count = -1, fetch all comments
	count = int(input("Enter no. of comments/tweets to extract : "))
	comments = CE.commentExtract(videoId, count)
	SYT.sentiment(comments, twitterQuery, count)


if __name__ == '__main__':
	main()
