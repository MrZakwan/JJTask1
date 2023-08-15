# Sentiment Analysis Of Tweets Using The Natural Language Toolkit (NLTK)
#
# Reference: http://www.laurentluce.com/posts/twitter-sentiment-analysis-using-python-and-nltk/
#
# Goal: Using our dataset, classify tweets (students feedbacks) in positive, negative, and neutral
#       sentiments using NLTK & Naive Bayes
#
# Steps:
# ------
#   Step 1 — Installing NLTK Package & Downloading The Different Resources
#   Step 2 — Tokenizing The Data
#   Step 3 — Normalizing The Data
#   Step 4 — Cleaning The Data
#   Step 5 — Determining Word Density/Frequency
#   Step 6 — Preparing Data For The Model
#   Step 7 — Building & Testing The Model
#   Step 8 — Predicting New/Unseen Tweets
#
# Note: The following packages should be installed using pip as follows:
#   pip install colorama
#   pip install nltk==3.3
#
import re
import string
from colorama import Fore

import nltk
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.tag import pos_tag
from nltk.tokenize import word_tokenize


# function: lemmatize_tokens_list
# -------------------------------
def lemmatize_tokens_list(tokens_list):
    lemmatizer = WordNetLemmatizer()
    lemmatized_list = []
    for token, tag in pos_tag(tokens_list):
        if tag.startswith("NN"):
            pos = "n"  # noun
        elif tag.startswith("VB"):
            pos = "v"  # verb
        else:
            pos = "a"  # adjective
        lemmatized_list.append(lemmatizer.lemmatize(token, pos))
    return lemmatized_list


# function: build_dataset
# -----------------------
# Using tweets_list & stop_words_list arguments, create a single list of tuples, such that
#   each tuple contains two elements:
#       * the first element is a list containing the cleaned words/tokens
#       * the second element is the sentiment type
#
# Preprocessing Rules:
#   * we eliminate English stop words
#   * we use lowercase for everything
#
# Note 1: The following noisy items are removed from the list of tokens
#           * hyperlinks
#           * Twitter handles in replies: Twitter usernames preceded by an @ symbol
#           * punctuation & special characters
#           * English stop words
# Note 2: All cleaned tokens are normalized/lemmatized
# Note 3: All cleaned tokens are converted to lowercase
def build_dataset(tweets_list, stop_words_list=()):
    lemmatizer = WordNetLemmatizer()
    cleaned_tweets = []

    for a_tweet, sentiment in tweets_list:
        tokens_list = word_tokenize(a_tweet)
        filtered_tokens = []
        for token, tag in pos_tag(tokens_list):
            token = re.sub("http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+#]|[!*\(\),]|" \
                           "(?:%[0-9a-fA-F][0-9a-fA-F]))+", "", token)
            token = re.sub("(@[A-Za-z0-9_]+)", "", token)

            if tag.startswith("NN"):
                pos = "n"  # noun
            elif tag.startswith("VB"):
                pos = "v"  # verb
            else:
                pos = "a"  # adjective

            token = lemmatizer.lemmatize(token, pos)
            token = token.lower()

            if len(token) > 0 and token not in string.punctuation and token not in stop_words_list:
                filtered_tokens.append(token)

        cleaned_tweets.append((filtered_tokens, sentiment))
    return cleaned_tweets


# function: get_words_in_tweets
# -----------------------------
def get_words_in_tweets(tweets_list):
    all_words = []
    for words, sentiment in tweets_list:
        all_words.extend(words)
    return all_words


# function: get_word_features
# ---------------------------
def get_word_features(words_list):
    freq_dist = nltk.FreqDist(words_list)
    word_features = freq_dist.keys()
    return word_features


# function: extract_features
# --------------------------
# Note: This function reads from the global variable all_word_features and return a dictionary as
#       follows:
#           Key: "contains(<word>)"
#           Value: True if <word> is a feature or False otherwise
def extract_features(words_list):
    words_set = set(words_list)
    features = {}
    for word in all_word_features:
        features["contains(%s)" % word] = (word in words_set)
    return features


# display the title
print("\n ****************************************************************************************")
print(" Sentiments Analysis Of Tweets (Students Feedbacks) Using Natural Language Toolkit (NLTK)")
print(" ****************************************************************************************")

# define the class labels. Each label is a sentiment type
POSITIVE = "Positive"
NEGATIVE = "Negative"
NEUTRAL = "Neutral"

# create the list of positive tweets
positive_tweets = [
    ("I love the Math teacher", POSITIVE),
    ("The midterm exam is amazing", POSITIVE),
    ("I feel great this morning", POSITIVE),
    ("I am so excited about the sport event", POSITIVE),
    ("I like Math", POSITIVE),
    ("The last English lesson is very nice", POSITIVE),
    ("Some Biology lessons are interesting", POSITIVE),
    ("Math is wonderful", POSITIVE),
    ("I love Math Algebra", POSITIVE),
    ("I am so happy to be in this school", POSITIVE)
]

# create the list of negative tweets
negative_tweets = [
    ("I hate the third lesson in English", NEGATIVE),
    ("The Math Geometry is horrible", NEGATIVE),
    ("I feel tired this morning", NEGATIVE),
    ("The principal's speach is annoying", NEGATIVE),
    ("I hate Biology", NEGATIVE),
    ("We all hate Physics", NEGATIVE),
    ("Many students hate Geography", NEGATIVE),
    ("Some students hate waking up early", NEGATIVE),
    ("History is boring", NEGATIVE),
    ("Arithmetic division is my enemy", NEGATIVE)
]

# create the list of neutral tweets
neutral_tweets = [
    ("The school building is red", NEUTRAL),
    ("The school is located in Baghdad", NEUTRAL),
    ("The school bus has cameras", NEUTRAL),
    ("Water is allowed in class", NEUTRAL),
    ("We use blue pen in the exams", NEUTRAL),
    ("We take Art classes twice per week", NEUTRAL),
    ("The first session is always Math", NEUTRAL),
    ("Biology is science", NEUTRAL),
    ("There are 3 sections per grade", NEUTRAL),
    ("The tuition fees are affordable", NEUTRAL)
]

# display a summary about the 3 tweets lists
n_positive = len(positive_tweets)
n_negative = len(negative_tweets)
n_neutral = len(neutral_tweets)
n = n_positive + n_negative + n_neutral
p_positive = round(n_positive * 100 / n, 2)
p_negative = round(n_negative * 100 / n, 2)
p_neutral = round(n_neutral * 100 / n, 2)
print("", n, "Tweets Were Used For Training:")
print("\t", n_positive, "Positive Tweets (" + str(p_positive) + "%)")
print("\t", n_negative, "Negative Tweets (" + str(p_negative) + "%)")
print("\t", n_neutral, "Neutral  Tweets (" + str(p_neutral) + "%)")

# get the list of stop words in English
stop_words = stopwords.words("english")
print("\n", len(stop_words), "Stop Words In English:", stop_words)

# create the tweets dataset from the 3 lists above
tweets = build_dataset(positive_tweets + negative_tweets + neutral_tweets, stop_words)

# display the list of tweets
print("\n List Of Tweets:", tweets)

# extract & then display all words in tweets
words_in_tweets = get_words_in_tweets(tweets)
print("\n", len(words_in_tweets), "Total Words In Tweets:", words_in_tweets)

# determine the frequency distribution of all words to find out which are the most common words
frequency_distribution = nltk.FreqDist(words_in_tweets)

# display top n words/tokens
n = 15
top_n = frequency_distribution.most_common(n)
print("\n\n Top", n, "Words:")
print("\n Rank \t\t Word \t\t\t\t Frequency")
print(" -----------------------------------------")
for i in range(len(top_n)):
    rank = str(i + 1)
    if i < 9: rank = " " + rank
    print("", rank, " " * (10 - len(rank)), top_n[i][0],
          " " * (20 - len(top_n[i][0])), f"{top_n[i][1]:,}")

# extract & then display the word features from the tweets
all_word_features = get_word_features(words_in_tweets)
print("\n All Word Features:", all_word_features)

# extract & then display the features from a sample tweet
sample_tweet = "I LOVE Math Algebra & Arithmetics"
sample_tweet_words = word_tokenize(sample_tweet.lower())
features = extract_features(sample_tweet_words)
print("\n Features Extracted From The Sample Tweet:", sample_tweet)
print("\t", features)

# determine & then display the training set that contains the labeled feature sets
training_set = nltk.classify.apply_features(extract_features, tweets)
print("\n\n Training Set:", training_set, "\n")

# create and then train the Naive Bayes classifier
classifier = nltk.NaiveBayesClassifier.train(training_set)

# display the top 50 informative features
print(classifier.show_most_informative_features(50))

# create a list of tweets for testing
tweets2 = [
    ("I feel happy this morning", POSITIVE),
    ("A lot of students like the Math teacher", POSITIVE),
    ("My classmate hates French subject", NEGATIVE),
    ("The weather is so hot", NEGATIVE),
    ("The Science coordinator voice is annoying", NEGATIVE),
    ("We take sports every day", NEUTRAL),
    ("We play Football in the school", NEUTRAL),
    ("Many students hate cafeteria food", NEGATIVE),
    ("Teaching is an art", POSITIVE),
    ("The final exam is today", NEUTRAL)
]

# create the test_tweets dataset from tweets2 list above
test_tweets = build_dataset(tweets2, stop_words)

# display the list of test tweets
print("\n", len(tweets2), "Tweets Were Used For Testing:", test_tweets)

# determine the test set
test_set = nltk.classify.apply_features(extract_features, test_tweets)

# determine & then display the classification's accuracy
accuracy = round(nltk.classify.accuracy(classifier, test_set) * 100, 2)
print(Fore.MAGENTA, "\n Naive Bayes Classification's Accuracy Score: " + str(accuracy) + "%", Fore.BLACK)

# create some new tweets to predict
new_tweets = [
    ("I love my school", POSITIVE),
    ("The Geography sessions are annoying", NEGATIVE),
    ("We take 6 sessions per day", NEUTRAL),
    ("He likes reading", POSITIVE),
    ("A lot of students hate drop quizes", NEGATIVE),
    ("Baghdad is the capital city of Iraq", NEUTRAL),
    ("The principal is friendly", POSITIVE),
    ("Some students hate making projects", NEGATIVE),
    ("Physics is science", NEUTRAL),
    ("My handwriting is nice", POSITIVE),
    ("The weekly plan is so packed", NEGATIVE),
    ("The playground floor is green", NEUTRAL)
]

# display report header
print("\n\n Classifying New Tweets:")
print(" -----------------------")

# classify the above new tweets by predicting their class label (Positive, Negative, or Neutral)
i = 1
n_correct = 0
n_wrong = 0
for new_tweet in new_tweets:
    # classify this new tweet by performing a prediction
    predicted_sentiment = classifier.classify(extract_features(word_tokenize(new_tweet[0])))

    # compare prediction to actual
    if predicted_sentiment == new_tweet[1]:
        n_correct += 1
        result = "(Correct)"
        color = Fore.GREEN
    else:
        n_wrong += 1
        result = "(Wrong! Actual Sentiment Is " + new_tweet[1] + ")"
        color = Fore.RED

    # display the result
    print(color, "Tweet # " + str(i) + ":", new_tweet[0])
    print(Fore.BLACK, "\t ==> Predicted Sentiment:", predicted_sentiment, "\t", result, "\n")
    i += 1

n_tweets = len(new_tweets)
p_correct = round(n_correct * 100 / n_tweets, 2)
p_wrong = round(n_wrong * 100 / n_tweets, 2)
print(Fore.BLACK, "\n", n_tweets, "New Tweets:")
print(Fore.GREEN, "\t", n_correct, "Correct Classifications (" + str(p_correct) + "%)")
print(Fore.RED, "\t", n_wrong, "Wrong   Classifications (" + str(p_wrong) + "%)")
print(Fore.BLACK)

# predict tweets input from the keyboard
while True:
    input_tweet = input("\n ? Tweet To Classify: ")
    input_tweet = input_tweet.lower()
    tokens_list = word_tokenize(input_tweet)
    lemmatized_tokens_list = lemmatize_tokens_list(tokens_list)
    predicted_sentiment = classifier.classify(extract_features(lemmatized_tokens_list))
    print("\t ==> Predicted Sentiment:", predicted_sentiment)
