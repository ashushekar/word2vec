import bs4
import nltk
import re
import urllib.request
from nltk.corpus import stopwords
from gensim.models import Word2Vec

scrapped_data = urllib.request.urlopen("https://en.wikipedia.org/wiki/Artificial_intelligence")
article = scrapped_data.read()

parsed_article = bs4.BeautifulSoup(article, 'lxml')
paragraphs = parsed_article.find_all('p')

article_text = ""
for para in paragraphs:
    article_text += para.text

# cleaning the text
processed_article = article_text.lower()
# remove all punctuations or other symbols
processed_article = re.sub('[^a-zA-Z]', ' ', processed_article)
# replace double space with a single space
processed_article = re.sub(r'\s+', ' ', processed_article)

# preparing the dataset
all_sentences = nltk.word_tokenize(processed_article)

all_words = [nltk.word_tokenize(sent) for sent in all_sentences]

# removing stop words
for i in range(len(all_words)):
    all_words[i] = [w for w in all_words[i] if w not in stopwords.words('english')]

# Word2Vec using Gensim
word2vec = Word2Vec(all_words, min_count=2)

# Finding Vectors for a word
v1 = word2vec.wv['artificial']

# Find Similar Words
sim_words = word2vec.wv.most_similar('intelligence')
print(sim_words)
