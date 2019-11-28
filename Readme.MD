# Word2Vec in Python with Gensim Library

We will implement Word2Vec model with the help of Python's Gensim library

## Creating Corpus
In order to create a Word2Vec model, we need a corpus. In real-life applications, Word2Vec models are created using 
billions of documents. For instance **Google's Word2Vec** model is trained using 3 million words and phrases. However, 
for the sake of simplicity, we will create a Word2Vec model using a _Single Wikipedia article_. Our model will not be as 
good as Google's. Although, it is good enough to explain how Word2Vec model can be implemented using the _Gensim_ library.

To fetch wikipedia articles we need to download _Beautiful Soup_ library, which is a very useful library for web scrapping.

```sh
$ pip install beautifulsoup4
```
Another important library that we need to parse XML and HTML is the lxml library.
```sh
$ pip install lxml
```

The article we are going to scrape is the Wikipedia article on <https://en.wikipedia.org/wiki/Artificial_intelligence>.

```python
scrapped_data = urllib.request.urlopen("https://en.wikipedia.org/wiki/Artificial_intelligence")
article = scrapped_data.read()

parsed_article = bs4.BeautifulSoup(article, 'lxml')
paragraphs = parsed_article.find_all('p')

article_text = ""
for para in paragraphs:
    article_text += para.text
```

In the script above, we first download the _Wikipedia_ article using the _urlopen_ method of the request class of the 
_urllib_ library. We then read the article content and parse it using an object of the _BeautifulSoup_ class. Wikipedia 
stores the text content of the article inside _p_ tags. We use the find_all function of the BeautifulSoup object to fetch 
all the contents from the paragraph tags of the article.

Finally, we join all the paragraphs together and store the scraped article in _article\_text_ variable for later use.

## Preprocessing
The next step is to preprocess the content for Word2Vec model. 

```python
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
```

### Creating Word2Vec Model
With **Gensim**, it is extremely straightforward to create Word2Vec model. The word list is passed to the Word2Vec class 
of the _gensim.models_ package. We need to specify the value for the min_count parameter. A value of 2 for min_count 
specifies to include only those words in the Word2Vec model that appear at least twice in the corpus.

```python
# Word2Vec using Gensim
word2vec = Word2Vec(all_words, min_count=2)
``` 

### Model Analysis
We successfully created our Word2Vec model in the last section. Now is the time to explore what we created.

#### Finding Vectors for a Word
We know that the Word2Vec model converts words to their corresponding vectors. Let us see how we can view vector 
representation of any particular word.

```python
v1 = word2vec.wv['artificial']
```

The vector _v1_ contains the vector representation for the word "artificial". By default, a hundred dimensional vector 
is created by Gensim Word2Vec. This is a much, much smaller vector as compared to what would have been produced by bag of 
words. If we use the bag of words approach for embedding the article, the length of the vector for each will be 1206 
since there are 1206 unique words with a minimum frequency of 2. If the minimum frequency of occurrence is set to 1, the 
size of the bag of words vector will further increase. On the other hand, vectors generated through Word2Vec are not 
affected by the size of the vocabulary.

#### Finding Similar words
We can verify this by finding all the words similar to the word "intelligence".

```python
sim_words = word2vec.wv.most_similar('intelligence')
```
