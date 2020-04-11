import numpy as np
from sklearn.decomposition import LatentDirichletAllocation as lda

import time
from sklearn.feature_extraction.text import CountVectorizer

from hott import sparse_ot
import progressbar
import nltk
from sklearn.metrics.pairwise import euclidean_distances
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
from nltk.tokenize import TreebankWordTokenizer

from sklearn.metrics.pairwise import euclidean_distances
from sklearn.preprocessing import LabelEncoder
import re


def balanced_sample_maker(X, y, sample_size, random_seed=None):
    """ return a balanced data set by sampling all classes with sample_size 
        current version is developed on assumption that the positive
        class is the minority.

    Parameters:
    ===========
    X: {numpy.ndarrray}
    y: {numpy.ndarray}
    """
    uniq_levels = np.unique(y)
    uniq_counts = {level: sum(y == level) for level in uniq_levels}

    if not random_seed is None:
        np.random.seed(random_seed)

    # find observation index of each class levels
    groupby_levels = {}
    for ii, level in enumerate(uniq_levels):
        obs_idx = [idx for idx, val in enumerate(y) if val == level]
        groupby_levels[level] = obs_idx
    # oversampling on observations of each label
    balanced_copy_idx = []
    for gb_level, gb_idx in groupby_levels.items():
        over_sample_idx = np.random.choice(gb_idx, size=sample_size, replace=True).tolist()
        balanced_copy_idx+=over_sample_idx
    np.random.shuffle(balanced_copy_idx)

    return (X[balanced_copy_idx], y[balanced_copy_idx], balanced_copy_idx)

def transform_dataframe(df, section = 'newDesk', balance = False):

	valid = {'sectionName', 'newDesk'}
	
	if section not in valid:
		raise ValueError("results: section must be one of %r." % valid)

	le = LabelEncoder()

	if section == 'newDesk':

		if balance : 
			unique, counts = np.unique(df['newDesk'], return_counts = True)
			a = dict(zip(unique, counts))

		# We delete the categories that have less than 200 observations 
			to_del = []
			for key in a.keys():
				if a[key] < 200 :
					to_del.append(key)
			df = df[~df['newDesk'].isin(to_del)].reset_index(drop = True)
			unique_, counts_ = np.unique(df['newDesk'], return_counts = True)
			a = dict(zip(unique_, counts_))
		
	# Create a balanced sample

			X = df['commentBody']
			y = df['newDesk']

			X_, y_ , idxs_ = balanced_sample_maker(X, y, 500, random_seed=None)
			df = df[df.index.isin(idxs_)].reset_index(drop = True)
			df = df[~df['newDesk'].isin(['Podcasts','NYTNow'])].reset_index(drop = True)

	# on conserve que les colonnes qui nous intéressent 
			df1 = df[['newDesk','commentBody']]
			df1.to_csv('reduced_dataframe.csv')
			del X_, y_ , idxs_

		else :
			df1 = df[['newDesk','commentBody']]

    	# on encode la variable y qui correspond aux catégories du NYT
		lib=np.array(df1[['newDesk']].copy())
		x=df1[['newDesk']].apply(lambda col: le.fit_transform(col))

	if section == 'sectionName' :

		if balance :

			unique, counts = np.unique(df['sectionName'], return_counts = True)
			a = dict(zip(unique, counts))

		# We delete the categories that have less than 200 observations 
			to_del = []
			for key in a.keys():
				if a[key] < 200 :
					to_del.append(key)
			df = df[~df['sectionName'].isin(to_del)].reset_index(drop = True)
			unique_, counts_ = np.unique(df['sectionName'], return_counts = True)
			a = dict(zip(unique_, counts_))
		
	# Create a balanced sample

			X = df['commentBody']
			y = df['sectionName']

			X_, y_ , idxs_ = balanced_sample_maker(X, y, 500, random_seed=None)
			df = df[df.index.isin(idxs_)].reset_index(drop = True)

		# on conserve que les colonnes qui nous intéressent 
			df1 = df[['sectionName','commentBody']]
			df1.to_csv('reduced_dataframe.csv')
			del X_, y_ , idxs_

		else : 
			df1 = df[['sectionName','commentBody']]

		# on encode la variable y qui correspond aux catégories du NYT
		
		lib=np.array(df1[['sectionName']].copy())
		x=df1[['sectionName']].apply(lambda col: le.fit_transform(col))



	y = np.asarray(x)

    # str preprocessing 
	a= df1['commentBody'].apply(lambda x: remove_html_tags(x))
	b= a.apply(lambda x: re.sub(r'[^\w\s]','',x))
	c= b.apply(lambda x: x.lower())
     
    #tokenization
	vocab_2 = c.apply(lambda x: np.asarray(TreebankWordTokenizer().tokenize(x)))

	return vocab_2, y, lib

def remove_html_tags(text):
    """Remove html tags from a string"""
    import re
    clean = re.compile('<.*?>')
    return re.sub(clean, '', text)


def word_count(str):
    counts = dict()

    for word in str:
        if word in counts:
            counts[word] += 1
        else:
            counts[word] = 1

    return counts


def gen_data(data, embed_path):


    """

    gen data in the right format:

    Input :

    data : a list of np arrays containing a comment
    embed_path : str variable containing the path to the embeding

    Output: 

    vocab : each word present in the dataset that has an embedding
    embed_vocab : dictionnary containing the embedding that corresponds to each word
    bow_data : bag of word representation of the data 
    """

    tic = time.time()
    
    all_embed_vocab = {}
    with open(embed_path, 'r') as file:
    	for line in progressbar.progressbar(file.readlines()):
    		word, vec = line.split(' ', 1)
    		all_embed_vocab[word] = np.fromstring(vec, sep=' ')
        
    
    S=set(np.concatenate(data))
    I=S.intersection(all_embed_vocab.keys())
    embed_vocab={ i : all_embed_vocab[i] for i in I}
    vocab=list(I)

    tf_vectorizer = CountVectorizer(max_df=1, min_df=1, #cut off
                                max_features=None,
                                stop_words=None,
                                vocabulary={v : i for i,v in enumerate(vocab)}
                                ) # output is a sparse matrix
    
    bow_data = tf_vectorizer.fit_transform([' '.join(txt) for txt in data])
    bow_data = bow_data.toarray()
    print("extraction done in %0.3fs." % (time.time() - tic))
    
    return vocab, embed_vocab, bow_data



def reduce_vocab(vocab, embed_vocab, bow_data, embed_aggregate='mean'):
    """

    Reduce vocabulary size by stemming and removing stop words.

    """
    vocab = np.array(vocab)
    short = np.array([len(w) > 2 for w in vocab])
    stop_words = set(stopwords.words('english'))
    stop = np.array([w not in stop_words for w in vocab])
    reduced_vocab = vocab[np.logical_and(short, stop)]
    reduced_bow_data = bow_data[:, np.logical_and(short, stop)]
    stemmer = SnowballStemmer("english")

    stemmed_dict = {}
    
    stemmed_idx_mapping = {}
    stemmed_vocab = []
    for i, w in enumerate(reduced_vocab):
        stem_w = stemmer.stem(w)
        if stem_w in stemmed_vocab:
            stemmed_dict[stem_w].append(w)
            stemmed_idx_mapping[stemmed_vocab.index(stem_w)].append(i)
        else:
            stemmed_dict[stem_w] = [w]
            stemmed_vocab.append(stem_w)
            stemmed_idx_mapping[stemmed_vocab.index(stem_w)] = [i]

    stemmed_bow_data = np.zeros((bow_data.shape[0], len(stemmed_vocab)),
                                dtype=np.int)
    for i in range(len(stemmed_vocab)):
        stemmed_bow_data[:, i] = reduced_bow_data[:, stemmed_idx_mapping[i]].sum(axis=1).flatten()

    word_counts = stemmed_bow_data.sum(axis=0)
    stemmed_reduced_vocab = np.array(stemmed_vocab)[word_counts > 2].tolist()
    stemmed_reduced_bow_data = stemmed_bow_data[:, word_counts > 2]

    stemmed_reduced_embed_vocab = {}
    for w in stemmed_reduced_vocab:
        old_w_embed = [embed_vocab[w_old] for w_old in stemmed_dict[w]]
        if embed_aggregate == 'mean':
            new_w_embed = np.mean(old_w_embed, axis=0)
        elif embed_aggregate == 'first':
            new_w_embed = old_w_embed[0]
        else:
            print('Unknown embedding aggregation')
            break
        stemmed_reduced_embed_vocab[w] = new_w_embed


    print("The vocabulary has been reduced from %s words to %s words. This represents a reduction of %s percent" %(len(vocab), len(stemmed_reduced_vocab), round((1-len(stemmed_reduced_vocab)/len(vocab))*100,2)))

    return (stemmed_reduced_vocab,
            stemmed_reduced_embed_vocab,
            stemmed_reduced_bow_data)

def print_top_words(model, feature_names, number_of_top_words=20):
    """print top words by topic in a given model
    """
    for topic_idx, topic in enumerate(model.components_):
        message = "Topic #%d: " % topic_idx
        message += " ".join([feature_names[i]
                             for i in topic.argsort()[:-number_of_top_words - 1:-1]])
        print(message)
    print()
    

def fit_topics(data, embeddings, vocab, K):
    """Fit a topic model to bag-of-words data."""
    
    tic = time.time()

    model = lda(n_components=K, max_iter=100, learning_method='online',learning_offset=50.,doc_topic_prior=1.,random_state=0,verbose=1)

    model.fit(data)
    topics = model.components_
    lda_centers = np.matmul(topics, embeddings)
    print('LDA Gibbs topics')
    n_top_words = 20
    print_top_words(model, vocab)
    topics_words = []
    for i, topic_dist in enumerate(topics):
        topic_words = np.array(vocab)[np.argsort(topic_dist)][:-(n_top_words + 1):-1]
        topics_words.append(topic_words)

    topic_proportions = model.transform(data)
    print("LDA fit done in %0.3fs." % (time.time() - tic))

    return topics, lda_centers, topic_proportions, topics_words

def load_data(df, embed_path, stemming = True, K=70, p=1, n_word_keep = 20, section = 'newDesk', balance = False):

	if section == 'sectionName' :
		if balance : 
			data, y, lib= transform_dataframe(df, section = 'sectionName', balance = True)
		else : 
			data, y, lib = transform_dataframe(df, section = 'sectionName')

	if section == 'newDesk' :
		if balance :
			data, y, lib = transform_dataframe(df, section = 'newDesk', balance = True)
		else : 
			data, y, lib = transform_dataframe(df, section = 'newDesk')

	y = y - 1


	if not stemming :
		vocab, embed_vocab, bow_data = gen_data(data, embed_path)

	if stemming :

		vocab1, embed_vocab1, bow_data1 = gen_data(data, embed_path)
		print("stemming")
		vocab, embed_vocab, bow_data = reduce_vocab(vocab1, embed_vocab1, bow_data1, embed_aggregate='mean') 

	embeddings = np.array([embed_vocab[w] for w in vocab])

	print("computing LDA")
	topics, lda_centers, topic_proportions, topics_words = fit_topics(
		bow_data, embeddings, vocab, K)

	print("computing distance")
	cost_embeddings = euclidean_distances(embeddings, embeddings) ** p
	cost_topics = np.zeros((topics.shape[0], topics.shape[0]))

	for k in range(K):
		to_0_idx = np.argsort(-topics[k])[n_word_keep:]
		topics[k][to_0_idx] = 0
    
	print("computing optimal transport calculation")
	for i in range(cost_topics.shape[0]):
		for j in range(i + 1, cost_topics.shape[1]):
			cost_topics[i, j] = sparse_ot(topics[i], topics[j], cost_embeddings)
	cost_topics = cost_topics + cost_topics.T

	out = {'vocab': vocab,'X': bow_data, 'y': y, 'lib': lib,
			'text' : data,
			'embeddings': embeddings,
			'topics': topics, 'proportions': topic_proportions, 'topic_words' : topics_words,
			'cost_E': cost_embeddings, 'cost_T': cost_topics}

	return out




