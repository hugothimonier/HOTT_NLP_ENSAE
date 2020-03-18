import numpy as np
from sklearn.decomposition import LatentDirichletAllocation as lda
from hott import sparse_ot

from sklearn.metrics.pairwise import euclidean_distances
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
from nltk.tokenize import TreebankWordTokenizer

from sklearn.metrics.pairwise import euclidean_distances
from sklearn.preprocessing import LabelEncoder
import re

def transform_dataframe(df):


    le = LabelEncoder()

    # on conserve que les colonnes qui nous interessent 
    df1 = df[['newDesk','commentBody']]
    # on encode la variable y qui correspond aux catégories du NYT
    df1[['newDesk_encoded']] = df1[['newDesk']].apply(lambda col: le.fit_transform(col))
    y = np.asarray(df1['newDesk_encoded'])

    # str preprocessing 
    df1['commentBody'] = df1['commentBody'].apply(lambda x: remove_html_tags(x))
    df1['commentBody'] = df1['commentBody'].apply(lambda x: re.sub(r'[^\w\s]','',x))
    df1['commentBody'] = df1['commentBody'].apply(lambda x: x.lower())

    #tokenization
    vocab_2 = df1['commentBody'].apply(lambda x: np.asarray(TreebankWordTokenizer().tokenize(x)))

    return vocab_2, y

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
    vocab_ : array of array containing each unique word in every comment, i.e. a word appearing twice will
             only  appear once in the position of its first appearance
    vocab_count : array of array containing the number of appearance of the words in each comment. The position
                  of the count corresponds to the position of the word in vocab_
    bow_data : bag of word representation of the data 
    """
  
    all_embed_vocab = {}
    with open(embed_path, 'r') as file:
        for line in file.readlines():
            word = line.split(' ')[0]
            embedding = [float(x) for x in line.split(' ')[1:]]
            all_embed_vocab[word] = embedding

    vocab = []
    embed_vocab = {}
    for txt in data:
        for i, w in enumerate(txt):
            if w not in vocab:
                if w in all_embed_vocab:
                    vocab.append(w)
                    embed_vocab[w] = all_embed_vocab[w]

    keep = []
    for txt in data :
        keep1 = []
        for word in txt :
            if word in vocab :
                keep1.append(True)
            else : 
                keep1.append(False)
        keep.append(keep1)

    for i in range(len(data)):
        data[i] = data[i][keep[i]]

    vocab_ = []
    vocab_count = []
    for txt in data:
        counts = word_count(txt)
        vocab_.append(np.asarray(list(counts.keys())))
        vocab_count.append(np.asarray(list(counts.values())))

    bow_data = np.zeros((len(data), len(vocab)),dtype=np.int)
    counts_ = dict()
    for idx, txt in enumerate(data):
        counts = word_count(txt)
        countz = dict()
        for key in counts.keys():
            countz[vocab.index(key)] = counts[key]
        counts_[idx] = countz
    for key in counts_.keys():
        for item in counts_[key].items():
            bow_data[key, item[0]] = item[1]

    return vocab, embed_vocab, vocab_, vocab_count, bow_data


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

    return (stemmed_reduced_vocab,
            stemmed_reduced_embed_vocab,
            stemmed_reduced_bow_data)


def fit_topics(data, embeddings, vocab, K):
    """Fit a topic model to bag-of-words data."""
    model = lda(n_components=K, max_iter=1500, random_state=1)
    model.fit(data)
    topics = model.components_
    lda_centers = np.matmul(topics, embeddings)
    print('LDA Gibbs topics')
    n_top_words = 20
    for i, topic_dist in enumerate(topics):
        topic_words = np.array(vocab)[np.argsort(topic_dist)][:-(n_top_words + 1):-1]
        print('Topic {}: {}'.format(i, ' '.join(topic_words)))
    print('\n')
    topic_proportions = model.transform(data)

    return topics, lda_centers, topic_proportions

def load_wmd(df, embed_path, stemming = True, K=70, p=1, n_word_keep = 20):

    data, y = transform_dataframe(df)
    y = y - 1

    if not stemming :
        vocab, embed_vocab, vocab_, vocab_count, bow_data = gen_data(data, embed_path)

    if stemming :
        vocab1, embed_vocab1, vocab_, vocab_count, bow_data1 = gen_data(data, embed_path)
        vocad, embed_vocab, bow_data = reduce_vocab(vocab1, embed_vocab1, bow_data1, embed_aggregate='mean') 

    embeddings = np.array([embed_vocab[w] for w in vocab])

    topics, lda_centers, topic_proportions = fit_topics(
        bow_data, embeddings, vocab, K)

    cost_embeddings = euclidean_distances(embeddings, embeddings) ** p
    cost_topics = np.zeros((topics.shape[0], topics.shape[0]))

    for k in range(K):
        to_0_idx = np.argsort(-topics[k])[n_word_keep:]
        topics[k][to_0_idx] = 0

    for i in range(cost_topics.shape[0]):
        for j in range(i + 1, cost_topics.shape[1]):
            cost_topics[i, j] = sparse_ot(topics[i], topics[j], cost_embeddings)
    cost_topics = cost_topics + cost_topics.T

    out = {'X': bow_data, 'y': y,
           'embeddings': embeddings,
           'topics': topics, 'proportions': topic_proportions,
           'cost_E': cost_embeddings, 'cost_T': cost_topics}

    return out




