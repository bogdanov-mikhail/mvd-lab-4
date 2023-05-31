from datetime import datetime

import pandas as pd
import vk_api
from matplotlib import pyplot as plt
import numpy as np
import plotly.express as px
import squarify

app_id = 0
token = "0"

session: vk_api.VkApi = vk_api.VkApi(
    app_id=51641800,
    token="e7bc02eee7bc02eee7bc02ee8be4afff26ee7bce7bc02ee83e0836334ca6f928adbe0be"
)
api = session.get_api()

post_number = 1000
iters = (post_number // 100)
if iters != 500 / 100:
    iters += 1

dfs = []

for i in range(iters):
    batch = api.wall.get(domain="kinopoisk", count=100, offset=i * 100)
    posts = batch["items"]

    texts = []
    dates = []

    for post in posts:
        text = post["text"]
        date = post["date"]
        formatted_date = str(datetime.fromtimestamp(date))

        texts.append(text)
        dates.append(formatted_date)

    df = pd.DataFrame(data={"text": texts, "date": dates})
    dfs.append(df)

main_df = pd.concat(dfs, ignore_index=True)

main_df_copy = main_df.copy()

# Приведение к нижнему регистру
main_df_copy['text'] = main_df_copy['text'].astype(str)
main_df_copy['tokenized'] = main_df_copy['text'].apply(lambda x: x.lower()).astype('str')
main_df_copy

# Токенизация
import nltk
from nltk.tokenize import word_tokenize

nltk.download('stopwords')
nltk.download('punkt')


def get_tokens(text):
    tokens = word_tokenize(text)
    for i in range(len(tokens)):
        tokens[i] = tokens[i].strip()
    return tokens


main_df_copy["tokenized"] = main_df_copy['tokenized'].apply(get_tokens)
main_df_copy

# Убираем пунктуацию, цифры и эмоджи
import string
import re


def to_be_removed(token):
    if token in string.punctuation or token in string.digits or token == '...' or token in '«»—':
        return True
    # todo
    if re.search("["
                 u"\U0001F600-\U0001F64F"
                 u"\U0001F300-\U0001F5FF"
                 u"\U0001F680-\U0001F6FF"
                 u"\U0001F1E0-\U0001F1FF"
                 "]+", token, flags=re.UNICODE):
        return True
    return False


main_df_copy['tokenized'] = main_df_copy['tokenized'].apply(
    lambda row: [token for token in row if not to_be_removed(token)])
main_df_copy

# Нормализация
import pymorphy2

analyzer = pymorphy2.MorphAnalyzer()

main_df_copy['tokenized'] = main_df_copy['tokenized'].apply(
    lambda row: [analyzer.parse(token)[0].normal_form for token in row if token])
main_df_copy.sample(10)

# Убираем стоп слова
from nltk.corpus import stopwords

stops = stopwords.words("russian") + ["это", "который", "наш", "мочь", "год",
                                      "такой", "знать", "мы", "свой", "один", "другой", "хотеть",
                                      "человек", "всё", "все", "весь", "очень", "думать", "нужно",
                                      "большой", "время", "использовать", "говорить", "сказать",
                                      "иметь", "сделать", "первый", "каждый", "день", "её", "ваш",
                                      "стать", "больший", "ваше", "день", "самый", "понять",
                                      "просто", "ещё", "проблема", "также", "например", "делать",
                                      "вещь", "хороший", "спасибо", "й"]

main_df_copy["tokenized"] = main_df_copy['tokenized'].apply(lambda row: [token for token in row if token not in stops])
main_df_copy.sample(10)


# Частотный анализ униграмм

def freq_analise(data):
    df = pd.DataFrame(data, columns=['data'])
    df = pd.DataFrame(df['data'].value_counts()).sort_values(by="count", ascending=False)[:10].sort_values(by="count")

    plt.figure(figsize=(12,8), dpi= 80)

    labels = df.reset_index(drop=False).apply(lambda x: str(x[0]) + "\n (" + str(x[1]) + ")", axis=1)
    sizes = df['count'].values.tolist()
    colors = [plt.cm.Spectral(i / float(len(labels))) for i in range(len(labels))]
    squarify.plot(sizes=sizes, label=labels, color=colors, alpha=.8)

    plt.title('Words Unigram')
    plt.axis('off')
    plt.show()


unigrams = []
for token in main_df_copy['tokenized'].tolist():
    unigrams.extend(token)
freq_analise(unigrams)


# Частотный анализ биграмм

def create_bigram(tokens):
    global bigrams
    for i in range(len(tokens) - 1):
        bigrams.append(str(tokens[i]) + ' ' + str(tokens[i + 1]))


bigrams = []
main_df_copy['tokenized'].apply(create_bigram)
freq_analise(bigrams)

from collections import Counter
from wordcloud import WordCloud
from PIL import Image
import requests
from io import BytesIO

words = dict(Counter(unigrams))
url = "https://www.pinclipart.com/picdir/big/559-5596483_smile-for-the-camera-clip-art.png"
response = requests.get(url)
cloud_mask = np.array(Image.open(BytesIO(response.content)))

wc = WordCloud(background_color="white", max_words=200, mask=cloud_mask, colormap="cool")
wc.generate_from_frequencies(words)
plt.figure(figsize=(11, 11))
plt.imshow(wc, interpolation="bilinear")
plt.axis("off")
# plt.show()


pairs = []
for doc in main_df_copy['tokenized'].tolist():
    if doc:
        b = list((nltk.bigrams(doc)))
        if b:
            pairs.extend(b)
len(pairs)

pairs = [tuple(sorted(pair)) for pair in pairs]
word_pairs = dict(Counter(pairs))
print(len(word_pairs))
word_pairs = [(pair[0], pair[1], val) for pair, val in word_pairs.items() if val > 5]
print(len(word_pairs))
word_pairs[:10]

import networkx as nx

G = nx.Graph()
edges = word_pairs
edges[:10], len(edges)

import matplotlib.pyplot as plt
import matplotlib

plt.figure(figsize=(20, 20))
G.add_weighted_edges_from(edges)
labels = nx.get_edge_attributes(G, "weight")
pos = nx.spring_layout(G, k=0.5, iterations=50)
nx.draw(G, pos, with_labels=True)
nx.draw_networkx_edge_labels(G, pos, edge_labels=labels)
plt.show()

remove = [node for node, degree in dict(G.degree()).items() if degree < 2]
G.remove_nodes_from(remove)

remove_edge = [pair for pair in G.edges() if pair[0] in remove and pair[1] in remove]
G.remove_edges_from(remove_edge)

remove = [node for node, degree in dict(G.degree()).items() if degree < 1]
G.remove_nodes_from(remove)

node_sizes = [deg * 40 for node, deg in dict(G.degree()).items()]

plt.figure(figsize=(20, 20))
pos = nx.layout.spring_layout(G, k=0.5, iterations=50)
edges, weights = zip(*nx.get_edge_attributes(G, 'weight').items())
nx.draw(G, pos, node_color='#008000', node_size=node_sizes, edgelist=G.edges(), edge_color=range(len(G.edges())),
        width=2.0, with_labels=True, edge_cmap=plt.cm.get_cmap('autumn'))
plt.show()
