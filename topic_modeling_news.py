#!/usr/bin/env python
# coding: utf-8

# # Imports
# 
# Zona em que são definidos os imports necessários.

# In[1]:


import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning) 
import json
import re
import numpy as np
import random
import spacy
import nltk
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk import chunk
import gensim
from gensim import corpora
from gensim.models import CoherenceModel
from gensim.models import HdpModel
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import collections
import unicodedata
import pandas as pd
from operator import itemgetter


# # Funções auxiliares
# 
# Funções que são utilizadas ao longo do código.

# Função que ordena um dicionário descendentemente de acordo com os seus values.

# In[2]:


def sort_dict(dictionary):
    sorted_dict = sorted(dictionary.items(), key=lambda item: item[1], reverse=True)
    sorted_dict = collections.OrderedDict(sorted_dict)
    sorted_dict = dict(sorted_dict)
    return sorted_dict


# Função que devolve uma lista com as palavras dos tópicos.

# In[3]:


def words_per_topic(topics):
    topics_list = []
    for topicid, topic in topics:
        words_per_topic = []
        for word, weight in topic:
            words_per_topic.append(word)
        topics_list.append(words_per_topic)
    return topics_list


# Função que devolve uma lista dicionários de das palavras e pesos por tópicos tópicos.

# In[4]:


def words_weigths_per_topic(topics):
    topics_list = []
    for topicid, topic in topics:
        words_weigths_per_topic = {}
        for word, weight in topic:
            words_weigths_per_topic.update({word:weight})
        topics_list.append(words_weigths_per_topic)
    return topics_list


# Função que devolve um dicionário com o key igual ao id do topico e o value como o nome do topico.
# Baseia-se a partir dos ficheiros/bibliotecas importados. Para cada ficheiro/biblioteca é criado um incrementador, é verificado quantas palavras de cada tópico existem, e para cada palavra igual, incrementa o peso associado.
# No final verifica qual é o nome do tópico que tem maior número de incremetador e adiciona a um dicionário o id desse tópico e o respectivo nome.
# É possivel verificar com maior detalhe o funcionamento desta função de busca, descomentando o que está comentado.

# In[5]:


def define_topics_names(words_weigths_per_topic, books_words_list, cars_words_list, computers_words_list, cookware_words_list, hotels_words_list, movies_words_list, music_words_list, phones_words_list):
    topics_weight = {}
    id_topics_weight = {}
    id_topic = -1
    for topics in words_weigths_per_topic:
        id_topic += 1
        #print("Palavras do tópico {}:".format(id_topic))
        #print(topics)
        books_weight = 0
        cars_weight = 0
        computers_weight = 0
        cookware_weight = 0
        hotels_weight = 0
        movies_weight = 0
        music_weight = 0
        phones_weight = 0
        for word_in_topic, weight_in_topic in topics.items():
            for word_books_list in books_words_list:
                if word_in_topic == word_books_list:
                    books_weight += abs(weight_in_topic)
                    #print("Para a biblioteca Books encontrou as palavras: " + str(word_in_topic))
            for word_cars_list in cars_words_list:
                if word_in_topic == word_cars_list:
                    cars_weight += abs(weight_in_topic)
                    #print("Para a biblioteca Cars encontrou as palavras: " + str(word_cars_list))
            for word_computers_list in computers_words_list:
                if word_in_topic == word_computers_list:
                    computers_weight += abs(weight_in_topic)
                    #print("Para a biblioteca Computers encontrou as palavras: " + str(word_computers_list))
            for word_cookware_list in cookware_words_list:
                if word_in_topic == word_cookware_list:
                    cookware_weight += abs(weight_in_topic)
                    #print("Para a biblioteca Cookware encontrou as palavras: " + str(word_cookware_list))
            for word_hotels_list in hotels_words_list:
                if word_in_topic == word_hotels_list:
                    hotels_weight += abs(weight_in_topic)
                    #print("Para a biblioteca Hotels encontrou as palavras: " + str(word_hotels_list))
            for word_movies_list in movies_words_list:
                if word_in_topic == word_movies_list:
                    movies_weight += abs(weight_in_topic)
                    #print("Para a biblioteca Movies encontrou as palavras: " + str(word_movies_list))
            for word_music_list in music_words_list:
                if word_in_topic == word_music_list:
                    music_weight += abs(weight_in_topic)
                    #print("Para a biblioteca Music encontrou as palavras: " + str(word_music_list))
            for word_phones_list in phones_words_list:
                if word_in_topic == word_phones_list:
                    phones_weight += abs(weight_in_topic)
                    #print("Para a biblioteca Phones encontrou as palavras: " + str(word_phones_list))
                    
                      
        topics_weight = {'Books':books_weight, 'Cars':cars_weight, 'Computers':computers_weight, 'Cookware':cookware_weight, 'Hotels':hotels_weight, 'Movies':movies_weight, 'Music':music_weight, 'Phones':phones_weight}
        sorted_topics_weight = sort_dict(topics_weight)
        id_topics_weight.update({id_topic:sorted_topics_weight})
    #print(id_topics_weight)
    #print("")
    
    final_dict = {}
    for key, value in id_topics_weight.items():
        id_topic = key
        iterate_items = 0
        for topic, weight in value.items():
            if iterate_items == 0:
                topic_name_1 = topic
                max_value_1 = weight
            if iterate_items == 1:
                if max_value_1 == weight:
                    topic_name_2 = topic
                    max_value_2 = weight
                    topics_names = str(topic_name_1) + ", " + str(topic_name_2)
                    final_dict.update({id_topic:topics_names})
                else:
                    final_dict.update({id_topic:topic_name_1})
                if max_value_1 == 0:
                    final_dict.update({id_topic:"???"})
            iterate_items += 1
    #print(final_dict)
    #print("")
    return final_dict


# # Importação dos dados
# 
# Importação dos dados para um dicionário.
# Visualização das keys dos mesmos.
# Criação de uma lista apenas com "text".
# É excluido "recommended" porque não nos interessa.
# Não é realizado o baralhamento dos mesmos porque na inferência é necessário verificar manualmente a que tópicos correspondem.

# In[6]:


dataset = []
for review in open("../TM/data/en/SFU_Review_Corpus.json", "r"):
    dataset.append(json.loads(review))
print(dataset[0].keys())

reviews = []
for review in dataset:
    reviews.append(review["text"])
    
print("Existem " + str(len(reviews)) + " reviews")


# # Importação dos ficheiros
# 
# Importação dos ficheiros/bibliotecas com as palarvas mais relacionadas com os tópicos.

# ## Books

# In[7]:


books_words_list = []
with open('../TM_Trabalho_2/Topics/books.csv', encoding='ISO-8859-1') as csvfile:
    for row in csvfile:
        row = re.sub("[\n]", "", row)
        books_words_list.append(row)


# ## Cars

# In[8]:


cars_words_list = []
with open('../TM_Trabalho_2/Topics/cars.csv', encoding='ISO-8859-1') as csvfile:
    for row in csvfile:
        row = re.sub("[\n]", "", row)
        cars_words_list.append(row)


# ## Computers

# In[9]:


computers_words_list = []
with open('../TM_Trabalho_2/Topics/computers.csv', encoding='ISO-8859-1') as csvfile:
    for row in csvfile:
        row = re.sub("[\n]", "", row)
        computers_words_list.append(row)


# ## Cookware

# In[10]:


cookware_words_list = []
with open('../TM_Trabalho_2/Topics/cookware.csv', encoding='ISO-8859-1') as csvfile:
    for row in csvfile:
        row = re.sub("[\n]", "", row)
        cookware_words_list.append(row)


# ## Hotels

# In[11]:


hotels_words_list = []
with open('../TM_Trabalho_2/Topics/hotels.csv', encoding='ISO-8859-1') as csvfile:
    for row in csvfile:
        row = re.sub("[\n]", "", row)
        hotels_words_list.append(row)


# ## Movies

# In[12]:


movies_words_list = []
with open('../TM_Trabalho_2/Topics/movies.csv', encoding='ISO-8859-1') as csvfile:
    for row in csvfile:
        row = re.sub("[\n]", "", row)
        movies_words_list.append(row)


# ## Musics

# In[13]:


music_words_list = []
with open('../TM_Trabalho_2/Topics/musics.csv', encoding='ISO-8859-1') as csvfile:
    for row in csvfile:
        row = re.sub("[\n]", "", row)
        music_words_list.append(row)


# ## Phones

# In[14]:


phones_words_list = []
with open('../TM_Trabalho_2/Topics/phones.csv', encoding='ISO-8859-1') as csvfile:
    for row in csvfile:
        row = re.sub("[\n]", "", row)
        phones_words_list.append(row)


# # Remoção da pontuação
# 
# Função que remove a pontuação.

# In[15]:


def punctuation_treatment(review):
    review = re.sub(r"[^\w\s]|_", " ", review)
    review = re.sub("[\n\r]", " ", review)
    review = re.sub("_"," ", review)
    review = re.sub(" +"," ",review)
    review = re.sub(" +$","",review)
    review = re.sub("^ +","",review)
    return review


# # Minúsculas
# 
# Função que altera todos os caracteres da review para minúsculas.

# In[16]:


def lower_case_treatment(review):
    return review.lower()


# # Stop words
# 
# Função que remove as stop words.

# In[17]:


def stop_words_treatment(review):
    stop_words = set(stopwords.words('english'))
    words = word_tokenize(review)
    correct_review = [] 
    for word in words: 
        if word not in stop_words: 
            correct_review.append(word)
    return " ".join(correct_review)


# # Números
# 
# Função que remove todos os números.

# In[18]:


def numbers_treatment(review):
    review = re.sub(r"[0-9]", "", review)
    return review


# # Números cardinais e ordinais
# 
# Função que remove todos os números cardinais e ordinais.

# In[19]:


def cardinal_ordinal_numbers_treatment(review):
    review = re.sub(r"\b((?i)(one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|nineteen|twenty|thirty|forty|fifty|sixty|seventy|eighty|ninety|first|second|third|fourth|fifth|sixth|seventh|eighth|ninth|tenth))\b", "", review)
    return review


# # Um Caracter
# 
# Função que remove todas as palavras com apenas 1 caracter.

# In[20]:


def one_character_treatment(review):
    review = re.sub(r"(?<!\S).(?!\S)\s*", "", review)
    return review


# # Lemmatization

# In[21]:


def lemmatizer_treatment(review):
    lemmatizer = WordNetLemmatizer()
    lemmatizer_list = [lemmatizer.lemmatize(word) for word in review.split()]
    new_text = " ".join(lemmatizer_list)
    return new_text


# # POS tagging
# 
# Apenas guarda os Nomes.

# In[22]:


def pos_treatment(review):
    words = word_tokenize(review)
    pos_tag_text = nltk.pos_tag(words)
    review = " ".join([word[0] for word in pos_tag_text if word[1] in ['NN', 'NNS', 'NNP', 'NNPS']])
    return review


# # Noun Phrase Chunking

# In[23]:


def chunking_treatment(review):
    nlp = spacy.load("en_core_web_sm")
    review = " ".join(word.text for word in nlp(review).noun_chunks)
    return review


# # Tratamento dos dados
# 
# Função para o tratamento dos dados.
# 

# In[24]:


def data_treatment(review, with_lower_case, with_lemmatizer, with_pos, with_chuncking):
    if with_lower_case:
        review = lower_case_treatment(review)
    review = numbers_treatment(review)
    review = cardinal_ordinal_numbers_treatment(review)
    review = punctuation_treatment(review)
    review = one_character_treatment(review)
    review = stop_words_treatment(review)
    if with_lemmatizer:
        review = lemmatizer_treatment(review)
    if with_chuncking:   
        review = chunking_treatment(review)
    if with_pos:
        review = pos_treatment(review)
    return review


# # Preparação dos dados
# 
# Tratamento das reviews e divisão de dados em 10 reviews para inferência de tópicos e os restantes para treino.

# Nesta fase o tratamento dos dados irá se basear em:
# 
# - Converter todas as palavras para minúsculas;
# - Remover todos os números;
# - Remover todos os números cardinais e ordinais;
# - Remover toda a pontuação;
# - Remover todas as palavras com apenas 1 caracter;
# - Remover todas stop words.
# 
# Neste caso não é realizado o tratamento das minúsculas e portanto nesta fase iremos obter piores resultados porque as palavras nos ficheiros/bibliotecas estão em minúsculas, logo não encontra essas palavras.

# In[25]:


train_list = [data_treatment(review, True, False, False, False).split() for review in reviews[10:]]
test_list = [data_treatment(review, True, False, False, False).split() for review in reviews[:10]]

print(str(len(train_list)) + " documentos para treino")
print(str(len(test_list)) + " documentos para teste")


# # Document-Term Matrix
# 
# Construção da matriz de documentos por termos necessária à construção dos modelos de tópicos.
# Modelos com 8 tópicos e 15 palavras por tópico.

# In[26]:


dictionary = corpora.Dictionary(train_list)
doc_term_matrix = [dictionary.doc2bow(review) for review in train_list]
topics = ["Books", "Cars", "Computers", "Cookware", "Hotels", "Movies", "Music", "Phones"]

num_of_words = 15
num_of_topics = len(topics)
passes_lda = 40


# Função para visualizar o resultado dos modelos, por tópico e respectivas palavras e pesos.

# In[27]:


def view_topics(topics):
    list_of_topics_names = []
    list_of_words_per_topic = words_weigths_per_topic(topics)
    names_topics_dict = define_topics_names(list_of_words_per_topic, books_words_list, cars_words_list, computers_words_list, cookware_words_list, hotels_words_list, movies_words_list, music_words_list, phones_words_list)
    for id_topic, words in topics:
        words_weights = {}
        for word, weight in words:
            words_weights.update({word: weight})
        words_by_topic = ""
        iterator = 1
        for key, value in words_weights.items():
            if iterator < num_of_words:
                    words_by_topic = words_by_topic + "'" + str(key) + "'" + ":" + str(value) + ", "
            else:
                words_by_topic = words_by_topic + "'" + str(key) + "'" + ":" + str(value)
            iterator+=1
        for identifier, name in names_topics_dict.items():
            if id_topic == identifier:
                name_of_topic = name
        print("Topico: {} -> Palavras e pesos = {}\n".format(name_of_topic, words_by_topic))
        list_of_topics_names.append(name_of_topic)
    
    return list_of_topics_names


# # Construção dos modelos de tópicos
# 
# Modelos com 8 tópicos e 15 palavras por tópico.

# ### LSA

# In[28]:


def create_lsa(doc_term_matrix, dictionary, num_of_topics):
    lsa = gensim.models.lsimodel.LsiModel
    lsamodel = lsa(doc_term_matrix, num_topics = num_of_topics, id2word = dictionary)
    return lsamodel


# In[29]:


lsamodel = create_lsa(doc_term_matrix, dictionary, num_of_topics)
lsa_topics = lsamodel.show_topics(num_topics = num_of_topics, num_words = num_of_words, formatted=False)
view_topics(lsa_topics)


# ### LDA

# In[30]:


def create_lda(doc_term_matrix, dictionary, num_of_topics):
    lda = gensim.models.ldamodel.LdaModel
    ldamodel = lda(doc_term_matrix, num_topics = num_of_topics, id2word = dictionary, passes = passes_lda)
    return ldamodel


# In[31]:


ldamodel = create_lda(doc_term_matrix, dictionary, num_of_topics)
lda_topics = ldamodel.show_topics(num_topics = num_of_topics, num_words = num_of_words, formatted=False)
view_topics(lda_topics)


# # Os 3 tópicos mais importantes por modelo
# 
# Funções que devolvem uma lista com os 3 tópicos mais importantes por modelo.

# ### LSA
# 
# Devolve os 3 primeiros tópicos.

# In[32]:


def get_3_topics_most_important_lsa(lsa_topics):
    count_most_important_topics = 0
    topics_most_important_list = []
    for topic, words in lsa_topics:
        topics_most_important_list.append(topic)
        count_most_important_topics += 1
        if count_most_important_topics == 3:
            break
            
    return topics_most_important_list


# ### LDA
# 
# Devolve os 3 tópicos mais importantes, i.e., os tópicos com maiores pesos por documentos.

# In[33]:


def get_3_topics_most_important_lda(ldamodel):
    increment_0 = 0
    increment_1 = 0
    increment_2 = 0
    increment_3 = 0
    increment_4 = 0
    increment_5 = 0
    increment_6 = 0
    increment_7 = 0
    increments_dict = {}
    
    for doc in doc_term_matrix:
        dict_topics = {}
        if len(ldamodel[doc]) > 1:
            for topic, weight in ldamodel[doc]:
                dict_topics.update({topic:weight})
            dict_topics = sort_dict(dict_topics)
            topic_most_important = next(iter(dict_topics))
            
            if topic_most_important == 0:
                increment_0 += 1
            if topic_most_important == 1:
                increment_1 += 1
            if topic_most_important == 2:
                increment_2 += 1
            if topic_most_important == 3:
                increment_3 += 1
            if topic_most_important == 4:
                increment_4 += 1
            if topic_most_important == 5:
                increment_5 += 1
            if topic_most_important == 6:
                increment_6 += 1
            if topic_most_important == 7:
                increment_7 += 1
                
    increments_dict = {0:increment_0, 1:increment_1, 2:increment_2, 3:increment_3, 4:increment_4, 5:increment_5, 6:increment_6, 7:increment_7}
    increments_dict = sort_dict(increments_dict)
    topics_most_important_list = []
    count_most_important_topics = 0
    for key in increments_dict:
        topics_most_important_list.append(key)
        count_most_important_topics += 1
        if count_most_important_topics == 3:
            break
            
    return topics_most_important_list


# # Word Cloud
# 
# Funções para a criação da Word Cloud de acordo com o modelo dado como parâmetro.

# In[34]:


def get_words_and_weights(most_important_topics, topics):
    wordcloud_topics = {}
    for most_important in most_important_topics:
        for topic in topics:
            if most_important == topic[0]:
                for weight in topic[1]:
                    if weight[0] in wordcloud_topics:
                        wordcloud_topics[weight[0]] = max(wordcloud_topics[weight[0]], abs(weight[1]))
                    else:
                        wordcloud_topics[weight[0]] = abs(weight[1])                  
    return wordcloud_topics


def create_word_cloud(model):
    wordCloud = WordCloud(background_color="white")
    if model == "lsa":
        most_important_topics = get_3_topics_most_important_lsa(lsa_topics)
        wordcloud_topics = get_words_and_weights(most_important_topics, lsa_topics)
    if model == "lda":
        most_important_topics = get_3_topics_most_important_lda(ldamodel)
        wordcloud_topics = get_words_and_weights(most_important_topics, lda_topics)
    wordCloud.generate_from_frequencies(wordcloud_topics)
    plt.figure()
    plt.imshow(wordCloud)
    plt.axis("off")
    plt.show()


# ### LSA - Word Cloud

# In[35]:


create_word_cloud("lsa")


# ### LDA - Word Cloud

# In[36]:


create_word_cloud("lda")


# # Número de tópicos variados
# 
# Para cada modelo iremos testar com 3, 8 e 13 tópicos.

# ### Métricas LSA

# In[37]:


def lsa_num_topics(num_topics):
    lsamodel = create_lsa(doc_term_matrix, dictionary, num_topics)
    lsamodel_topics = lsamodel.show_topics(num_topics=num_topics, num_words=num_of_words, formatted=False)
    topics_list = words_per_topic(lsamodel_topics)
    lsamodel_coherence = CoherenceModel(topics=topics_list, texts=train_list, dictionary=dictionary, coherence='c_v').get_coherence()
    print("LSA com {} tópicos: {}".format(num_topics, lsamodel_coherence))
    
    return lsamodel_topics, lsamodel_coherence


# In[38]:


print("Coherence\n")
lsamodel_3_topics, lsamodel_3_coherence = lsa_num_topics(3)
lsamodel_8_topics,lsamodel_8_coherence = lsa_num_topics(8)
lsamodel_13_topics, lsamodel_13_coherence = lsa_num_topics(13)


# #### LSA com 3 tópicos

# In[39]:


view_topics(lsamodel_3_topics)


# #### LSA com 13 tópicos

# In[40]:


view_topics(lsamodel_13_topics)


# ### Métricas LDA

# In[41]:


def lda_num_topics(num_topics):
    ldamodel = create_lda(doc_term_matrix, dictionary, num_topics)
    ldamodel_topics = ldamodel.show_topics(num_topics=num_topics, num_words=num_of_words, formatted=False)
    topics_list = words_per_topic(ldamodel_topics)
    ldamodel_coherence = CoherenceModel(topics=topics_list, texts=train_list, dictionary=dictionary, coherence='c_v').get_coherence()
    print("LDA com {} tópicos\n".format(num_topics))
    print("Perplexity: {}".format(ldamodel.log_perplexity(doc_term_matrix)))
    print("Bound: {}".format(ldamodel.bound(doc_term_matrix)))
    print("Coherence: {}\n".format(ldamodel_coherence))
    
    return ldamodel_topics, ldamodel_coherence


# In[42]:


ldamodel_3_topics, ldamodel_3_coherence = lda_num_topics(3)
ldamodel_8_topics, ldamodel_8_coherence = lda_num_topics(8)
ldamodel_13_topics, ldamodel_13_coherence = lda_num_topics(13)


# #### LDA com 3 tópicos

# In[43]:


view_topics(ldamodel_3_topics)


# #### LDA com 13 tópicos

# In[44]:


view_topics(ldamodel_13_topics)


# ## Gráfico com a comparação da Coherence entre o modelo e o número de tópicos

# In[45]:


def create_histogram(model_list, coherence_list):
    y = np.arange(len(coherence_list))
    plt.bar(y, model_list)
    plt.xticks(y, coherence_list)
    plt.xlabel('Models')
    plt.ylabel('Coherence')
    plt.title('Coherence comparison')
    plt.show()


# In[46]:


model_list = ['LSA 3', 'LSA 8', 'LSA 13', 'LDA 3', 'LDA 8', 'LDA 13']
coherence_list = [lsamodel_3_coherence, lsamodel_8_coherence, lsamodel_13_coherence, ldamodel_3_coherence, ldamodel_8_coherence, ldamodel_13_coherence]

create_histogram(coherence_list, model_list)


# # Representação dos documentos

# Função que corre os modelos, mostrando a Coherence e os gráficos associados, devolvendo um dicionário com os nomes dos tópicos calculados por modelo.

# In[47]:


def run_models(reviews, approach):
    dictionary = corpora.Dictionary(reviews)
    doc_term_matrix = [dictionary.doc2bow(review) for review in reviews]
    
    print(approach)
    
    # LSA
    lsamodel = create_lsa(doc_term_matrix, dictionary, num_of_topics)
    lsa_topics = lsamodel.show_topics(num_topics=num_of_topics, num_words=num_of_words, formatted=False)
    lsa_topics_list = words_per_topic(lsa_topics)
    lsamodel_coherence = CoherenceModel(topics=lsa_topics_list, texts=reviews, dictionary=dictionary, coherence='c_v').get_coherence()
    print("\nLSA - Latent Semantic Indexing")
    print("Coherence: {}\n\n".format(lsamodel_coherence))
    lsa_topics_names = view_topics(lsa_topics)
    
    # LDA
    ldamodel = create_lda(doc_term_matrix, dictionary, num_of_topics)
    lda_topics = ldamodel.show_topics(num_topics=num_of_topics, num_words=num_of_words, formatted=False)
    lda_topics_list = words_per_topic(lda_topics)
    ldamodel_coherence = CoherenceModel(topics=lda_topics_list, texts=reviews, dictionary=dictionary, coherence='c_v').get_coherence()
    print("\nLDA - Latent Dirichlet Allocation")
    print("Coherence: {}\n\n".format(ldamodel_coherence))
    lda_topics_names = view_topics(lda_topics)
    
    # HDP
    hdpmodel = HdpModel(doc_term_matrix, dictionary)
    hdp_topics = hdpmodel.show_topics(num_topics=num_of_topics, num_words=num_of_words, formatted=False)
    hdp_topics_list = words_per_topic(hdp_topics)
    hdpmodel_coherence = CoherenceModel(topics=hdp_topics_list, texts=reviews, dictionary=dictionary, coherence='c_v').get_coherence()
    print("\nHDP - Hierarchical Dirichlet Process")
    print("Coherence: {}\n\n".format(hdpmodel_coherence))
    hdp_topics_names = view_topics(hdp_topics)
    
    topics_names_dictionary = {'lsa':lsa_topics_names, 'lda':lda_topics_names, 'hdp':hdp_topics_names}
    
    model_list = ['LSA', 'LDA', 'HDP']
    coherence_list = [lsamodel_coherence, ldamodel_coherence, hdpmodel_coherence]

    create_histogram(coherence_list, model_list)
    
    return topics_names_dictionary


# ### Com Minúsculas

# In[48]:


train_list_lowercase = [data_treatment(review, True, False, False, False).split() for review in reviews[10:]]
lowercase_dictionary = run_models(train_list_lowercase, "Com Minúsculas")


# ### Com Minúsculas e Lemmatization

# In[49]:


train_list_lowercase_lemmatization = [data_treatment(review, True, True, False, False).split() for review in reviews[10:]]
lowercase_lemmatization_dictionary = run_models(train_list_lowercase_lemmatization, "Com Minúsculas e Lemmatization")


# ### Com Minúsculas e POS tagging

# In[50]:


train_list_lowercase_pos = [data_treatment(review, True, False, True, False).split() for review in reviews[10:]]
lowercase_pos_dictionary = run_models(train_list_lowercase_pos, "Com Minúsculas e POS tagging")


# ### Com Minúsculas e Chunking

# In[51]:


train_list_lowercase_chunking = [data_treatment(review, True, False, False, True).split() for review in reviews[10:]]
lowercase_chunking_dictionary = run_models(train_list_lowercase_chunking, "Com Minúsculas e Chunking")


# ### Com Minúsculas Lemmatization e Chunking

# In[52]:


train_list_lowercase_lemmatization_chunking = [data_treatment(review, True, True, False, True).split() for review in reviews[10:]]
lowercase_lemmatization_chunking_dictionary = run_models(train_list_lowercase_lemmatization_chunking, "Com Minúsculas, Lemmatization e Chunking")


# ### Com Minúsculas POS e Chunking

# In[53]:


train_list_lowercase_pos_chunking = [data_treatment(review, True, False, True, True).split() for review in reviews[10:]]
lowercase_pos_chunking_dictionary = run_models(train_list_lowercase_pos_chunking, "Com Minúsculas, POS e Chunking")


# ### Com Minúsculas Lemmatization, POS e Chunking

# In[54]:


train_list_lowercase_lemmatization_pos_chunking = [data_treatment(review, True, True, True, True).split() for review in reviews[10:]]
lowercase_lemmatization_pos_chunking_dictionary = run_models(train_list_lowercase_lemmatization_pos_chunking, "Com Minúsculas, Lemmatization, POS e Chunking")


# # Número ideal de tópicos
# 
# Funções para descobrir qual é o número ideal de tópicos por modelo.

# Aplicação do tratamento com POS tagging porque é o que apresenta melhores resultados.

# In[55]:


def compute_coherence_values(model_type, dictionary, doc_term_matrix, texts, limit, start=2, step=3):
    coherence_values = []
    model_list = []
    for num_topics in range(start, limit, step):
        if model_type == 'lsa':
            lsamodel = create_lsa(doc_term_matrix, dictionary, num_topics)
            model_topics = lsamodel.show_topics(num_topics=num_topics, num_words=num_of_words, formatted=False)
        if model_type == 'lda':
            ldamodel = create_lda(doc_term_matrix, dictionary, num_topics)
            model_topics = ldamodel.show_topics(num_topics=num_topics, num_words=num_of_words, formatted=False)
        if model_type == 'hdp':
            hdpmodel = HdpModel(doc_term_matrix, dictionary)
            model_topics = hdpmodel.show_topics(num_topics=num_topics, num_words=num_of_words, formatted=False)
        model_list.append(model)
        topics_list = []
        for topicid, topic in model_topics:
            words_per_topic = []
            for word, weight in topic:
                words_per_topic.append(word)
            topics_list.append(words_per_topic)
        coherencemodel = CoherenceModel(model=model, topics=topics_list, texts=train_list, dictionary=dictionary, coherence='c_v').get_coherence()
        coherence_values.append(coherencemodel)

    return model_list, coherence_values


def optimal_number_of_topics(coherence_values, model):
    limit=40; start=2; step=6;
    x = range(start, limit, step)
    plt.plot(x, coherence_values)
    plt.xlabel("Num Topics")
    plt.ylabel("Coherence score")
    plt.legend(("coherence_values"), loc='best')
    plt.title(model)
    plt.show()
    
    for m, cv in zip(x, coherence_values):
        print("Numero de Topicos {} tem Coherence: {}".format(m, round(cv, 4)))


# ### LSA

# In[56]:


model = 'LSA'
dictionary = corpora.Dictionary(train_list_lowercase_pos)
doc_term_matrix = [dictionary.doc2bow(review) for review in train_list_lowercase_pos]
model_list, coherence_values = compute_coherence_values(model.lower(), dictionary=dictionary, doc_term_matrix=doc_term_matrix, texts=train_list, start=2, limit=40, step=6)
optimal_number_of_topics(coherence_values, model)


# In[57]:


lsamodel = create_lsa(doc_term_matrix, dictionary, 2)
lsa_topics = lsamodel.show_topics(num_topics=2, num_words=num_of_words, formatted=False)
view_topics(lsa_topics)


# ### LDA

# In[58]:


model = 'LDA'
dictionary = corpora.Dictionary(train_list_lowercase_pos)
doc_term_matrix = [dictionary.doc2bow(review) for review in train_list_lowercase_pos]
model_list, coherence_values = compute_coherence_values(model.lower(), dictionary=dictionary, doc_term_matrix=doc_term_matrix, texts=train_list, start=2, limit=40, step=6)
optimal_number_of_topics(coherence_values, model)


# In[59]:


ldamodel = create_lda(doc_term_matrix, dictionary, 8)
lda_topics = ldamodel.show_topics(num_topics=8, num_words=num_of_words, formatted=False)
view_topics(lda_topics)


# ### HDP

# In[60]:


model = 'HDP'
dictionary = corpora.Dictionary(train_list_lowercase_pos)
doc_term_matrix = [dictionary.doc2bow(review) for review in train_list_lowercase_pos]
model_list, coherence_values = compute_coherence_values(model.lower(), dictionary=dictionary, doc_term_matrix=doc_term_matrix, texts=train_list, start=2, limit=40, step=6)
optimal_number_of_topics(coherence_values, model)


# In[73]:


hdpmodel = HdpModel(doc_term_matrix, dictionary)
model_topics = hdpmodel.show_topics(num_topics=14, num_words=num_of_words, formatted=False)
view_topics(model_topics)


# # Inferência

# Categorização original dos 10 documentos de inferência.
# Esta categorização foi realizada manualmente, lendo cada uma das reviews.

# In[61]:


data = {'Documento':[0, 1, 2, 3, 4, 5, 6, 7, 8, 9], 'Tópico':['Books', 'Phones', 'Movies', 'Music', 'Books', 'Music', 'Computers', 'Music', 'Books', 'Music']} 
df = pd.DataFrame(data)
df


# Lista com os 10 documentos de inferência refletidos acima.

# In[62]:


real_topics = ['Books', 'Phones', 'Movies', 'Music', 'Books', 'Music', 'Computers', 'Music', 'Books', 'Music']


# Função que cria um dicionário com o nome do modelo como key e uma lista com o id dos tópicos com respectivo peso como value.

# In[63]:


def get_inference_dictionary(test_list, lsamodel, ldamodel, hdpmodel, dictionary):
    inference_list_lsa = []
    inference_list_lda = []
    inference_list_hdp = []
    inference_dictionary = {}
    for review in test_list:
        inference_list_lsa.append(lsamodel[dictionary.doc2bow(review)])
    inference_dictionary.update({'lsa':inference_list_lsa})
    for review in test_list:
        inference_list_lda.append(ldamodel[dictionary.doc2bow(review)])
    inference_dictionary.update({'lda':inference_list_lda})
    for review in test_list:
        inference_list_hdp.append(hdpmodel[dictionary.doc2bow(review)])
    inference_dictionary.update({'hdp':inference_list_hdp})

    return inference_dictionary


# Função que vai tentar preditar corretamente qual o nome do tópico para cada um dos 10 últimos reviews do conjunto de dados por modelo.
# Baseia-se no dicionário de treino já calculado acima que tem os tópicos que o modelo considerou, dependendo da abordagem utilizada, e baseia-se também num dicionário de teste/inferência, que contém o id do tópico e um peso.
# Percorre o dicionário de teste/inferência e procura pelo id do tópico que tem maior peso, de seguida pesquisa no dicionário de treino por esse mesmo id e devolve o nome do tópico.
# Pelo meio filtra o dicionário de teste/inferência por id <= 7 porque existem ids com valor maior que 7 e no nosso caso só temos 8 tópicos.
# Calcula para cada um dos modelos.

# In[94]:


def results(inference_dictionary, train_dictionary, approach):
    models = ['lsa', 'lda', 'hdp']
    result_lsa = 0
    result_lda = 0
    result_hdp = 0
    print("\n{}\n".format(approach))
    for model in models:
        inference_list = inference_dictionary[model]
        number_review = 0
        print("\nModelo: {}".format(model.upper()))
        
        reviews_cleaned = []
        for review in inference_list:
            review = [(i[0], abs(i[1])) for i in review if i[0] <= 7]
            reviews_cleaned.append(review)
        
        count_predict = 0
        for review in reviews_cleaned:
            id_topic = max(review, key=itemgetter(1))[0]
            predict_name = train_dictionary[model][id_topic]
            predict_name_list = predict_name.split(",")
            real_name = real_topics[number_review]
            print("\n\tReview número {}:".format(number_review))
            print("\t\tTópico real = {} \n\t\tTópico predicto = {}".format(real_name, predict_name))
            number_review += 1
            
            for predict_name in predict_name_list:
                if predict_name == real_name:
                    count_predict += 1
                if model == 'lsa':
                    result_lsa = count_predict
                if model == 'lda':
                    result_lda = count_predict
                if model == 'hdp':
                    result_hdp = count_predict
            
    print("\nResultados:")
    print("\tLSA = {}/10".format(result_lsa))
    print("\tLDA = {}/10".format(result_lda))
    print("\tHDP = {}/10".format(result_hdp))


# ## Inferência com Minúsculas

# In[96]:


dictionary = corpora.Dictionary(train_list_lowercase)
doc_term_matrix = [dictionary.doc2bow(review) for review in train_list_lowercase]

lsamodel = create_lsa(doc_term_matrix, dictionary, num_of_topics)
ldamodel = create_lda(doc_term_matrix, dictionary, num_of_topics)
hdpmodel = HdpModel(doc_term_matrix, dictionary)

lowercase_test_list = [data_treatment(review, True, False, False, False).split() for review in reviews[:10]]
inference_dictionary = get_inference_dictionary(lowercase_test_list, lsamodel, ldamodel, hdpmodel, dictionary)

results(inference_dictionary, lowercase_dictionary, "Inferência com Minúsculas")


# ## Inferência com Minúsculas e Lemmatization

# In[97]:


dictionary = corpora.Dictionary(train_list_lowercase_lemmatization)
doc_term_matrix = [dictionary.doc2bow(review) for review in train_list_lowercase_lemmatization]

lsamodel = create_lsa(doc_term_matrix, dictionary, num_of_topics)
ldamodel = create_lda(doc_term_matrix, dictionary, num_of_topics)
hdpmodel = HdpModel(doc_term_matrix, dictionary)

lowercase_lemmatization_test_list = [data_treatment(review, True, True, False, False).split() for review in reviews[:10]]
inference_dictionary = get_inference_dictionary(lowercase_lemmatization_test_list, lsamodel, ldamodel, hdpmodel, dictionary)

results(inference_dictionary, lowercase_lemmatization_dictionary, "Inferência com Minúsculas e Lemmatization")


# ## Inferência com Minúsculas e POS

# In[105]:


dictionary = corpora.Dictionary(train_list_lowercase_pos)
doc_term_matrix = [dictionary.doc2bow(review) for review in train_list_lowercase_pos]

lsamodel = create_lsa(doc_term_matrix, dictionary, num_of_topics)
ldamodel = create_lda(doc_term_matrix, dictionary, num_of_topics)
hdpmodel = HdpModel(doc_term_matrix, dictionary)

lowercase_pos_test_list = [data_treatment(review, True, False, True, False).split() for review in reviews[:10]]
inference_dictionary = get_inference_dictionary(lowercase_pos_test_list, lsamodel, ldamodel, hdpmodel, dictionary)

results(inference_dictionary, lowercase_pos_dictionary, "Inferência com Minúsculas e POS")


# ## Inferência com Minúsculas e Chunking

# In[108]:


dictionary = corpora.Dictionary(train_list_lowercase_chunking)
doc_term_matrix = [dictionary.doc2bow(review) for review in train_list_lowercase_chunking]

lsamodel = create_lsa(doc_term_matrix, dictionary, num_of_topics)
ldamodel = create_lda(doc_term_matrix, dictionary, num_of_topics)
hdpmodel = HdpModel(doc_term_matrix, dictionary)

lowercase_chunking_test_list = [data_treatment(review, True, False, False, True).split() for review in reviews[:10]]
inference_dictionary = get_inference_dictionary(lowercase_chunking_test_list, lsamodel, ldamodel, hdpmodel, dictionary)

results(inference_dictionary, lowercase_chunking_dictionary, "Inferência com Minúsculas e Chunking")


# ## Inferência com Minúsculas, Lemmatization e Chunking

# In[110]:


dictionary = corpora.Dictionary(train_list_lowercase_lemmatization_chunking)
doc_term_matrix = [dictionary.doc2bow(review) for review in train_list_lowercase_lemmatization_chunking]

lsamodel = create_lsa(doc_term_matrix, dictionary, num_of_topics)
ldamodel = create_lda(doc_term_matrix, dictionary, num_of_topics)
hdpmodel = HdpModel(doc_term_matrix, dictionary)

lowercase_lemmatization_chunking_test_list = [data_treatment(review, True, True, False, True).split() for review in reviews[:10]]
inference_dictionary = get_inference_dictionary(lowercase_lemmatization_chunking_test_list, lsamodel, ldamodel, hdpmodel, dictionary)

results(inference_dictionary, lowercase_lemmatization_chunking_dictionary, "Inferência com Minúsculas, Lemmatization e Chunking")


# ## Inferência com Minúsculas, POS e Chunking

# In[112]:


dictionary = corpora.Dictionary(train_list_lowercase_pos_chunking)
doc_term_matrix = [dictionary.doc2bow(review) for review in train_list_lowercase_pos_chunking]

lsamodel = create_lsa(doc_term_matrix, dictionary, num_of_topics)
ldamodel = create_lda(doc_term_matrix, dictionary, num_of_topics)
hdpmodel = HdpModel(doc_term_matrix, dictionary)

lowercase_pos_chunking_test_list = [data_treatment(review, True, False, True, True).split() for review in reviews[:10]]
inference_dictionary = get_inference_dictionary(lowercase_pos_chunking_test_list, lsamodel, ldamodel, hdpmodel, dictionary)

results(inference_dictionary, lowercase_pos_chunking_dictionary, "Inferência com Minúsculas, POS e Chunking")


# ## Inferência Com Minúsculas, Lemmatization, POS e Chunking

# In[114]:


dictionary = corpora.Dictionary(train_list_lowercase_lemmatization_pos_chunking)
doc_term_matrix = [dictionary.doc2bow(review) for review in train_list_lowercase_lemmatization_pos_chunking]

lsamodel = create_lsa(doc_term_matrix, dictionary, num_of_topics)
ldamodel = create_lda(doc_term_matrix, dictionary, num_of_topics)
hdpmodel = HdpModel(doc_term_matrix, dictionary)

lowercase_lemmatization_pos_chunking_test_list = [data_treatment(review, True, True, True, True).split() for review in reviews[:10]]
inference_dictionary = get_inference_dictionary(lowercase_lemmatization_pos_chunking_test_list, lsamodel, ldamodel, hdpmodel, dictionary)

results(inference_dictionary, lowercase_lemmatization_pos_chunking_dictionary, "Inferência com Minúsculas, Lemmatization, POS e Chunking")

