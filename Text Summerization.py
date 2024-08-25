#DataFlair Project
#import all required libraries

import numpy as np
import pandas as pd
import pickle
from statistics import moode
import nltk
from nltk import word_tokenize
from nltk.stem import LancasterStemmer
nltk.download('wordnet')
nltk.download('punkt')
from nltk.corpus import stopwords
from tensorflow.keras.models import Moddel
from tensorflow.keras import models
from tensorflow.keras import backend as K
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.utils import plot_model
from tensorflow.keras.layers import Input,LSTM,Embedding,Dense,Concatenate,Attention
from sklearn.model_selection import train_test_split
from bs4 import BeautifulSoup

#read the dataset file for the text Summerizer
df = pd.read_csv('Reviews.csv',nrows=100000)
#drop the duplicate and na values  from the record
df.drop_duplicates(subset=['Text'],inplace=True)
df.dropna(axis=0,inplace=True)
input_data = df.loc[:,'Text']
target_data = df.loc[:,'Summary']
target.replace('',np.nan,inplace=True)

input_texts=[]
target_texts=[]
input_words=[]
target_words=[]
contractions=pickle.load(open("contractions.pkl","rb"))['contractions']
#initialize stop words and Lencaster Stemmer
stop_words=set(stopwords.words('english'))
stemm=LancasterStemmer()

def clean(texts,src):
    #remove the html tags
    texts = BeautifulSoup(texts,"lxml").text
    #tokenize the text into words
    words = word_tokenize(texts.lower())
    #filter words which contains \
    #integers or their length is less than or equal to 3
    words = list(filter(lambda w:(w.isalpha() and len(w)>=3),words))

#contracttion file to expand shortened words
words = [contractions[w] if w in contractions else w for w in words]
#stem the words to their root word and filter stop words
if src == "inputs":
    words= [stemm.stem(w) for w in words if w not in stop_words]
else:
    words= [w for w in words if w not in stop_words]
return words
for in_txt,tr_txt in zip(input_data,target_data):
    in_words = clean(inn_txt,"inputs")
    input_texts+=[' '.join(in_words)]
    input_words+= input_words
    #add 'sos' at start and 'eos' at end of text
    tr_words = clean("sos "+tr_txt+" eos","target")
    target_texts+= [' '.join(tr_words)]
    target_words+= tr_words


#store only unique words from input and target list of words
input_words= sorrted(list(set(input_words)))
target_words = sorted(list(set(target_words)))
num_in_words = len(input_words) #total no of  input words
num_tr_words = len(target_words) #total no of target words
#get the length of the  input and target texts which appears most often
max_in_length = mode([len(i) for i in input_texts])
max_tr_length = mode([len(i) for i in target_texts])

print('number of input words : ',num_in_words)
print('number of target words : ',num_tr_words)
print('maximum input length : ',max_in_length)
print('maximum target length : ',max_tr_len)

#split the input and target text into  80:20 ratio or testing size oof
x_train,x_test,y_train,y_test = train_test_split(input_texts,taget_texts,test_size=0.2,random_state=0)

#traiin the tokenizer witth all the words
in_tokenizer = Tokenizer()
in_tokenizer.fit_on_texts(x_train)
tr_tokenizer = Tokenizer()
tr_tokenizer.fit_on_texts(y_train)

#convert text into sequence of integers
#where the integers will be the index of that word
x_train = in_tokenizer.texts_to_sequences(x_train)
y_train = tr_tokenizer.texts_to_sequences(y_train)

#pad array of 0's if the length is less than the maximum length
en_in_data = pad_sequences(x_train, maxlen = max_in_len, padding='post')
dec_data = pad_sequences(y_train, maxlen = max_tr_len, padding='post')

#decoder input data will not include the last word
#i.e. 'eos' in decoder input data
dec_in_data = dec_data[:,:-1]
#decode target data will be one time step ahead as it will not include
#the first word i.e.'sos'
dec_tr_data = dec_data.reshape(len(dec_data),max_tr_len,1)[:,1:]

K.clear_session()
latent_dim = 500

#create input object of total number of encoder words
en_inputs = Input(shape=(max_in_len,))
en_embedding = Embedding(num_in_words+1, latent_dim)(en_inputs)