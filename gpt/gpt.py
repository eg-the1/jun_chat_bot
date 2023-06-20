import keras
from konlpy.tag import Okt
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
import os
import re
import math
PAD = "<PADDING>"
STA = "<STRAT>"
END = "<END>"
OOV = "<OOV>"
PAD_INDEX = 0
STA_INDEX = 1
END_INDEX = 2
OOV_INDEX = 3
ENCODER_INPUT  = 0
DECODER_INPUT  = 1
DECODER_TARGET = 2
max_sequences = 300
embedding_dim = 2000
lstm_hidden_dim = 2560
okt=Okt()
RE_FILTER = re.compile("[.,!?\"':;~()]")
# chatbot_data = pd.read_csv('C://Users//kds96//OneDrive//바탕 화면//PYTHON//gpt//ChatbotData.csv')
text = open("C://Users//kds96//OneDrive//바탕 화면//PYTHON//gpt//KakaoTalkChats.txt", 'r',encoding="utf-8-sig")
a = text.readlines()
chatbot_data=[]
chatbot=[i.strip().split(",") for i in a if i.strip() != ""]
for i in chatbot:
    if len(i) == 1:
        pass
    else:
        split_result = i[1].split(':')
        if len(split_result) == 2:
            a, b = split_result
            chatbot_data.append([a.strip(),b.strip()])
try:
    for i in range(len(chatbot_data)):
        if chatbot_data[i][0]=='pass':
            continue
        if chatbot_data[i][0]=='김준서':
            state = True
            b = 0
            context=chatbot_data[i][1]
            while state:
                b += 1
                if chatbot_data[i+b][0]=='김준서':
                    context+=" "+chatbot_data[i+b][1]
                    chatbot_data[i+b][0]="pass"
                    chatbot_data[i+b][1]="pass"
                else:
                    state = False
            chatbot_data[i][1]=context
        if chatbot_data[i][0]=='세준':
            state = True
            b = 0
            context=chatbot_data[i][1]
            while state:
                b += 1
                if chatbot_data[i+b][0]=='세준':
                    context+=" "+chatbot_data[i+b][1]
                    chatbot_data[i+b][0]="pass"
                    chatbot_data[i+b][1]="pass"
                else:
                    state = False
            chatbot_data[i][1]=context
except:
    pass
chatbot_data = [i for i in chatbot_data if i[0] != "pass"]
question = []
answer= []
for i in range(len(chatbot_data)-1):
    question.append(chatbot_data[i][1]) 
    answer.append(chatbot_data[i+1][1])
del a
del chatbot_data
# question, answer = list(chatbot_data['Q']), list(chatbot_data['A'])
# question = question[0:3000]
#answer = answer[0:3000]

def pos_tag(sentences):
    sentence_pos = []

    for sentence in sentences:
        sentence = re.sub(RE_FILTER, "", sentence)
        # sentence = okt.morphs(sentence)
        sentence = " ".join(okt.morphs(sentence))
        sentence_pos.append(sentence)
    return sentence_pos
question = pos_tag(question)
answer = pos_tag(answer)

sentences = []
sentences.extend(question)
sentences.extend(answer)

words = []

for sentence in sentences:
    for word in sentence.split():
        words.append(word)

words = [word for word in words if len(word) > 0]
words = list(set(words))
words[:0] = [PAD, STA, END, OOV]
word_to_index = {word: index for index, word in enumerate(words)}
index_to_word = {index: word for index, word in enumerate(words)}

def convert_text_to_index(sentences, vocabulary, type):
    sentences_index = []
    for sentence in sentences:
        sentence_index=[]
        if type == DECODER_INPUT:
            sentence_index.extend([vocabulary[STA]])
        for word in sentence.split():
            if vocabulary.get(word) is not None:
                sentence_index.extend([vocabulary[word]])
            else:
                sentence_index.extend([vocabulary[OOV]])
        if type == DECODER_TARGET:
            if len(sentence_index) >= max_sequences:
                sentence_index = sentence_index[:max_sequences-1] + [vocabulary[END]]
            else:
                sentence_index += [vocabulary[END]]
        else:
            if len(sentence_index) > max_sequences:
                sentence_index = sentence_index[:max_sequences]
        sentence_index += (max_sequences - len(sentence_index)) * [vocabulary[PAD]]
        sentences_index.append(sentence_index)
    return np.asarray(sentences_index)
def convert_index_to_text(indexs, vocabulary): 
    
    sentence = ''
    
    for index in indexs:
        if index == END_INDEX:
            break;
        elif vocabulary.get(index) is not None:
            sentence += vocabulary[index]
        else:
            sentence += vocabulary[OOV_INDEX]
            
        sentence += ' '

    return sentence


encoder_inputs = keras.layers.Input(shape=(None,))
encoder_outputs = keras.layers.Embedding(len(words), embedding_dim)(encoder_inputs)

encoder_outputs, state_h, state_c = keras.layers.LSTM(lstm_hidden_dim,
                                                dropout=0.1,
                                                # recurrent_dropout=0.5,
                                                return_state=True)(encoder_outputs)
encoder_states = [state_h,state_c]

decoder_inputs = keras.layers.Input(shape=(None,))

decoder_embedding = keras.layers.Embedding(len(words), embedding_dim)
decoder_outputs = decoder_embedding(decoder_inputs)
decoder_lstm = keras.layers.LSTM(lstm_hidden_dim,
                        dropout=0.1,
                        #    recurrent_dropout=0.5,
                        return_state=True,
                        return_sequences=True)
decoder_outputs, _, _ = decoder_lstm(decoder_outputs,
                                    initial_state=encoder_states)


decoder_dense = keras.layers.Dense(len(words), activation='softmax')
decoder_outputs = decoder_dense(decoder_outputs)

model = keras.models.Model([encoder_inputs, decoder_inputs], decoder_outputs)

model.compile(optimizer='rmsprop',
            loss='categorical_crossentropy',
            metrics=['acc'])

encoder_model = keras.models.Model(encoder_inputs,encoder_states)

decoder_state_input_h = keras.layers.Input(shape=(lstm_hidden_dim,))
decoder_state_input_c = keras.layers.Input(shape=(lstm_hidden_dim,))
decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]

decoder_outputs = decoder_embedding(decoder_inputs)

decoder_outputs, state_h, state_c = decoder_lstm(decoder_outputs,
                                                initial_state=decoder_states_inputs)
decoder_states = [state_h, state_c]
decoder_outputs = decoder_dense(decoder_outputs)
decoder_model = keras.models.Model([decoder_inputs] + decoder_states_inputs,
                    [decoder_outputs] + decoder_states)

size = int(len(question)/20)
for _ in range(20):
    start = _*size
    end = (_+1)*size
    x_encoder = convert_text_to_index(question[start:end], word_to_index, ENCODER_INPUT)
    x_decoder = convert_text_to_index(answer[start:end], word_to_index, DECODER_INPUT)
    y_decoder = convert_text_to_index(answer[start:end], word_to_index, DECODER_TARGET)
    one_hot_data = np.zeros((len(y_decoder), max_sequences, len(words)))

    for i, sequence in enumerate(y_decoder):
        for j, index in enumerate(sequence):
            one_hot_data[i, j, index] = 1

    y_decoder = one_hot_data



    print('Total Epoch :', _ + 1)

    history = model.fit([x_encoder, x_decoder],
                        y_decoder,
                        epochs=100,
                        batch_size=4,
                        verbose=0)
    model.compile(optimizer='rmsprop',
        loss='categorical_crossentropy',
        metrics=['acc'])
    print('accuracy :', history.history['acc'][-1])
    print('loss :', history.history['loss'][-1])
    
    print()
encoder_model.save('./model/seq2seq_chatbot_encoder_model.h5')
decoder_model.save('./model/seq2seq_chatbot_decoder_model.h5')

with open('./model/word_to_index.pkl', 'wb') as f:
    pickle.dump(word_to_index, f, pickle.HIGHEST_PROTOCOL)
with open('./model/index_to_word.pkl', 'wb') as f:
    pickle.dump(index_to_word, f, pickle.HIGHEST_PROTOCOL)
os.system("shutdown -s -t 0")