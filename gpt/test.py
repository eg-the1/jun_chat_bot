import pickle
from keras import models
from konlpy.tag import Okt
okt=Okt()
import random
from keras import layers
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import re

encoder_model = models.load_model('./model/seq2seq_chatbot_encoder_model.h5')
decoder_model = models.load_model('./model/seq2seq_chatbot_decoder_model.h5')
RE_FILTER = re.compile("[.,!?\"':;~()]")

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
max_sequences = 30
embedding_dim = 100
lstm_hidden_dim = 128
# 인덱스 파일 로드
inv_list = [1581,
17110,
1598,
16642,
7189,
6264,
2879,
13936
]
with open('./model/word_to_index.pkl', 'rb') as f:
    word_to_index = pickle.load(f)
with open('./model/index_to_word.pkl', 'rb') as f:
    index_to_word = pickle.load(f)
def pos_tag(sentences):
    sentence_pos = []

    for sentence in sentences:
        sentence = re.sub(RE_FILTER, "", sentence)
        # sentence = okt.morphs(sentence)
        sentence = " ".join(okt.morphs(sentence))
        sentence_pos.append(sentence)
    return sentence_pos
def convert_index_to_text(indexs, vocabulary): 
    
    sentence = ''
    # 모든 문장에 대해서 반복
    for index in indexs:
        inv=False
        for i in inv_list:
            if index == i:
                inv=True
        if not inv:
            sentence += ' '
        if index == END_INDEX:
            # 종료 인덱스면 중지
            break;
        elif vocabulary.get(index) is not None:
            # 사전에 있는 인덱스면 해당 단어를 추가
            sentence += vocabulary[index]
        else:
            # 사전에 없는 인덱스면 OOV 단어를 추가
            sentence += vocabulary[OOV_INDEX]
            
        # 빈칸 추가


    return sentence
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

def make_predict_input(sentence):
    sentences = []
    sentences.append(sentence)
    sentences = pos_tag(sentences)
    input_seq = convert_text_to_index(sentences, word_to_index, ENCODER_INPUT)
    
    return input_seq

def generate_text(input_seq):
    
    # 입력을 인코더에 넣어 마지막 상태 구함
    states = encoder_model.predict(input_seq)
    # 목표 시퀀스 초기화
    target_seq = np.zeros((1, 1))
    
    # 목표 시퀀스의 첫 번째에 <START> 태그 추가
    target_seq[0, 0] = STA_INDEX
    
    # 인덱스 초기화
    indexs = []
    
    # 디코더 타임 스텝 반복
    while 1:
        # 디코더로 현재 타임 스텝 출력 구함
        # 처음에는 인코더 상태를, 다음부터 이전 디코더 상태로 초기화
        decoder_outputs, state_h, state_c = decoder_model.predict(
                                                [target_seq] + states)
        # 결과의 원핫인코딩 형식을 인덱스로 변환
        index_in = np.argsort(-decoder_outputs[0, 0, :])
        index = random.choice(index_in[:3])
        indexs.append(index)
    
        # 종료 검사
        if index == END_INDEX or len(indexs) >= max_sequences:
            break

        # 목표 시퀀스를 바로 이전의 출력으로 설정
        target_seq = np.zeros((1, 1))
        target_seq[0, 0] = index
        
        # 디코더의 이전 상태를 다음 디코더 예측에 사용
        states = [state_h, state_c]

    # 인덱스를 문장으로 변환
    sentence = convert_index_to_text(indexs, index_to_word)
        
    return sentence

while True:
    input_seq = make_predict_input(input())
    print(generate_text(input_seq))