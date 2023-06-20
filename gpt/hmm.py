import pickle
with open('./model/word_to_index(1).pkl', 'rb') as f:
    word_to_index = pickle.load(f)
with open('./model/index_to_word(1).pkl', 'rb') as f:
    index_to_word = pickle.load(f)
print(word_to_index["이"])
print(word_to_index["가"])
print(word_to_index["은"])
print(word_to_index["는"])
print(word_to_index["의"])
print(word_to_index["에"])
print(word_to_index["에서"])
print(word_to_index["한테"])