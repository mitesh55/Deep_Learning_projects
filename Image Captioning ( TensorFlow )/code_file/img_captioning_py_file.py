import os
# import tensorflow.keras
import tensorflow as tf
# import tensorflow.keras.backend as K
# import matplotlib.pyplot as plt
import sys
import pickle
# from tensorflow.keras.utils import plot_model
from tensorflow.keras.applications.xception import Xception
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications.xception import preprocess_input
from tensorflow.keras.models import Model, load_model
import string
import numpy as np
from pickle import load
from tensorflow.keras.preprocessing.sequence import pad_sequences
# from tensorflow.keras.utils import to_categorical
# from tensorflow.keras.utils import plot_model
# from tensorflow.keras.models import Model
# from tensorflow.keras.layers import Input
# from tensorflow.keras.layers import Dense
# from tensorflow.keras.layers import LSTM ,GRU
# from tensorflow.keras.layers import Embedding
# from tensorflow.keras.layers import Dropout, Reshape, Lambda, Concatenate
# from tensorflow.keras.layers.merge import add
# from tensorflow.keras.layers import add
# from tensorflow.keras.callbacks import ModelCheckpoint
# from tensorflow.keras import optimizers
# from nltk.translate.bleu_score import corpus_bleu

# with open('train_descriptions.pkl','rb') as f:
#     train_descriptions= pickle.load(f)
# with open('test_descriptions.pkl','rb') as f:
#     test_descriptions= pickle.load(f)
# with open('dev_descriptions.pkl','rb') as f:
#     dev_descriptions= pickle.load(f)
with open('../necessary_files/vocab.pkl','rb') as f:
    vocab= pickle.load(f)
with open('../necessary_files/wordtoix.pkl','rb') as f:
    wordtoix= pickle.load(f)
with open('../necessary_files/ixtoword.pkl','rb') as f:
    ixtoword= pickle.load(f)
# with open('dev_features.pkl','rb') as f:
#     dev_features= pickle.load(f)
# with open('train_features.pkl','rb') as f:
#     train_features= pickle.load(f)
# with open('test_features.pkl','rb') as f:
#     test_features= pickle.load(f)
# with open('E:\img_captioning_reference/final_weights/embedding_matrix.pkl','rb') as f:
#     embedding_matrix= pickle.load(f)
with open('../necessary_files/descriptions.pkl','rb') as f:
    descriptions= pickle.load(f)

max_length = 18
start_token = '<startseq>'
end_token = '<endseq>'
oov_token = '<UNK>'
filters = '!"#$%&()*+,-./:;=?@[\\]^_`{|}~\t\n' # making sure all the last non digit non alphabet chars are removed
tokenizer = tf.keras.preprocessing.text.Tokenizer(filters = filters, oov_token=oov_token)
tokenizer.fit_on_texts(vocab)
vocab_size = len(tokenizer.word_index) + 1
# print('vocab_size :', vocab_size)


# generate a description for an image greedy way
def generate_desc(model, photo_fe, inference=False):
    # seed the generation process
    in_text = start_token
    # iterate over the whole length of the sequence
    # generate one word at each iteratoin of the loop
    # appends the new word to a list and makes the whole sentence
    for i in range(max_length):
        # integer encode input sequence
        sequence = tokenizer.texts_to_sequences(
            in_text.split())  # [wordtoix[w] for w in in_text.split() if w in wordtoix]
        # pad input
        photo_fe = photo_fe.reshape((1, 2048))
        sequence = pad_sequences([sequence], maxlen=max_length).reshape((1, max_length))
        # predict next word
        yhat = model.predict([photo_fe, sequence], verbose=0)
        # convert probability to integer
        yhat = np.argmax(yhat)
        # map integer to word
        word = ixtoword[yhat]
        # stop if we cannot map the word
        if word is None:
            break
        # append as input for generating the next v
        in_text += ' ' + word
        # stop if we predict the end of the sequence
        if word == end_token:
            break

    if inference == True:
        in_text = in_text.split()
        if len(in_text) == max_length:
            in_text = in_text[1:]  # if it is already at max len and endseq hasn't appeared
        else:
            in_text = in_text[1:-1]
        in_text = ' '.join(in_text)

    return in_text


def beam_search_pred(model, pic_fe, wordtoix, K_beams=3, log=False):
    start = [wordtoix[start_token]]

    start_word = [[start, 0.0]]

    while len(start_word[0][0]) < max_length:
        temp = []
        for s in start_word:
            sequence = pad_sequences([s[0]], maxlen=max_length).reshape(
                (1, max_length))  # sequence of most probable words
            # based on the previous steps
            preds = model.predict([pic_fe.reshape(1, 2048), sequence])
            word_preds = np.argsort(preds[0])[
                         -K_beams:]  # sort predictions based on the probability, then take the last
            # K_beams items. words with the most probs
            # Getting the top <K_beams>(n) predictions and creating a
            # new list so as to put them via the model again
            for w in word_preds:

                next_cap, prob = s[0][:], s[1]
                next_cap.append(w)
                if log:
                    prob += np.log(preds[0][w])  # assign a probability to each K words4
                else:
                    prob += preds[0][w]
                temp.append([next_cap, prob])
        start_word = temp
        # Sorting according to the probabilities
        start_word = sorted(start_word, reverse=False, key=lambda l: l[1])

        # Getting the top words
        start_word = start_word[-K_beams:]

    start_word = start_word[-1][0]
    captions_ = [ixtoword[i] for i in start_word]

    final_caption = []

    for i in captions_:
        if i != end_token:
            final_caption.append(i)
        else:
            break

    final_caption = ' '.join(final_caption[1:])
    return final_caption



from tensorflow.keras.models import model_from_json
json_file = open('../necessary_files/model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
#
# print("+++++++++++++++++++++++++++++++++++++++++++++++++++++++")
loaded_model = model_from_json(loaded_model_json)
#loaded_model = load_model("model_json.h5")
# print("-------------------------------------------------")
# load weights into new model
loaded_model.load_weights("../necessary_files/model_json.h5")
#json_file.close()
# for masking pad0 outputs
# def masked_loss_function(real, pred):
#     mask = tf.math.logical_not(tf.math.equal(real, 0))
#     loss_ = K.sparse_categorical_crossentropy(real, pred, from_logits= False) # sparse cat gets pred classes in 'int' form
#     mask = tf.cast(mask, dtype=loss_.dtype)
#     loss_ *= mask
#     return tf.reduce_mean(loss_)
# loaded_model = tf.keras.models.load_model('E:\img_captioning_reference/final_weights/my_model.h5', custom_objects={'masked_loss_function':masked_loss_function})
# print("Loaded model from disk")

# pic = list(test_features.keys())[np.random.randint(1,1000)]
# fe = test_features[pic].reshape((1,2048))
# x=plt.imread('E:\img_captioning_reference\dataset\Flickr8k_Dataset\Flicker8k_Dataset/'+pic+'.jpg')
# plt.imshow(x)
# plt.show()
# print("Greedy:",generate_desc(model=loaded_model, photo_fe=fe, inference=True))
# print("Beam K= 3:",beam_search_pred(model=loaded_model, pic_fe=test_features[pic], wordtoix=wordtoix, K_beams = 3, log=False))
# print("Beam K= 5:",beam_search_pred(model=loaded_model, pic_fe=test_features[pic], wordtoix=wordtoix, K_beams = 5, log=False))
# print("Beam log K= 7:",beam_search_pred(model=loaded_model, pic_fe=test_features[pic], wordtoix=wordtoix, K_beams = 7, log=False))
# print("Beam log K= 3:",beam_search_pred(model=loaded_model, pic_fe=test_features[pic], wordtoix=wordtoix, K_beams = 3, log=True))
# print("Beam log K= 5:",beam_search_pred(model=loaded_model, pic_fe=test_features[pic], wordtoix=wordtoix, K_beams = 5, log=True))
# print("Beam log K= 7:",beam_search_pred(model=loaded_model, pic_fe=test_features[pic], wordtoix=wordtoix, K_beams = 7, log=True))

xception = Xception()
extractor = Model(inputs=xception.inputs, outputs=xception.layers[-2].output) # removing 2 last fully connected layers
