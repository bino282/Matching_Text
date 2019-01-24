import json
from utils.utils import *
from model import biMPM,lstm_cnn,lstm_cnn_att_sub
from keras import optimizers
import keras.backend as K
from keras.callbacks import ModelCheckpoint,EarlyStopping
import pickle
path_dev = './data/test/SemEval2016-Task3-CQA-QL-test-subtaskA.xml.subtaskA.relevancy'
path_test= './data/test/SemEval2017-Task3-CQA-QL-test-subtaskA.xml.subtaskA.relevancy'
path_embeding = '../../local/word_vector/gensim_glove_vectors_300d.txt'
config = json.load(open('config.json', 'r'))
dataPath = config['TRAIN']['path']
fileList = config['TRAIN']['files']
data_train = constructData(dataPath, fileList)
dataPath = config['DEV']['path']
fileList = config['DEV']['files']
data_dev = constructData(dataPath, fileList,mode='DEV',path_dev=path_dev)
dataPath = config['TEST']['path']
fileList = config['TEST']['files']
data_test = constructData(dataPath, fileList,mode='DEV',path_dev=path_test)

s1s_train,s2s_train,subj_train,users_train,labels_train,cat_train = read_constructData(data_train)

vocab, voc2index, index2voc = creat_vocab(s1s_train+s2s_train+subj_train)
with open('voc2index.pkl','wb') as fw:
    pickle.dump(voc2index,fw)
print('vocab_size: ',len(vocab))
embed_matrix = read_embed(path_embeding,embed_size=300,vocab=vocab)
max_len_q = 100
max_len_a = 100
max_len_s = 100
seq1_input = convertData_model(s1s_train,voc2index,max_len=max_len_q)
seq2_input = convertData_model(s2s_train,voc2index,max_len=max_len_a)
subj_input = convertData_model(subj_train,voc2index,max_len = max_len_s)
s1s_len_train = [len(s.split()) for s in s1s_train]
s2s_len_train = [len(s.split()) for s in s2s_train]
s1s_len_train = np.asarray(s1s_len_train)
s2s_len_train = np.asarray(s2s_len_train)
labels_train = np.asarray(labels_train)

s1s_dev,s2s_dev,subj_dev,users_dev,labels_dev,cat_dev = read_constructData(data_dev)
seq1_input_dev = convertData_model(s1s_dev,voc2index,max_len=max_len_q)
seq2_input_dev = convertData_model(s2s_dev,voc2index,max_len=max_len_a)
subj_input_dev = convertData_model(subj_dev,voc2index,max_len=max_len_s)
s1s_len_dev = [len(s.split()) for s in s1s_dev]
s2s_len_dev = [len(s.split()) for s in s2s_dev]
s1s_len_dev = np.asarray(s1s_len_dev)
s2s_len_dev = np.asarray(s2s_len_dev)
labels_dev = np.asarray(labels_dev)

s1s_test,s2s_test,subj_test,users_test,labels_test,cat_test = read_constructData(data_test)
seq1_input_test = convertData_model(s1s_test,voc2index,max_len=max_len_q)
seq2_input_test = convertData_model(s2s_test,voc2index,max_len=max_len_a)
subj_input_test = convertData_model(subj_test,voc2index,max_len=max_len_s)
s1s_len_test = [len(s.split()) for s in s1s_test]
s2s_len_test = [len(s.split()) for s in s2s_test]
s1s_len_test = np.asarray(s1s_len_test)
s2s_len_test = np.asarray(s2s_len_test)
labels_test = np.asarray(labels_test)

print(seq1_input.shape)
model_config={'seq1_maxlen':max_len_q,'seq2_maxlen':max_len_a,'seq3_maxlen':max_len_s,
                'vocab_size':len(voc2index)+1,'embed_size':300,
                'hidden_size':128,'dropout_rate':0.5,
                'embed':embed_matrix,
                'embed_trainable':True,
                'channel':5,
                'aggre_size':100,
                'target_mode':'ranking'}
def ranknet(y_true, y_pred):
    return K.mean(K.log(1. + K.exp(-(y_true * y_pred - (1-y_true) * y_pred))), axis=-1)

model_lstm = biMPM.BiMPM(config=model_config).model
print(model_lstm.summary())
optimize = optimizers.Adam(lr=0.0001)
model_lstm.compile(loss='binary_crossentropy',optimizer=optimize,metrics=['accuracy'])
checkpoint = ModelCheckpoint("./model_saved/model-lstm-cnn-{epoch:02d}-{val_acc:.2f}.h5", monitor='val_loss', verbose=1, save_best_only=True)
early_stop = EarlyStopping(monitor='val_loss',min_delta=0.0001,patience=3)


MAP_last = 0
for epoch in range(150):
    print('Train on iteration {}'.format(epoch))
    model_lstm.fit([seq1_input,seq2_input,s1s_len_train,s2s_len_train],labels_train,batch_size=128,epochs=1,
                validation_data=([seq1_input_dev,seq2_input_dev,s1s_len_dev,s2s_len_dev],labels_dev))
    y_pred = model_lstm.predict([seq1_input_dev,seq2_input_dev,s1s_len_dev,s2s_len_dev])
    MAP_dev,MRR_dev = map_score(s1s_dev,s2s_dev,y_pred,labels_dev)
    print('MAP_dev = {}, MRR_dev = {}'.format(MAP_dev,MRR_dev))
    if(MAP_dev>MAP_last):
        model_lstm.save('./model_saved/model-lstm-cnn.h5')
        print('Model saved !')
        MAP_last = MAP_dev
    y_test = model_lstm.predict([seq1_input_test,seq2_input_test,s1s_len_test,s2s_len_test])
    MAP_test,MRR_test = map_score(s1s_test,s2s_test,y_test,labels_test)
    print('MAP_test = {}, MRR_test = {}'.format(MAP_test,MRR_test))