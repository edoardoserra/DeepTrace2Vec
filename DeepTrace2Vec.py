import gc
import sys
import os
import pandas as pd
import numpy as np
import scipy.stats
import numpy as np
import pandas as pd
from keras import layers
from keras.preprocessing.text import one_hot
from keras.models import Model,Sequential
from keras.layers import LSTM,Permute,Flatten,Multiply,RepeatVector, Lambda, Reshape, Activation, Dense, Dropout, Input, Embedding, Bidirectional
from keras.layers import Bidirectional, GlobalMaxPool1D
from keras.optimizers import RMSprop, Adam
from keras.preprocessing.text import Tokenizer
from keras.preprocessing import sequence
from keras.utils import to_categorical
from keras.callbacks import EarlyStopping
from sklearn.preprocessing import LabelEncoder
from keras.preprocessing.sequence import pad_sequences
from keras import backend as K
import tensorflow as tf

from sklearn.preprocessing import StandardScaler


from sklearn.base import BaseEstimator, ClassifierMixin
    
class MSLSTM(BaseEstimator, ClassifierMixin):  

    def __init__(self, batch_size = 500, epochs = 100):
        self.BATCH_SIZE=batch_size
        self.EPOCHS=epochs
        self.ss='/'
        self.ss1=':'

    def paddedSeq(self,tok,a):
        sequences = tok.texts_to_sequences(a)
        max_len= np.max([len(sequences[i]) for i in range(len(sequences))])
        return sequence.pad_sequences(sequences,maxlen=max_len)

    def createKerasModel(self):
        li=[]
        lo=[]
        dicim={}
        if self.l!='':
            inputs = Input(shape=(None, self.maxs1))
            dicim[self.l]=inputs
            li.append(inputs)
            lo.append(inputs)
        for l in self.LAV:
            inpu=layers.Input(shape=(None,), dtype='float32')
            dicim[l]=inpu
            seq=Reshape((-1,1))(inpu)
            li.append(inpu)
            lo.append(seq)
        for i in range(len(self.LAN)):
            (l,ES)=self.LAN[i]
            tok = self.LTAN[i]
            vocab_sizet=len(tok.word_counts)+1
            text = layers.Input(shape=(None,), dtype='int32')
            dicim[l]=text
            li.append(text)
            encoded_text = layers.Embedding(vocab_sizet, ES)(text)
            lo.append(encoded_text)
        #newInput = layers.concatenate(lo, axis=2)
        if len(lo)>1:
            newInput = layers.concatenate(lo, axis=2)
        else:
            newInput=lo[0]
        #newInput = layers.ReLU()(newInput)
        LSTM_EMBEDDING_SIZE = int(newInput.shape[2])
        #LSTM(LSTM_EMBEDDING_SIZE,return_sequences=True)(newInput)
        newInput=LSTM(LSTM_EMBEDDING_SIZE,return_sequences=True)(newInput)
        def scores(arg):
            sh=K.shape(arg)
            w = K.random_normal_variable((LSTM_EMBEDDING_SIZE,LSTM_EMBEDDING_SIZE),mean=0.0,scale=1.0)
            val=K.dot(arg,w)
            last=arg[:,sh[1]-1,:]
            tot=K.batch_dot(last,val, axes=(1,2))
            return tot
        def concatenate(arg):
            sh=K.shape(arg[0])
            last=arg[0][:,sh[1]-1,:]
            return K.concatenate([arg[1], last], axis=1)
        attention = Lambda(scores)(newInput)
        attentionLayer = Activation('softmax')(attention)
        attention = RepeatVector(LSTM_EMBEDDING_SIZE)(attentionLayer)
        attention = Permute([2, 1])(attention)
        sent_representation = Multiply()([newInput, attention])
        sent_representation = Lambda(lambda xin: K.sum(xin, axis=-2), output_shape=(LSTM_EMBEDDING_SIZE,))(sent_representation)
        finaloutput = Lambda(concatenate)([newInput,sent_representation])#print(last)
        lis=[finaloutput]
        for l in self.AV:
            inpu=layers.Input(shape=(1,), dtype='float32')
            dicim[l]=inpu
            li.append(inpu)
            lis.append(inpu)
            #print(inpu)
        for i in range(len(self.AN)):
            (l,ES)=self.AN[i]
            le=self.LP[i]
            text = layers.Input(shape=(1,), dtype='int32')
            dicim[l]=text
            li.append(text)
            vocab_sizet=len(le.classes_)+1
            encoded_text = layers.Embedding(vocab_sizet, ES)(text)
            #encoded_text=Permute([2,1])(encoded_text)
            encoded_text=Reshape(target_shape=(ES,))(encoded_text)
            #print(encoded_text)
            lis.append(encoded_text)
            #print(encoded_text)
        #newInput = layers.concatenate(lis, axis=1)
        if len(lis)>1:
            newInput = layers.concatenate(lis, axis=1)
        else:
            newInput=lis[0]
        #print(newInput)
        shlayer=layers.Dense(self.embSize,activation='tanh')(newInput)
        out=[]
        out10=[]
        losses=[]
        ino=[]
        def squ(aaa):
            f=K.shape(aaa)
            d=K.reshape(aaa,(f[0],f[1]))
            #print(d.shape)
            return d
        #'sparse_categorical_crossentropy'mean_squared_error
        def repeat_vector(args):
            layer_to_repeat = args[0]
            reference_layer = args[1]
            return RepeatVector(K.shape(reference_layer)[1])(layer_to_repeat)
        def func(ccc):
            sss= K.sum(tf.losses.mean_squared_error(ccc[0],ccc[1],reduction=tf.losses.Reduction.NONE),axis=1)
            #print('ddddddd',sss.shape)
            return K.reshape(sss,(K.shape(sss)[0],1))
        def fund(ccc):
            sss= K.sum(tf.losses.sparse_softmax_cross_entropy(ccc[0],ccc[1],reduction=tf.losses.Reduction.NONE),axis=1)
            #print('ddddddd',sss.shape)
            oo=K.reshape(sss,(K.shape(sss)[0],1))  
            #print('ddddddd',oo.shape)
            return oo
        def fund1(ccc):
            sss= tf.losses.sparse_softmax_cross_entropy(ccc[0],ccc[1],reduction=tf.losses.Reduction.NONE)
            #print('ddddddd',sss.shape)
            oo=K.reshape(sss,(K.shape(sss)[0],1))  
            #print('ddddddd',oo.shape)
            return oo
        if self.l!='' and self.Yl:
            losses.append('mean_squared_error')
            shlayer1=Dense(self.embSize1,activation='tanh')(shlayer)
            out10.append(shlayer1)
            rhlayer = Lambda(repeat_vector) ([shlayer1, dicim[self.l]])
            decoded = LSTM(self.embSize1, return_sequences=True)(rhlayer)
            aaa=Dense(self.maxs1)(decoded)
            bbb=Lambda(func,output_shape=(1,))([dicim[self.l],aaa])
            ino.append(bbb)
            out.append(aaa)
        for l in self.LAV:
            if l in self.YLAV:
                shlayer1=Dense(self.embSize1,activation='tanh')(shlayer)
                out10.append(shlayer1)
                losses.append('mean_squared_error')
                rhlayer = Lambda(repeat_vector,output_shape=(None,self.embSize1)) ([shlayer1, dicim[l]])
                decoded = LSTM(self.embSize1, return_sequences=True)(rhlayer)
                aaa=Dense(1)(decoded)
                #print('ginor ',aaa.shape)
                finaloutput = Lambda(squ,output_shape=(None))(aaa)
                #print('ginor ',finaloutput.shape)
                bbb=Lambda(func,output_shape=(1,))([dicim[l],finaloutput])
                ino.append(bbb)
                out.append(finaloutput)
        for i in range(len(self.LAN)):
            (l,ES)=self.LAN[i]
            print('a')
            print(l)
            print(self.YLAN)
            if l in self.YLAN:
                print('b')
                shlayer1=Dense(self.embSize1,activation='tanh')(shlayer)
                out10.append(shlayer1)
                tok = self.LTAN[i]
                vocab_sizet=len(tok.word_counts)+1
                losses.append('sparse_categorical_crossentropy')
                rhlayer = Lambda(repeat_vector,output_shape=(None,self.embSize1)) ([shlayer1, dicim[l]])
                decoded = LSTM(self.embSize1, return_sequences=True)(rhlayer)
                aaa1=Dense(vocab_sizet,activation='softmax')(decoded)
                aaa=Activation('softmax')(aaa1)
                #print('ginor ',aaa.shape)
                #ino.append(K.sparse_categorical_crossentropy(dicim[l], aaa, from_logits=True, axis=0))
                bbb=Lambda(fund,output_shape=(1,))([dicim[l],aaa1])
                ino.append(bbb)
                out.append(aaa)
                print(bbb)
        for l in self.AV:
            if l in self.YAV:
                losses.append('mean_squared_error')
                shlayer1=Dense(self.embSize1,activation='tanh')(shlayer)
                out10.append(shlayer1)
                aaa=Dense(1)(shlayer1)
                bbb=Lambda(func,output_shape=(1,))([dicim[l],aaa])
                ino.append(bbb)
                out.append(aaa)
        for i in range(len(self.AN)):
            (l,ES)=self.AN[i]
            #print(self.YAN)
            #print(l)
            if l in self.YAN:
                tok = self.LP[i]
                vocab_sizet=len(tok.classes_)
                #print(l)
                losses.append('sparse_categorical_crossentropy')
                shlayer1=Dense(self.embSize1,activation='tanh')(shlayer)
                out10.append(shlayer1)
                aaa1=Dense(vocab_sizet)(shlayer1)
                aaa=Activation('softmax')(aaa1)
                #ino.append(K.sparse_categorical_crossentropy(dicim[l], aaa, from_logits=True, axis=0))
                bbb=Lambda(fund1,output_shape=(1,))([dicim[l],aaa1])
                ino.append(bbb)
                out.append(aaa)
        m=Model(li, out)
        self.predMpdel=Model(li,shlayer)
        
        print(len(ino))
        
        def daa(lll):
            t=K.concatenate(lll, axis=1)
            f=K.sum(t,axis=1)
            f=K.reshape(f,(K.shape(f)[0],1))
            return K.concatenate([f,t], axis=1)
        if len(ino)>1:
            oberon=Lambda(daa,output_shape=(len(self.attributes),))(ino)
        else:
            oberon=ino[0]
        #print(oberon.shape)
        self.mscore=Model(li,oberon)
        self.mscoreAttention=Model(li,[oberon,attentionLayer])
        self.mmemb=Model(li,out10)
        #print(losses)
        m.compile(optimizer='adam',loss=losses)
        return m


    def fit(self, X,embSize=100, embSize1=30,ll='',LAN=[],LAV=[],AN=[],AV=[],YLAN=set(),YLAV=set(),YAN=set(),YAV=set(),Yll=False,class_weight=None,sample_weight=None):
        data=X
        LI=[]
        LO=[]
        self.embSize=embSize
        self.embSize1=embSize1
        self.LAN=LAN
        self.LAV=LAV
        self.AN=AN
        self.AV=AV
        self.YLAN=YLAN
        self.YLAV=YLAV
        self.YAN=YAN
        self.YAV=YAV
        self.LTAN=[]
        self.LP=[]
        self.l=ll
        self.Yl=Yll
        self.scalersl=[]
        self.scalers=[]
        self.attributes=['TotalScore']
        if self.l!='':
            VAL=X[self.l].map(lambda x: [[float(a) for a in b.split(self.ss1) ]for b in x.split(self.ss) if b!='']).values
            self.maxs=np.max([len(x) for x in VAL])
            self.maxs1=len(VAL[0][0]) 
            aa=pad_sequences(VAL,self.maxs)
            LI.append(aa)
            if Yll:
                LO.append(aa)
                self.attributes.append('a( '+self.l+' )')
        for l in self.LAV:
            val=data[l].map(lambda x: [float(a) for a in x.split(self.ss)]).values
            maxs=np.max([len(x) for x in val])
            vall=np.hstack(np.array([x for x in val]))
            mean1=vall.mean()
            std1=vall.std()
            val=data[l].map(lambda x: [(float(a)-mean1)/std1 for a in x.split(self.ss)]).values
            self.scalersl.append((mean1,std1))
            aa=pad_sequences(val,maxs)
            LI.append(aa)
            if l in YLAV:
                LO.append(aa.astype(float))
                self.attributes.append('a( '+l+' )')
        for (l,_) in self.LAN:
            tok = Tokenizer()
            self.LTAN.append(tok)
            seq=data[l].map(lambda x: x.replace(self.ss,' ')).values
            tok.fit_on_texts(seq)
            aa=self.paddedSeq(tok,seq)
            LI.append(aa)
            if l in YLAN:
                LO.append(aa.reshape(aa.shape[0],aa.shape[1],1))
                self.attributes.append('a( '+l+' )')
        for l in self.AV:
            f=data[l].values
            scaler = StandardScaler()
            self.scalers.append(scaler)
            vv=scaler.fit_transform(f.reshape(f.shape[0],1))
            #print(l,vv.shape)
            LI.append(vv)
            if l in YAV:
                LO.append(vv.astype(float))
                self.attributes.append('a( '+l+' )')
        for (l,_) in self.AN:
            le = LabelEncoder()
            self.LP.append(le)
            f=le.fit_transform(data[l].values)
            aa=f.reshape(f.shape[0],1)
            LI.append(aa)
            if l in YAN:
                LO.append(aa)
                self.attributes.append('a( '+l+' )')
        self.model=self.createKerasModel()
        #print(LI)
        #print(LO)
        self.model.fit(LI,LO,batch_size=self.BATCH_SIZE,epochs=self.EPOCHS,class_weight=class_weight,sample_weight=sample_weight)
        return self


    def predict(self, X, y=None):
        #return embedding layer
        data=X
        LI=[]
        if self.l!='':
            VAL=X[self.l].map(lambda x: [[float(a) for a in b.split(self.ss1) ]for b in x.split(self.ss) if b!='']).values
            LI.append(pad_sequences(VAL,self.maxs))
        for loo in range(len(self.LAV)):
            l=self.LAV[loo]
            (mean1,std1)=self.scalersl[loo]
            val=data[l].map(lambda x: [(float(a)-mean1)/std1 for a in x.split(self.ss)]).values
            maxs=np.max([len(x) for x in val])
            vv=pad_sequences(val,maxs)
            LI.append(vv)
#            for l in self.LAV:
#                val=data[l].map(lambda x: [float(a) for a in x.split(self.ss)]).values
#                maxs=np.max([len(x) for x in val])
#                LI.append(pad_sequences(val,maxs))
        for l in range(len(self.LAN)):
            tok = self.LTAN[l]
            seq=data[self.LAN[l][0]].map(lambda x: x.replace(self.ss,' ')).values
            LI.append(self.paddedSeq(tok,seq))
        for loo in range(len(self.AV)):
            l=self.AV[loo]
            scaler=self.scalers[loo]
            f=data[l].values
            LI.append(scaler.transform(f.reshape(f.shape[0],1)))
#            for l in self.AV:
#                f=data[l].values
#                LI.append(f.reshape(f.shape[0],1))
        for l in range(len(self.AN)):
            le=self.LP[l]
            f=le.transform(data[self.AN[l][0]].values)
            LI.append(f.reshape(f.shape[0],1))
        return self.predMpdel.predict(LI)

    def score(self, X, y=None):
        data=X
        LI=[]
        if self.l!='':
            VAL=X[self.l].map(lambda x: [[float(a) for a in b.split(self.ss1) ]for b in x.split(self.ss) if b!='']).values
            LI.append(pad_sequences(VAL,self.maxs))
        for loo in range(len(self.LAV)):
            l=self.LAV[loo]
            (mean1,std1)=self.scalersl[loo]
            val=data[l].map(lambda x: [(float(a)-mean1)/std1 for a in x.split(self.ss)]).values
            maxs=np.max([len(x) for x in val])
            vv=pad_sequences(val,maxs)
            LI.append(vv)
        for l in range(len(self.LAN)):
            tok = self.LTAN[l]
            seq=data[self.LAN[l][0]].map(lambda x: x.replace(self.ss,' ')).values
            LI.append(self.paddedSeq(tok,seq))
        for loo in range(len(self.AV)):
            l=self.AV[loo]
            scaler=self.scalers[loo]
            f=data[l].values
            LI.append(scaler.transform(f.reshape(f.shape[0],1)))
        for l in range(len(self.AN)):
            le=self.LP[l]
            f=le.transform(data[self.AN[l][0]].values)
            LI.append(f.reshape(f.shape[0],1))
        return self.mscore.predict(LI)[:,0]

    def scores(self, X, y=None):
        data=X
        LI=[]
        if self.l!='':
            VAL=X[self.l].map(lambda x: [[float(a) for a in b.split(self.ss1) ]for b in x.split(self.ss) if b!='']).values
            LI.append(pad_sequences(VAL,self.maxs))
        for loo in range(len(self.LAV)):
            l=self.LAV[loo]
            (mean1,std1)=self.scalersl[loo]
            val=data[l].map(lambda x: [(float(a)-mean1)/std1 for a in x.split(self.ss)]).values
            maxs=np.max([len(x) for x in val])
            vv=pad_sequences(val,maxs)
            LI.append(vv)
        for l in range(len(self.LAN)):
            tok = self.LTAN[l]
            seq=data[self.LAN[l][0]].map(lambda x: x.replace(self.ss,' ')).values
            LI.append(self.paddedSeq(tok,seq))
        for loo in range(len(self.AV)):
            l=self.AV[loo]
            scaler=self.scalers[loo]
            f=data[l].values
            LI.append(scaler.transform(f.reshape(f.shape[0],1)))
        for l in range(len(self.AN)):
            le=self.LP[l]
            f=le.transform(data[self.AN[l][0]].values)
            LI.append(f.reshape(f.shape[0],1))
        return (self.attributes,self.mscore.predict(LI))

    def predictMultipleEmb(self, X, y=None):
        data=X
        LI=[]
        if self.l!='':
            VAL=X[self.l].map(lambda x: [[float(a) for a in b.split(self.ss1) ]for b in x.split(self.ss) if b!='']).values
            LI.append(pad_sequences(VAL,self.maxs))
        for loo in range(len(self.LAV)):
            l=self.LAV[loo]
            (mean1,std1)=self.scalersl[loo]
            val=data[l].map(lambda x: [(float(a)-mean1)/std1 for a in x.split(self.ss)]).values
            maxs=np.max([len(x) for x in val])
            vv=pad_sequences(val,maxs)
            LI.append(vv)
        for l in range(len(self.LAN)):
            tok = self.LTAN[l]
            seq=data[self.LAN[l][0]].map(lambda x: x.replace(self.ss,' ')).values
            LI.append(self.paddedSeq(tok,seq))
        for loo in range(len(self.AV)):
            l=self.AV[loo]
            scaler=self.scalers[loo]
            f=data[l].values
            LI.append(scaler.transform(f.reshape(f.shape[0],1)))
        for l in range(len(self.AN)):
            le=self.LP[l]
            f=le.transform(data[self.AN[l][0]].values)
            LI.append(f.reshape(f.shape[0],1))
        return (self.attributes,self.mmemb.predict(LI))

    def scoresAttention(self, X, y=None):
        data=X
        LI=[]
        if self.l!='':
            VAL=X[self.l].map(lambda x: [[float(a) for a in b.split(self.ss1) ]for b in x.split(self.ss) if b!='']).values
            LI.append(pad_sequences(VAL,self.maxs))
        for loo in range(len(self.LAV)):
            l=self.LAV[loo]
            (mean1,std1)=self.scalersl[loo]
            val=data[l].map(lambda x: [(float(a)-mean1)/std1 for a in x.split(self.ss)]).values
            maxs=np.max([len(x) for x in val])
            vv=pad_sequences(val,maxs)
            LI.append(vv)
        for l in range(len(self.LAN)):
            tok = self.LTAN[l]
            seq=data[self.LAN[l][0]].map(lambda x: x.replace(self.ss,' ')).values
            LI.append(self.paddedSeq(tok,seq))
        for loo in range(len(self.AV)):
            l=self.AV[loo]
            scaler=self.scalers[loo]
            f=data[l].values
            LI.append(scaler.transform(f.reshape(f.shape[0],1)))
        for l in range(len(self.AN)):
            le=self.LP[l]
            f=le.transform(data[self.AN[l][0]].values)
            LI.append(f.reshape(f.shape[0],1))
        #print(LI)
        ss=self.mscoreAttention.predict(LI)
        return (self.attributes,ss[0],ss[1])

    def close(self):
        del self.model
        del self.mscore
        del self.predMpdel
        K.clear_session()
        gc.collect()
        return self
