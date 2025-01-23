# coding: utf-8
import sys
sys.path.append('..')
import numpy as np
from common.time_layers import *


class SimpleRnnlm:
    def __init__(self, vocab_size, wordvec_size, hidden_size):
        V, D, H = vocab_size, wordvec_size, hidden_size 
        #Vは単語数 入力はid形式で入る Wの特定の行は特定の単語に相当する．（Vの数は想定されるすべての単語数）
        rn = np.random.randn 
        #重みの初期化
        embed_W = (rn(V, D) / 100).astype('f')
        rnn_Wx = (rn(D, H) / np.sqrt(D)).astype('f') #初期値の設定は前の層のノード数で決めると良い
        rnn_Wh = (rn(H, H) / np.sqrt(H)).astype('f')
        rnn_b = np.zeros(H).astype('f') #各隠れ層のバイアスは0で初期化
        affine_W = (rn(H, V) / np.sqrt(H)).astype('f') #正解ラベル（id形式）と比較するので，Affine変換後の次元はV
        affine_b = np.zeros(V).astype('f')
        
        #レイヤの作成
        self.layers = [
            TimeEmbedding(embed_W), 
            TimeRNN(rnn_Wx, rnn_Wh, rnn_b, stateful = True),  #右の層へhが伝播する
            TimeAffine(affine_W, affine_b)
        ]
        self.loss_layer =  TimeSoftmaxWithLoss() #損失関数
        self.rnn_layer = self.layers[1]
        #パラメータ，重みをまとめる
        self.params, self.grads = [], []
        for layer in self.layers:
            self.params += layer.params
            self.grads += layer.grads
        
        
        
    def forward(self, xs, ts):
        for layer in self.layers:
            xs = layer.forward(xs)
        loss = self.loss_layer.forward(xs, ts)
        return loss
        

    def backward(self, dout=1):
        dout = self.loss_layer.backward(dout)
        for layer in reversed(self.layers):
            dout = layer.backward(dout)
        return dout
        

    def reset_state(self):
        self.rnn_layer.reset_state()
