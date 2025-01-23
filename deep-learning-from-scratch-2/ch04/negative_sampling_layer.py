# coding: utf-8
import sys
sys.path.append('..')
from common.np import *  # import numpy as np
from common.layers import Embedding, SigmoidWithLoss
import collections


class EmbeddingDot:
    def __init__(self, W):
        #行を選択的に選ぶレイヤ　params, gradsという名前は固定する（後でOptimizerと組み合わせるときに参照する）
        #W_outは縦に単語のベクトルが並んでいたが，この場合は横方向に単語のベクトルが並んでいる．そのため行列の積として計算することはできないので注意
        # * とsumを用いることでnp.dotを使わずに内積を計算している
        self.embed = Embedding(W) 
        self.params = self.embed.params
        self.grads = self.embed.grads
        self.cache = None 
    def forward(self, h, idx): 
        target_W = self.embed.forward(idx)#idxで指定した行をWから選ぶ
        out = np.sum(target_W*h, axis = 1) #内積の計算
        self.cache = (h, target_W) #覚えておく（backwardのときに使う）
        return out

    def backward(self, dout):
        h, target_W = self.cache 
        dout = dout.reshape(dout.shape[0],1) #doutを２次元に整形
        dtarget_W = dout * h #Wの微分は結局内積
        self.embed.backward(dtarget_W) #Wの更新 参照渡しで更新されるためdWが変化するとgradsも更新される
        dh = dout *target_W
        return dh
        
class UnigramSampler:
    def __init__(self, corpus, power, sample_size):
        self.sample_size = sample_size
        self.vocab_size = None
        self.word_p = None

        counts = collections.Counter()
        for word_id in corpus:
            counts[word_id] += 1

        vocab_size = len(counts)
        self.vocab_size = vocab_size

        self.word_p = np.zeros(vocab_size)
        for i in range(vocab_size):
            self.word_p[i] = counts[i]

        self.word_p = np.power(self.word_p, power)
        self.word_p /= np.sum(self.word_p)

    def get_negative_sample(self, target): 
        #指定したターゲット（id）を入力するとコーパスの中からターゲットid以外のidをsample_sizeの数だけランダムに抽出する
        batch_size = target.shape[0]

        if not GPU:
            negative_sample = np.zeros((batch_size, self.sample_size), dtype=np.int32)

            for i in range(batch_size):
                p = self.word_p.copy()
                target_idx = target[i]
                p[target_idx] = 0
                p /= p.sum()
                negative_sample[i, :] = np.random.choice(self.vocab_size, size=self.sample_size, replace=False, p=p)
        else:
            # GPU(cupy）で計算するときは、速度を優先
            # 負例にターゲットが含まれるケースがある
            negative_sample = np.random.choice(self.vocab_size, size=(batch_size, self.sample_size),
                                               replace=True, p=self.word_p)

        return negative_sample #idの組み合わせで返却する

class NegativeSamplingLoss:
    def __init__ (self, W, corpus, power = 0.75, sample_size=5):
        self.sample_size = sample_size
        self.sampler = UnigramSampler(corpus, power, sample_size) #↑で定義したクラスを使えるように
        self.loss_layers = [SigmoidWithLoss() for _ in range(sample_size +1)] #負例用がsample_size個，正解用が１個
        self.embed_dot_layers = [EmbeddingDot(W) for _ in range(sample_size +1)]#パラメータはこの層が担っている

        self.params, self.grads = [], [] #この層そのものに学習するパラメータはない
        for layer in self.embed_dot_layers: #損失関数lossも学習するパラメータはないのでfor文はembed_dot_layersだけ
            self.params += layer.params
            self.grads += layer.grads
    def forward(self, h, target):
        batch_size = target.shape[0] #targetはidの組み合わせ，行の数が正例の数
        
        negative_sample = self.sampler.get_negative_sample(target)

        #正例
        score = self.embed_dot_layers[0].forward(h, target) #targetは注目するid, つまりWの行を指定する
        correct_label = np.ones(batch_size, dtype = np.int32) #正解ラベルの作成. すべて１のリスト
        loss = self.loss_layers[0].forward(score, correct_label)

        #負例
        negative_label = np.zeros(batch_size, dtype = np.int32)
        for i in range(self.sample_size):
            negative_target = negative_sample[:,i] #batchに含まれるすべてのデータに対するtarget. 列ベクトル
            score = self.embed_dot_layers[1+i].forward(h, negative_target) #1+なのは0番目が正例だから．negative_targetもid, つまり行を指定する
            loss += self.loss_layers[1+i].forward(score, negative_label)
        return loss

    def backward(self, dout = 1):
        dh = 0
        for l0, l1 in zip(self.loss_layers, self.embed_dot_layers):
            dscore = l0.backward(dout)
            dh += l1.backward(dscore) #正，負例があるので誤差は足し合わせる
        return dh 