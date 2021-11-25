import math

import paddle
import paddle.nn as nn
import paddle.nn.functional as F
import paddlenlp as ppnlp

from data import PinyinTokenizer


class LazyEmbedding:
    def __init__(self, model):
        self.model = model
        for n, p in self.model.named_parameters():
            if 'word_embeddings' in n:
                self.embedding_new = p
                break
        self.embedding_old = self.embedding_new.clone()

    def stop_update(self):
        # if self.embedding_new.grad.isnan().any() or self.embedding_new.grad.isinf().any():
        #     print('Embedding 梯度有 nan/inf，Embedding 不进行稀疏更新')
        #     return

        with paddle.no_grad():
            self.embedding_old = paddle.where(self.embedding_new.grad != 0, self.embedding_new, self.embedding_old)
            self.embedding_new.stop_gradient = True
            self.embedding_new.add_(self.embedding_old - self.embedding_new)
            self.embedding_new.stop_gradient = False


class FGM:
    def __init__(self, model):
        self.model      = model
        self.inf_or_nan = False
        self.backup     = dict()

    def attack(self, epsilon=1., emb_name='word_embeddings'):
        # emb_name这个参数要换成你模型中embedding的参数名
        for name, param in self.model.named_parameters():
            if param.stop_gradient or (emb_name not in name): continue

            norm = param.grad.norm()
            if norm.isinf() or norm.isnan():
                self.inf_or_nan = True
                break

            r_at = epsilon * param.grad / (norm + 1e-8)
            self.backup[name] = r_at.clone()

            param.stop_gradient = True
            param.add_(r_at)
            param.stop_gradient = False

    def restore(self, emb_name='word_embeddings'):
        # emb_name这个参数要换成你模型中embedding的参数名
        for name, param in self.model.named_parameters():
            if param.stop_gradient or (emb_name not in name): continue
            if self.inf_or_nan:
                self.inf_or_nan = False
                break
            assert name in self.backup

            param.stop_gradient = True
            param.subtract_(self.backup[name])
            param.stop_gradient = False

        self.backup = dict()


class PinyinEmbeddings(nn.Layer):
    def __init__(self,
                 vocab_size,
                 emb_size=128,
                 hidden_dropout_prob=0.1,
                 max_position_embeddings=512,
                 type_vocab_size=2,
                 pad_token_id=0):
        super(PinyinEmbeddings, self).__init__()
        self.word_embeddings       = nn.Embedding(vocab_size,              emb_size, padding_idx=pad_token_id)
        self.position_embeddings   = nn.Embedding(max_position_embeddings, emb_size)
        self.token_type_embeddings = nn.Embedding(type_vocab_size,         emb_size)
        self.layer_norm            = nn.LayerNorm(emb_size)
        self.dropout               = nn.Dropout(hidden_dropout_prob)

    def forward(self, input_ids, token_type_ids, cls_emb):
        ones         = paddle.ones_like(input_ids, dtype="int64")
        seq_length   = paddle.cumsum(ones, axis=1)
        position_ids = seq_length - ones
        position_ids.stop_gradient = True

        input_embedings       = self.word_embeddings(input_ids)
        input_embedings       = paddle.concat([cls_emb, input_embedings[:, 1:]], axis=1)
        position_embeddings   = self.position_embeddings(position_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)

        embeddings = input_embedings + position_embeddings + token_type_embeddings
        embeddings = self.layer_norm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings


class Pinyin(nn.Layer):
    def __init__(self, scale, ptm_hidden_size, vocab_size, pad_token_id):
        super().__init__()
        self.pad_token_id = pad_token_id
        hidden_size = int(768 * scale)

        self.dropout = nn.Dropout(0.1)
        self.cls_linear = nn.Linear(ptm_hidden_size, hidden_size)

        self.embeddings = PinyinEmbeddings(
            vocab_size,
            emb_size=hidden_size,
            hidden_dropout_prob=0.1,
            max_position_embeddings=512,
            type_vocab_size=2,
            pad_token_id=0)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=int(12 * scale),
            dim_feedforward=hidden_size * 4,
            dropout=0.1,
            activation='gelu',
            attn_dropout=0.1,
            act_dropout=0)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=1)
        self.pooler  = ppnlp.transformers.ernie.modeling.ErniePooler(hidden_size)

    def forward(self,
                cls_emb,
                input_ids,
                token_type_ids):
        cls_emb = self.cls_linear(cls_emb)
        embedding_output = self.embeddings(input_ids, token_type_ids, cls_emb)

        attention_mask = paddle.unsqueeze(
            (input_ids == self.pad_token_id
             ).astype(self.pooler.dense.weight.dtype) * -1e9,
            axis=[1, 2])
        encoder_outputs = self.encoder(embedding_output, attention_mask)
        feat = encoder_outputs[:, 0]
        cls_emb = self.pooler(encoder_outputs)
        return cls_emb, feat


class QuestionMatching(nn.Layer):
    def __init__(self, pretrained_model, dropout=None, use_pinyin=False):
        super().__init__()
        self.ptm  = pretrained_model

        self.use_pinyin = use_pinyin
        if use_pinyin:
            pinyin2num = PinyinTokenizer().pinyin_dict
            pinyin_scale = 1 / 3
            self.pinyin     = Pinyin(scale=pinyin_scale, ptm_hidden_size=self.ptm.config["hidden_size"], vocab_size=len(pinyin2num), pad_token_id=pinyin2num['[PAD]'])
            self.classifier = nn.Linear(int(768*pinyin_scale), 2)
        else:
            self.classifier = nn.Linear(self.ptm.config['hidden_size'], 2)

        self.dropout    = nn.Dropout(dropout if dropout is not None else 0.1)

    def forward(self,
                input_ids,
                token_type_ids=None,
                position_ids=None,
                pinyin_ids=None,
                pinyin_type_ids=None,
                attention_mask=None,
                do_evaluate=False):

        sequence_output, cls_embedding = self.ptm(input_ids, token_type_ids, position_ids, attention_mask)
        feat = sequence_output[:, 0]

        if self.use_pinyin:
            cls_embedding, feat = self.pinyin(sequence_output[:, :1], pinyin_ids, pinyin_type_ids)

        cls_embedding = self.dropout(cls_embedding)
        logits        = self.classifier(cls_embedding)

        return logits, feat
