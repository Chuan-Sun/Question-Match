import random
import numpy as np
import pypinyin

import paddle
from paddlenlp.datasets import MapDataset


class PinyinTokenizer:
    def __init__(self):
        self.pinyin_dict = {
            '[PAD]': 0,
            '[CLS]': 1,   # 不用
            '[SEP]': 2,
            '[MASK]': 3,  # 不用
            'a': 4,
            'ai': 5,
            'an': 6,
            'ang': 7,
            'ao': 8,
            'ba': 9,
            'bai': 10,
            'ban': 11,
            'bang': 12,
            'bao': 13,
            'bei': 14,
            'ben': 15,
            'beng': 16,
            'bi': 17,
            'bian': 18,
            'biao': 19,
            'bie': 20,
            'bin': 21,
            'bing': 22,
            'bo': 23,
            'bu': 24,
            'ca': 25,
            'cai': 26,
            'can': 27,
            'cang': 28,
            'cao': 29,
            'ce': 30,
            'cen': 31,
            'ceng': 32,
            'cha': 33,
            'chai': 34,
            'chan': 35,
            'chang': 36,
            'chao': 37,
            'che': 38,
            'chen': 39,
            'cheng': 40,
            'chi': 41,
            'chong': 42,
            'chou': 43,
            'chu': 44,
            'chuai': 45,
            'chuan': 46,
            'chuang': 47,
            'chui': 48,
            'chun': 49,
            'chuo': 50,
            'ci': 51,
            'cong': 52,
            'cou': 53,
            'cu': 54,
            'cuan': 55,
            'cui': 56,
            'cun': 57,
            'cuo': 58,
            'da': 59,
            'dai': 60,
            'dan': 61,
            'dang': 62,
            'dao': 63,
            'de': 64,
            'dei': 65,
            'deng': 66,
            'di': 67,
            'dia': 68,
            'dian': 69,
            'diao': 70,
            'die': 71,
            'ding': 72,
            'diu': 73,
            'dong': 74,
            'dou': 75,
            'du': 76,
            'duan': 77,
            'dui': 78,
            'dun': 79,
            'duo': 80,
            'e': 81,
            'ei': 82,
            'en': 83,
            'er': 84,
            'fa': 85,
            'fan': 86,
            'fang': 87,
            'fei': 88,
            'fen': 89,
            'feng': 90,
            'fo': 91,
            'fou': 92,
            'fu': 93,
            'ga': 94,
            'gai': 95,
            'gan': 96,
            'gang': 97,
            'gao': 98,
            'ge': 99,
            'gei': 100,
            'gen': 101,
            'geng': 102,
            'gong': 103,
            'gou': 104,
            'gu': 105,
            'gua': 106,
            'guai': 107,
            'guan': 108,
            'guang': 109,
            'gui': 110,
            'gun': 111,
            'guo': 112,
            'ha': 113,
            'hai': 114,
            'han': 115,
            'hang': 116,
            'hao': 117,
            'he': 118,
            'hei': 119,
            'hen': 120,
            'heng': 121,
            'hong': 122,
            'hou': 123,
            'hu': 124,
            'hua': 125,
            'huai': 126,
            'huan': 127,
            'huang': 128,
            'hui': 129,
            'hun': 130,
            'huo': 131,
            'ji': 132,
            'jia': 133,
            'jian': 134,
            'jiang': 135,
            'jiao': 136,
            'jie': 137,
            'jin': 138,
            'jing': 139,
            'jiong': 140,
            'jiu': 141,
            'ju': 142,
            'juan': 143,
            'jue': 144,
            'jun': 145,
            'ka': 146,
            'kai': 147,
            'kan': 148,
            'kang': 149,
            'kao': 150,
            'ke': 151,
            'ken': 152,
            'keng': 153,
            'kong': 154,
            'kou': 155,
            'ku': 156,
            'kua': 157,
            'kuai': 158,
            'kuan': 159,
            'kuang': 160,
            'kui': 161,
            'kun': 162,
            'kuo': 163,
            'la': 164,
            'lai': 165,
            'lan': 166,
            'lang': 167,
            'lao': 168,
            'le': 169,
            'lei': 170,
            'leng': 171,
            'li': 172,
            'lia': 173,
            'lian': 174,
            'liang': 175,
            'liao': 176,
            'lie': 177,
            'lin': 178,
            'ling': 179,
            'liu': 180,
            'long': 181,
            'lou': 182,
            'lu': 183,
            'luan': 184,
            'lun': 185,
            'luo': 186,
            'lv': 187,
            'lve': 188,
            'ma': 189,
            'mai': 190,
            'man': 191,
            'mang': 192,
            'mao': 193,
            'me': 194,
            'mei': 195,
            'men': 196,
            'meng': 197,
            'mi': 198,
            'mian': 199,
            'miao': 200,
            'mie': 201,
            'min': 202,
            'ming': 203,
            'miu': 204,
            'mo': 205,
            'mou': 206,
            'mu': 207,
            'n': 208,
            'na': 209,
            'nai': 210,
            'nan': 211,
            'nang': 212,
            'nao': 213,
            'ne': 214,
            'nei': 215,
            'nen': 216,
            'neng': 217,
            'ni': 218,
            'nian': 219,
            'niang': 220,
            'niao': 221,
            'nie': 222,
            'nin': 223,
            'ning': 224,
            'niu': 225,
            'nong': 226,
            'nu': 227,
            'nuan': 228,
            'nuo': 229,
            'nv': 230,
            'nve': 231,
            'o': 232,
            'ou': 233,
            'pa': 234,
            'pai': 235,
            'pan': 236,
            'pang': 237,
            'pao': 238,
            'pei': 239,
            'pen': 240,
            'peng': 241,
            'pi': 242,
            'pian': 243,
            'piao': 244,
            'pie': 245,
            'pin': 246,
            'ping': 247,
            'po': 248,
            'pou': 249,
            'pu': 250,
            'qi': 251,
            'qia': 252,
            'qian': 253,
            'qiang': 254,
            'qiao': 255,
            'qie': 256,
            'qin': 257,
            'qing': 258,
            'qiong': 259,
            'qiu': 260,
            'qu': 261,
            'quan': 262,
            'que': 263,
            'qun': 264,
            'ran': 265,
            'rang': 266,
            'rao': 267,
            're': 268,
            'ren': 269,
            'reng': 270,
            'ri': 271,
            'rong': 272,
            'rou': 273,
            'ru': 274,
            'ruan': 275,
            'rui': 276,
            'run': 277,
            'ruo': 278,
            'sa': 279,
            'sai': 280,
            'san': 281,
            'sang': 282,
            'sao': 283,
            'se': 284,
            'sen': 285,
            'seng': 286,
            'sha': 287,
            'shai': 288,
            'shan': 289,
            'shang': 290,
            'shao': 291,
            'she': 292,
            'shei': 293,
            'shen': 294,
            'sheng': 295,
            'shi': 296,
            'shou': 297,
            'shu': 298,
            'shua': 299,
            'shuai': 300,
            'shuan': 301,
            'shuang': 302,
            'shui': 303,
            'shun': 304,
            'shuo': 305,
            'si': 306,
            'song': 307,
            'sou': 308,
            'su': 309,
            'suan': 310,
            'sui': 311,
            'sun': 312,
            'suo': 313,
            'ta': 314,
            'tai': 315,
            'tan': 316,
            'tang': 317,
            'tao': 318,
            'te': 319,
            'teng': 320,
            'ti': 321,
            'tian': 322,
            'tiao': 323,
            'tie': 324,
            'ting': 325,
            'tong': 326,
            'tou': 327,
            'tu': 328,
            'tuan': 329,
            'tui': 330,
            'tun': 331,
            'tuo': 332,
            'wa': 333,
            'wai': 334,
            'wan': 335,
            'wang': 336,
            'wei': 337,
            'wen': 338,
            'weng': 339,
            'wo': 340,
            'wu': 341,
            'xi': 342,
            'xia': 343,
            'xian': 344,
            'xiang': 345,
            'xiao': 346,
            'xie': 347,
            'xin': 348,
            'xing': 349,
            'xiong': 350,
            'xiu': 351,
            'xu': 352,
            'xuan': 353,
            'xue': 354,
            'xun': 355,
            'ya': 356,
            'yan': 357,
            'yang': 358,
            'yao': 359,
            'ye': 360,
            'yi': 361,
            'yin': 362,
            'ying': 363,
            'yo': 364,
            'yong': 365,
            'you': 366,
            'yu': 367,
            'yuan': 368,
            'yue': 369,
            'yun': 370,
            'za': 371,
            'zai': 372,
            'zan': 373,
            'zang': 374,
            'zao': 375,
            'ze': 376,
            'zei': 377,
            'zen': 378,
            'zeng': 379,
            'zha': 380,
            'zhai': 381,
            'zhan': 382,
            'zhang': 383,
            'zhao': 384,
            'zhe': 385,
            'zhen': 386,
            'zheng': 387,
            'zhi': 388,
            'zhong': 389,
            'zhou': 390,
            'zhu': 391,
            'zhua': 392,
            'zhuai': 393,
            'zhuan': 394,
            'zhuang': 395,
            'zhui': 396,
            'zhun': 397,
            'zhuo': 398,
            'zi': 399,
            'zong': 400,
            'zou': 401,
            'zu': 402,
            'zuan': 403,
            'zui': 404,
            'zun': 405,
            'zuo': 406,
            'den': 407,
            'nou': 408
        }

    def __call__(self, q1, q2, max_seq_len):
        q1_py, q2_py = self.get_pinyin_id(q1), self.get_pinyin_id(q2)
        q12_py = q1_py + q2_py
        q1_py  = q12_py[:min(max_seq_len-3, len(q1_py))]      # -3 是因为 [CLS], [SEP], [SEP]
        q2_py  = q12_py[ min(max_seq_len-3, len(q1_py)):max_seq_len-3]

        pinyin_ids = [self.pinyin_dict['[CLS]'],
                      *q1_py,
                      self.pinyin_dict['[SEP]'],
                      *q2_py,
                      self.pinyin_dict['[SEP]']]
        pinyin_type_ids = [0] * (len(q1_py) + 2) + [1] * (len(q2_py) + 1)
        return pinyin_ids, pinyin_type_ids

    def get_pinyin_id(self, q):
        q         = ''.join(c for c in q if u'\u4e00' <= c <= u'\u9fa5')
        pinyin    = [py[0] for py in pypinyin.pinyin(q, style=pypinyin.Style.NORMAL)]
        pinyin_id = [self.pinyin_dict[py] for py in pinyin]
        return pinyin_id


def create_dataloader(dataset,
                      mode='train',
                      batch_size=1,
                      batchify_fn=None,
                      trans_fn=None):
    if trans_fn:
        dataset = dataset.map(trans_fn)

    shuffle = True if mode == 'train' else False
    if mode == 'train':
        batch_sampler = paddle.io.DistributedBatchSampler(
            dataset, batch_size=batch_size, shuffle=shuffle)
    else:
        batch_sampler = paddle.io.BatchSampler(
            dataset, batch_size=batch_size, shuffle=shuffle)

    num_workers = 4 if mode == 'train' else 0
    return paddle.io.DataLoader(
        dataset=dataset,
        batch_sampler=batch_sampler,
        collate_fn=batchify_fn,
        return_list=True,
        num_workers=num_workers
    )


def read_text_pair(data_path, mode='train'):
    """Reads data."""
    with open(data_path, 'r', encoding='utf-8') as f:
        for line in f:
            data = line.rstrip().split("\t")
            if mode == 'train':     # query, logit, prob, (pseudo) label, pseudo_bool
                yield {'query1': data[0], 'query2': data[1], 'logit0': data[2], 'logit1': data[3],
                       'prob0':  data[4], 'prob1':  data[5], 'label':  data[6], 'pseudo': data[7],
                       'weight': data[8]}
            elif mode == 'dev':
                yield {'query1': data[0], 'query2': data[1], 'label': data[2]}
            elif mode == 'test':
                yield {'query1': data[0], 'query2': data[1]}
            else:
                raise


def convert_example(example, tokenizer, max_seq_length=512, absolute_position_aware=False, aug_exchange=0, mode='train'):
    q1, q2 = example["query1"], example["query2"]

    if mode == 'train' and random.random() < aug_exchange:         # random.random() 是 0 ~ 1 均匀分布
        q1, q2 = q2, q1

    if absolute_position_aware:
        len1 = max_seq_length // 2 - 1
        len2 = max_seq_length // 2 - 2
        encoded_inputs1 = tokenizer(text=q1, max_seq_len=len1+2)
        encoded_inputs2 = tokenizer(text=q2, max_seq_len=len2+2)
        input_ids1, input_ids2 = encoded_inputs1['input_ids'], encoded_inputs2['input_ids'][1:]

        input_ids      = input_ids1 + input_ids2
        token_type_ids = [0] * len(input_ids1) + [1] * len(input_ids2)
        position_ids   = list(range(len(input_ids1))) + list(range(len1+2, len1+2+len(input_ids2)))
    else:
        encoded_inputs = tokenizer(text=q1, text_pair=q2, max_seq_len=max_seq_length)

        input_ids      = encoded_inputs["input_ids"]
        token_type_ids = encoded_inputs["token_type_ids"]
        position_ids   = list(range(len(input_ids)))

    pinyin_tokenizer = PinyinTokenizer()
    pinyin_ids, pinyin_type_ids = pinyin_tokenizer(q1, q2, max_seq_length)

    if mode == 'train':
        logit  = np.array([example['logit0'], example['logit1']], dtype='float32')
        prob   = np.array([example['prob0'],  example['prob1']],  dtype='float32')
        label  = np.array([example['label']], dtype='int64')
        pseudo = np.array([example['pseudo']], dtype='int64')
        weight = np.array([example['weight']], dtype='float32')
        return input_ids, token_type_ids, position_ids, pinyin_ids, pinyin_type_ids, logit, prob, label, pseudo, weight
    elif mode == 'dev':
        label = np.array([example["label"]], dtype="int64")
        return input_ids, token_type_ids, position_ids, pinyin_ids, pinyin_type_ids, label
    elif mode == 'test':
        return input_ids, token_type_ids, position_ids, pinyin_ids, pinyin_type_ids
    else:
        raise
