import argparse
import re
import pickle
import numpy as np
import pandas as pd
# import jieba
# import jieba.posseg as psg
# import pycorrector
# from pycorrector import word_frequency, ppl_score, ngram_score
import pypinyin
from pypinyin.style._utils import get_initials, get_finals
# import cn2an
# import hanlp


parser = argparse.ArgumentParser()
parser.add_argument("--input_path",  default=[], type=str, nargs='+', help="模型预测结果，多个输入就集成")
parser.add_argument("--input_cols",  default=[], type=str, nargs='+', help="输入文件列名")
parser.add_argument("--output_path", default='../data/ccf_qianyan_qm_result_A2.csv', type=str, help="后处理后的文件")
parser.add_argument("--output_cols", default=[], type=str, nargs='+', help="输出文件列名")
parser.add_argument("--can_alpha",   default=1,  type=int, help="CAN 正负样本比例调节的指数")
parser.add_argument("--can_iters",   default=2,  type=int, help="CAN 正负样本比例调节的迭代次数")

args = parser.parse_args()


class Misspelling:
    def __init__(self):
        self.homophony_mask = None
        self.sim_char_dic = np.load('similar_character.npy', allow_pickle=True).item()

    def postprocess(self, pred):
        pred = pred.copy()
        self.homophony_mask = pred.apply(self.is_homophony, axis=1)
        n_changed = (pred.loc[self.homophony_mask, 'pred'] == 0).sum()
        pred.loc[self.homophony_mask, 'pred'] = 1
        pred.loc[self.homophony_mask, 'prob0'] = 0.05
        pred.loc[self.homophony_mask, 'prob1'] = 0.95
        print('homophony: ', self.homophony_mask.sum(), n_changed)
        return pred

    def is_homophony(self, d: pd.Series):
        q1, q2 = d['query1'], d['query2']
        if (len(q1) != len(q2)) or (q1 == q2): return False

        q1_arr, q2_arr = np.array(list(q1)), np.array(list(q2))
        mask = q1_arr != q2_arr

        # 同音
        py1 = [p[0] for p in pypinyin.pinyin(q1_arr[mask], style=pypinyin.Style.NORMAL)]
        py2 = [p[0] for p in pypinyin.pinyin(q2_arr[mask], style=pypinyin.Style.NORMAL)]

        if py1 == py2: return True

        # 多音
        py1 = pypinyin.pinyin(q1_arr[mask], style=pypinyin.Style.NORMAL, heteronym=True)
        py2 = pypinyin.pinyin(q2_arr[mask], style=pypinyin.Style.NORMAL, heteronym=True)

        if all([len(set(p1) & set(p2)) != 0 for p1, p2 in zip(py1, py2)]) and (d['prob0'] < 0.9):
            return True

        # 相似字
        flags = []
        for ch1, ch2 in zip(q1_arr[mask], q2_arr[mask]):
            ch1_sim = self.sim_char_dic.get(ch1, '')
            flags.append(ch2 in ch1_sim)
        if all(flags) and (d['prob0'] < 0.9): return True

        return False


class Syntactic:
    def __init__(self, syntactic_analysis):
        self.symmetry_mask = None
        self.asymmetry_mask = None
        self.negative_asymmetry_mask = None
        self.syntactic_analysis = syntactic_analysis

    def postprocess(self, pred):
        pred = pred.copy()

        self.symmetry_mask = pred.apply(self.symmetry, axis=1)
        n_changed = (pred.loc[self.symmetry_mask, 'pred'] == 0).sum()
        pred.loc[self.symmetry_mask, 'pred'] = 1
        pred.loc[self.symmetry_mask, 'prob0'] = 0.05
        pred.loc[self.symmetry_mask, 'prob1'] = 0.95
        print('symmetry: ', self.symmetry_mask.sum(), n_changed)

        self.asymmetry_mask = pred.apply(self.asymmetry, axis=1)
        n_changed = (pred.loc[self.asymmetry_mask, 'pred'] == 1).sum()
        pred.loc[self.asymmetry_mask, 'pred'] = 0
        pred.loc[self.asymmetry_mask, 'prob0'] = 0.95
        pred.loc[self.asymmetry_mask, 'prob1'] = 0.05
        print('asymmetry: ', self.asymmetry_mask.sum(), n_changed)

        self.negative_asymmetry_mask = pred.apply(self.negative_asymmetry, axis=1)
        n_changed = (pred.loc[self.negative_asymmetry_mask, 'pred'] == 0).sum()
        pred.loc[self.negative_asymmetry_mask, 'pred'] = 1
        pred.loc[self.negative_asymmetry_mask, 'prob0'] = 0.05
        pred.loc[self.negative_asymmetry_mask, 'prob1'] = 0.95
        print('negative_asymmetry: ', self.negative_asymmetry_mask.sum(), n_changed)

        return pred

    def symmetry(self, d: pd.Series):
        q1, q2 = d['query1'], d['query2']
        if (len(q1) != len(q2)) or (q1 == q2): return False

        q1_word, q1_pos = self.syntactic_analysis[q1]['jieba']
        q2_word, q2_pos = self.syntactic_analysis[q2]['jieba']

        # 确保是俩相同词语调换顺序
        if sorted(q1_word) != sorted(q2_word): return False
        diff = self._diff_index(q1_word, q2_word)
        if len(diff) != 2: return False
        if (q1_word[diff[0]] != q2_word[diff[1]]) or (q1_word[diff[1]] != q2_word[diff[0]]): return False


        if '还是' in q1_word[diff[0]+1:diff[1]]:
            return True

        return False

    def asymmetry(self, d: pd.Series):
        q1, q2 = d['query1'], d['query2']
        if (len(q1) != len(q2)) or (q1 == q2): return False

        q1_word, q1_pos = self.syntactic_analysis[q1]['jieba']
        q2_word, q2_pos = self.syntactic_analysis[q2]['jieba']

        # 确保是俩相同词语调换顺序
        if sorted(q1_word) != sorted(q2_word): return False
        diff = self._diff_index(q1_word, q2_word)
        if len(diff) != 2: return False
        if (q1_word[diff[0]] != q2_word[diff[1]]) or (q1_word[diff[1]] != q2_word[diff[0]]): return False

        tok, dep = self.syntactic_analysis[q1]['hanlp']
        if tok != q1_word: return False
        dep0, dep1 = dep[diff[0]], dep[diff[1]]

        # 名词短语的一部分调换位置
        if (dep0[1] == dep1[1] in {'nn', 'amod', 'assmod', 'rcmod', 'tmod', 'dep'}) and (tok[ dep0[0]-1 ] != tok[ dep1[0]-1 ]):
            return True

        # 关联修饰调换
        p0, p1 = diff
        while dep[p0][1] in {'assmod', 'lobj', 'pobj'}:
            p0 = dep[p0][0] - 1
        while dep[p1][1] in {'nn', 'pobj'}:
            p1 = dep[p1][0] - 1
        if p0 == p1:
            return True

        return False

    def negative_asymmetry(self, d: pd.Series):
        q1, q2 = d['query1'], d['query2']
        if (len(q1) != len(q2)) or (q1 == q2): return False

        q1_word, q1_pos = self.syntactic_analysis[q1]['jieba']
        q2_word, q2_pos = self.syntactic_analysis[q2]['jieba']

        if len(q1_word) != len(q2_word): return False
        diff = self._diff_index(q1_word, q2_word)
        if len(diff) != 3: return False
        if ('比' not in q1_word) or ('比' not in q2_word): return False
        if (q1_word[diff[0]] != q2_word[diff[1]]) or (q1_word[diff[1]] != q2_word[diff[0]]): return False

        return True

    def _diff_index(self, q1_word, q2_word):
        diff = []
        for i, (w1, w2) in enumerate(zip(q1_word, q2_word)):
            if w1 == w2: continue
            diff.append(i)
        return diff


class Math:
    def __init__(self):
        self.diff_num_mask = None
        self.symbols = [
            '加', '加上', '+', '减', '减去', '-', '乘', '乘以', '×', '*', 'x', 'X', '除', '除以', '/', '÷', '等于', '=', '公倍数', '公约数', '公约数', '公因数'
        ]

    def postprocess(self, pred):
        pred = pred.copy()
        self.diff_num_mask = pred.apply(self.is_diff_num, axis=1)
        n_changed = (pred.loc[self.diff_num_mask, 'pred'] == 1).sum()
        pred.loc[self.diff_num_mask, 'pred'] = 0
        pred.loc[self.diff_num_mask, 'prob0'] = 0.95
        pred.loc[self.diff_num_mask, 'prob1'] = 0.05
        print('diff_num: ', self.diff_num_mask.sum(), n_changed)
        return pred

    def is_diff_num(self, d: pd.Series):
        q1, q2 = d['query1'], d['query2']

        nums1 = re.findall(r"\d+\.?\d*", q1)
        nums2 = re.findall(r"\d+\.?\d*", q2)

        if len(nums1) != len(nums2): return False
        if sorted(nums1) == sorted(nums2): return False
        if len(re.findall(r'[+\-×xX*/÷=]', q1 + q2)) == 0: return False     # 等于 加减乘除
        return True


class CAN:
    def __init__(self, alpha, iters):
        self.alpha = alpha
        self.iters = iters

    def postprocess(self, pred):
        pred = pred.copy()
        prob = self.prob_postprocess(pred[['prob0', 'prob1']].values)
        print('CAN: ', (pred['pred'] != prob.argmax(1)).sum())
        pred[['prob0', 'prob1']] = prob
        pred['pred'] = prob.argmax(1)
        return pred

    def prob_postprocess(self, y_pred):
        prior = np.array([0.6903327690476333, 0.3096672309523667]) # 训练集 oppo正负样本比例
        y_pred_uncertainty = -(y_pred * np.log(y_pred)).sum(1) / np.log(2)

        threshold = 0.95
        y_pred_confident = y_pred[y_pred_uncertainty < threshold]
        y_pred_unconfident = y_pred[y_pred_uncertainty >= threshold]

        post = []
        for i, y in enumerate(y_pred_unconfident):
            Y = np.concatenate([y_pred_confident, y[None]], axis=0)
            for j in range(self.iters):
                Y = Y ** self.alpha
                Y /= Y.sum(axis=0, keepdims=True)
                Y *= prior[None]
                Y /= Y.sum(axis=1, keepdims=True)
            y = Y[-1]
            post.append(y.tolist())

        post = np.array(post)
        y_pred[y_pred_uncertainty >= threshold] = post

        return y_pred


class LexicalSemantics:
    def __init__(self):
        self.frequency_insert_adv = [
            {'总是', '总', '一直', '永远', '老是', '老'},
            {'经常', '常常', '常'},
            {'偶尔'},
            {'很少'},
            {'很多'},
        ]
        self.frequency_replace_adv = [
            {'总是', '总', '一直', '永远', '老是', '老'},
            {'经常', '常常', '常'},
            {'偶尔', '有时', '有时候'},
            {'很少'},
            {'很多'},
            {'有点'},
        ]
        self.word_phase_insert_adv_mask = None
        self.word_phase_replace_adv_mask = None

        self.temporal_insert = [
            {'最近'},
            {'今年', '去年', '前年'},
            {'刚', '刚刚', '刚刚在', '刚刚的', '曾经', '之前', '以前'},
            {'将来', '未来', '以后', '之后', '以后的'},
            {'依然'}
        ]
        self.temporal_replace = [
            {'现在', '时', '的时候', '在'},
            {'最近'},
            {'今年', '去年', '前年'},
            {'刚', '刚刚', '刚刚在', '刚刚的', '曾经', '之前', '以前'},
            {'将来', '未来', '后', '以后', '之后', '以后的'},
            {'还', '依然'}
        ]
        self.temporal_insert_mask = None
        self.temporal_replace_mask = None

    def postprocess(self, pred):
        pred = pred.copy()

        self.word_phase_insert_adv_mask = pred.apply(lambda d: self.insert_detect(d, self.frequency_insert_adv), axis=1)
        n_changed = (pred.loc[self.word_phase_insert_adv_mask, 'pred'] == 1).sum()
        pred.loc[self.word_phase_insert_adv_mask, 'pred'] = 0
        pred.loc[self.word_phase_insert_adv_mask, 'prob0'] = 0.95
        pred.loc[self.word_phase_insert_adv_mask, 'prob1'] = 0.05
        print('Word & Phase / insert adv: ', self.word_phase_insert_adv_mask.sum(), n_changed)

        self.word_phase_replace_adv_mask = pred.apply(lambda d: self.replace_detect(d, self.frequency_replace_adv), axis=1)
        n_changed = (pred.loc[self.word_phase_replace_adv_mask, 'pred'] == 1).sum()
        pred.loc[self.word_phase_replace_adv_mask, 'pred'] = 0
        pred.loc[self.word_phase_replace_adv_mask, 'prob0'] = 0.95
        pred.loc[self.word_phase_replace_adv_mask, 'prob1'] = 0.05
        print('Word & Phase / replace adv: ', self.word_phase_replace_adv_mask.sum(), n_changed)

        self.temporal_insert_mask = pred.apply(lambda d: self.insert_detect(d, self.temporal_insert), axis=1)
        n_changed = (pred.loc[self.temporal_insert_mask, 'pred'] == 1).sum()
        pred.loc[self.temporal_insert_mask, 'pred'] = 0
        pred.loc[self.temporal_insert_mask, 'prob0'] = 0.95
        pred.loc[self.temporal_insert_mask, 'prob1'] = 0.05
        print('Temporal / insert: ', self.temporal_insert_mask.sum(), n_changed)

        self.temporal_replace_mask = pred.apply(lambda d: self.replace_detect(d, self.temporal_replace), axis=1)
        n_changed = (pred.loc[self.temporal_replace_mask, 'pred'] == 1).sum()
        pred.loc[self.temporal_replace_mask, 'pred'] = 0
        pred.loc[self.temporal_replace_mask, 'prob0'] = 0.95
        pred.loc[self.temporal_replace_mask, 'prob1'] = 0.05
        print('Temporal / replace: ', self.temporal_replace_mask.sum(), n_changed)

        return pred

    def insert_detect(self, d: pd.Series, insert_words):
        q1, q2 = d['query1'], d['query2']
        fre_adv = sum([list(adv) for adv in insert_words], [])
        for w in fre_adv:
            if (len(re.findall(w, q1+q2)) == 1) and (q1.replace(w, '') == q2.replace(w, '')):
                return True
        return False

    def replace_detect(self, d: pd.Series, replace_words):
        q1, q2 = d['query1'], d['query2']
        fre_adv = set.union(*[adv for adv in replace_words])
        for advs in replace_words:
            fre_adv_other = fre_adv - advs
            for w in advs:
                if len(re.findall(w, q1)) + len(re.findall(w, q2)) != 1: continue
                if w in q2: q1, q2 = q2, q1
                q10, q11 = q1.split(w)
                ww = q2.replace(q10, '')
                ww = ww.replace(q11, '')
                if len(ww) == 0: continue
                if q1.replace(w, '') != q2.replace(ww, ''): continue
                if ww in fre_adv_other: return True
        return False


class SpeechFiller:
    def __init__(self):
        self.filler_words = ['就是', '急急', '问', '告诉', '介绍', '求', '谁能', '谢', '请', '简述', '有没有', '什么意思', '纠结', '怎么办', '怎么回事', '怎么理解']
        self.speech_filler_mask = None

    def postprocess(self, pred):
        pred = pred.copy()

        self.speech_filler_mask = pred.apply(lambda d: self.filler_disturb(d, self.filler_words), axis=1)
        n_changed = (pred.loc[self.speech_filler_mask, 'pred'] == 0).sum()
        pred.loc[self.speech_filler_mask, 'pred'] = 1
        pred.loc[self.speech_filler_mask, 'prob0'] = 0.05
        pred.loc[self.speech_filler_mask, 'prob1'] = 0.95
        print('Speech Filler: ', self.speech_filler_mask.sum(), n_changed)

        return pred

    def filler_disturb(self, d: pd.Series, filler_words):
        q1, q2 = d['query1'], d['query2']
        if len(q1) > len(q2):
            short, long = q2, q1
        else:
            short, long = q1, q2

        if long.startswith(short) or long.endswith(short):
            for w in filler_words:
                if w in long.replace(short, ''):
                    return True

        return False


def main():
    syntactic_analysis = np.load('syntactic_analysis.npy', allow_pickle=True).item()

    # Input
    names = []
    if 'query' in args.input_cols: names.extend(['query1', 'query2'])
    if 'logit' in args.input_cols: names.extend(['logit0', 'logit1'])
    if 'probability' in args.input_cols: names.extend(['prob0', 'prob1'])
    names.append('pred')

    test_dataset = pd.read_table(args.input_path[0], names=names, na_filter=False, quoting=3)
    for p in args.input_path[1:]:
        d = pd.read_table(p, names=names, na_filter=False, quoting=3)
        test_dataset[['prob0', 'prob1']] += d[['prob0', 'prob1']]
    test_dataset[['prob0', 'prob1']] /= len(args.input_path)
    test_dataset['pred'] = test_dataset[['prob0', 'prob1']].values.argmax(1)

    test_dataset = CAN(alpha=args.can_alpha, iters=args.can_iters).postprocess(test_dataset)
    test_dataset = LexicalSemantics().postprocess(test_dataset)
    test_dataset = Misspelling().postprocess(test_dataset)

    test_dataset = Syntactic(syntactic_analysis).postprocess(test_dataset)
    test_dataset = Math().postprocess(test_dataset)
    test_dataset = SpeechFiller().postprocess(test_dataset)

    # Output
    output_cols = []
    if 'query' in args.output_cols: output_cols.extend(['query1', 'query2'])
    if 'logit' in args.output_cols: output_cols.extend(['logit0', 'logit1'])
    if 'probability' in args.output_cols: output_cols.extend(['prob0', 'prob1'])
    output_cols.append('pred')
    test_dataset = test_dataset[output_cols]
    test_dataset.to_csv(args.output_path, sep='\t', header=False, index=False, quoting=3, line_terminator='\n')


if __name__ == '__main__':
    main()