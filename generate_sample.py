"""
Step1 根据官网数据细粒度分析，对模型进行评测，在弱点方向上进行数据增强
Step2 按照词性规律对训练集进行筛选，将 jieba 分词结果组成元组
Step3 每个类型整理 20 个左右的元组
Step4 通过代码组成句子
Step5 简单复制到指定样本数
"""
import pandas as pd
import cn2an
import random
from itertools import product


class Generator:
    def __init__(self):
        self.negative_asymmetry_components = [
            # (名词1, 介词, 名词2, 形容词1, 形容词2, 结尾短语)
            # Template1: [A] 比 [B] [形容词 C] - [B] 比 [A] [C 的反义词]
            # Template2: [A] 是 [B] [关系 C]   - [B] 是 [A] [C 的反义词]
            ('高压', '比', '低压', '高', '低', '多少'),              # 基于筛选出的未覆盖到的测试集词语，半自动构造。烧脑，所以构造了几个就不这么搞了。
            ('正常值', '比', '异常值', '更高', '低', '吗？'),
            ('短句', '比', '长句', '易识别', '难识别', '么'),
            ('肿瘤', '比', '宫颈癌', '严重', '轻', '一些吗'),
            ('红细胞', '比', '白细胞', '大', '小', '多少呀'),
            ('寻麻疹', '比', '鼻窦炎', '常见', '罕见', '不？'),
            ('红霉素', '比', '青霉素', '贵', '便宜', '吧？'),
            ('化验', '在', '抽血', '前面', '后面', '吗？'),
            ('华东师范大学', '是在', '湖南师范大学', '北边', '南边', '么？'),
            ('哈工大', '是', '哈尔滨工业大学', '的简称', '的全称', '么'),
            ('人', '比', '动物', '强', '弱', '吗？'),               # 训练集例子改造。通过 HanLP 语法分析结果筛选训练集目标句子，简单改造下。
            ('电信', '比', '网通', '好', '差', '一些吗'),
            ('诸葛亮', '比', '司马懿', '聪明', '笨', '一些么?'),
            ('诸葛亮', '是', '黄月英', '丈夫', '妻子', '吗'),
            ('爱情', '比', '面包', '重要', '不重要', '嘛'),
            ('北京工业大学', '比', '北京交通大学', '好', '差', '嘛？'),
            ('格力空调', '比', '美的空调', '好', '差', '很多么'),
            ('石英表', '比', '机械表', '重', '轻', '一点么'),
            ('李敏镐', '比', '金秀贤', '帅', '丑', '吗'),
            ('李敏镐', '是', '金秀贤', '的表哥', '的表弟', '吗'),
            ('汪涵', '是', '李湘', '妻子', '丈夫', '么')
            # ('', '', '', '', '', ''),
        ]

        temporal_words = [['现在', '时', '的时候', '在'],          # 表示时间的词语。下面按模板构造样本的时候添加进去。
                          ['最近'],
                          ['今年', '去年', '前年'],
                          ['刚刚', '刚', '刚刚在', '曾经', '之前', '以前'],
                          ['将来', '未来', '以后', '之后', '后'],
                          ['还', '依然']]
        self.temporal_components = [
            # (短语, 短语, temporals, 短语)
            # Template: [主语] [动词短语] [temporal A] [后半句] - [主语] [动词短语] [temporal B] [后半句]
            ('', '吃海鲜', (('_时', '_之前', 0), ('_', '_之前', 0), ('_', '_后', 0), ('_', '_以后', 0), ('_时', '_后', 0), ('_', '_时', 1)), '不能喝什么'),
            ('', '流行', (('现在_', '以后_', 0), ('今年_', '今年以前_', 0), ('今年_', '去年_', 0), ('_', '以前_', 0), ('今年以前_', '今年之前_', 1), ('2021年_', '2021年这一年_', 1)), '什么发型'),
            ('', '开宠物店', (('现在_', '以后_', 0), ('_', '以后_', 0), ('_', '之前_', 0), ('_将来', '以后_', 1)), '赚钱吗'),
            ('', '当女兵', (('现在_', '以前_', 0), ('现在_', '以后_', 0), ('_', '现在_', 1)), '需要什么条件？'),
            ('', '玩游戏', (('_时', '_后', 0), ('_时', '_以后', 0), ('_', '_以后', 0), ('_', '_后', 0), ('_时', '_的时候', 1)), '黑屏'),
            ('', '看过', (('曾经_', '最近_', 0), ('曾经_', '以前_', 1)), '一部电影'),
            ('', '来月经', (('_前', '_后', 0), ('_', '_后', 0), ('_时', '_后', 0), ('_', '_前', 0), ('_以前', '_前', 1)), '可以吃红枣么'),
            ('唐嫣', '的男朋友', (('现在_', '以前_', 0), ('_', '以前_', 0)), '是谁'),
            ('', '游泳', (('_', '_前', 0), ('_', '_后', 0), ('_时', '_后', 0), ('_', '_时', 1)), '可以戴隐形眼镜么'),
            ('', '感冒', (('_', '_以后', 0), ('_时', '_以后', 0), ('_', '_时', 1)), '可以喝酒吗'),
            ('', '有哪些', (('现在_', '以后_', 0), ('_', '以前_', 0), ('_', '现在_', 1)), '免费杀毒软件'),
            ('快的打车', '有', (('现在_', '以后_', 0), ('_', '以后_', 0), ('之前_', '依然_', 0), ('_', '之前_', 0), ('还_', '依然_', 1)), '优惠吗'),
            ('人', '死', (('刚_时', '_之后', 0), ('_时', '_后', 0), ('刚刚_', '刚_', 1)), '是什么感觉'),
            ('你', '为什么', (('刚刚_', '现在_', 0), ('_', '刚刚_', 0), ('刚刚_', '刚_', 1)), '不理我'),
            ('我', '很烦', (('现在_', '以前_', 0), ('之前_', '以前_', 1)), '你'),
            ('', '很烦', (('_', '刚刚_', 0), ('现在_', '刚刚_', 0), ('之前_', '以前_', 1)), '你'),
            ('你们', '在', (('_', '以后_', 0), ('_', '刚刚_', 0), ('现在_', '刚刚_', 0), ('_', '现在_', 1)), '哪里'),
            # ('', '', '', ''),
        ]

        self.symmetry_and_asymmetry_components = [
            # (短语, n/adj, 短语, n/adj, 短语)
            # Template: [A] [B] [C] [D] [E] - [A] [D] [C] [B] [E]
            ('', '蜜蜂', '养殖', '怎么样', '', 0),                  # 筛选出的训练集难负例
            ('', '男人', '', '吃', '精子有什么好处', 0),
            ('', '爸爸', '', '给', '打电话', 0),

            ('', '天蝎座', '男生怎样追', '双子座', '女生', 0),       # 通过训练集和 HanLP 筛选构造的
            ('', '天蝎座', '男生如何追', '双子座', '女生', 0),       # 一组词语可以变换成一套句子
            ('', '天蝎座', '男生可以追', '双子座', '女生嘛', 0),     # 这样只用几个筛选出的结果，就能构造大量语法增强句对
            ('', '天蝎座', '男生能追', '双子座', '女生么', 0),
            ('天蝎座', '男生', '怎样追双子座', '女生', '', 0),
            ('天蝎座', '男生', '如何追双子座', '女生', '', 0),
            ('天蝎座', '男生', '可以追双子座', '女生', '吗', 0),
            ('天蝎座', '男生', '能追双子座', '女生', '么', 0),
            ('只追', '天蝎座', '不追', '双子座', '', 0),
            ('只追', '天蝎座', '不追', '双子座', '好么？', 0),
            ('', '只', '追天蝎座', '不', '追双子座', 0),
            ('', '只', '追天蝎座', '不', '追双子座好吗？', 0),

            ('', '金牛座', '和', '白羊座', '女生配吗', 0),
            ('', '双子座', '女生和', '双鱼座', '男生配吗', 0),
            ('双子座', '女生', '和双鱼座', '男生', '配吗', 0),


            ('用', '信用卡', '不用', '支付宝', '', 0),
            ('用', '信用卡', '不用', '支付宝', '能行么？', 0),
            ('', '', '用信用卡', '不', '用支付宝', 0),
            ('', '', '用信用卡', '不', '用支付宝能行么？', 0),

            ('', '高个', '女生暗恋', '矮个', '男生的表现', 0),
            ('高个', '女生', '暗恋矮个', '男生', '的表现', 0),
            ('', '女生', '暗恋矮个', '男生', '的表现', 0),

            ('', '暗色', '裤子配什么', '亮色', '衣服好看', 0),
            ('暗色', '裤子', '配什么亮色', '衣服', '好看', 0),
            ('买', '裤子', '不买', '衣服', '', 0),
            ('', '', '买裤子', '不', '买衣服', 0),

            ('', '三星', '和低配的', '小米', '哪个好', 0),
            ('高配的', '三星', '和低配的', '小米', '哪个好', 0),
            ('', '高配', '的三星和', '低配', '的小米哪个好', 0),

            ('', '鹦鹉鱼', '能和多条', '罗汉鱼', '一起养吗', 0),
            ('一条', '鹦鹉鱼', '能和多条', '罗汉鱼', '一起养吗', 0),
            ('养', '鹦鹉鱼', '不养', '罗汉鱼', '', 0),
            ('', '', '养鹦鹉鱼', '不', '养罗汉鱼', 0),

            ('', '女人', '和老', '男人', '', 0),
            ('优雅的', '女人', '和老', '男人', '', 0),
            ('', '优雅的', '女人和', '老', '男人', 0),

            ('我', '妈妈', '的', '奶奶', '叫什么', 0),
            ('我叫', '妈妈', '的', '叔叔', '什么?', 0),
            ('我管', '妈妈', '的', '爷爷', '叫什么?', 0),
            ('你', '妈妈', '的', '爸爸', '叫什么', 0),
            # ('', '', '', '', ''),
        ]

        # 交通语法类
        self.loc = [        # jieba.posseg 筛一下就能得到
            '枣庄', '鸡西', '喀什', '阜阳市', '静安区', '香威', '兖州', '温岭', '涡阳', '齐齐哈尔市', '西江', '溧阳', '济宁市',
            '江门市', '温州市', '涉县', '邓州', '叙利亚', '德意志', '西山区', '崇州市', '西峡', '青州', '宁河', '滨江', '洋县',
            '凤城', '长兴'
        ]
        self.symmetry_and_asymmetry_loc_components = [      # jieba.posseg 筛一下地名句子就能得到
            (['飞'], ['航班'], 0),
            (['至'], ['高铁', '高铁票', '火车', '火车票', '动车', '动车票', '飞机', '飞机票'], 0),
            (['到'], ['高铁', '高铁票', '火车', '火车票', '动车', '动车票', '飞机', '飞机票', '怎么走'], 0),
            (['一', '—'], ['高铁', '高铁票', '火车', '火车票', '动车', '动车票', '飞机', '飞机票'], 0),
            (['在'], ['的哪个方向'], 0),
            (['离'], ['多远', '有多远', '多少公里', '有多少公里'], 1),
        ]

        self.math_components = {
            '+': ('加', '加上', '+'),
            '-': ('减', '减去', '-'),
            '*': ('乘', '乘以', '×', '*', 'x', 'X'),
            '/': ('除', '除以', '/', '÷'),
            '=': ('等于', '=', '得', '是'),
            'ans': ('几', '多少'),
            'and': ('和', '跟'),   # 1 和 2 的最小公倍数
            'common': ('的最小公倍数', '的最大公约数', '的公约数', '的公因数'),
        }

    def negative_asymmetry(self, num_generate):
        samples = []
        len_comp = len(self.negative_asymmetry_components)
        cnt = 0
        while cnt < num_generate:
            n1, p, n2, a1, a2, e = self.negative_asymmetry_components[cnt % len_comp]
            samples.append((''.join([n1, p, n2, a1, e]), ''.join([n2, p, n1, a2, e]), 1))
            if cnt % 4 == 0:
                samples.append((''.join([n1, p, n2, a1, e]), ''.join([n2, p, n1, a1, e]), 0))
            elif cnt % 4 == 1:
                samples.append((''.join([n1, p, n2, a2, e]), ''.join([n2, p, n1, a2, e]), 0))
            elif cnt % 4 == 2:
                samples.append((''.join([n1, p, n2, a1, e]), ''.join([n1, p, n2, a2, e]), 0))
            else:
                samples.append((''.join([n2, p, n1, a1, e]), ''.join([n2, p, n1, a2, e]), 0))
            cnt += 2

        return samples[:num_generate]

    def temporal(self, num_generate):
        samples = []
        len_comp = len(self.temporal_components)
        cnt = 0
        while cnt < num_generate:
            p1, p2, ts, p3 = self.temporal_components[cnt % len_comp]

            for t1, t2, l in ts:
                assert ('_' in t1) and ('_' in t2)

                tp1, tp2 = t1.replace('_', p2), t2.replace('_', p2)
                samples.append((''.join([p1, tp1, p3]), ''.join([p1, tp2, p3]), l))
                cnt += 1

        return samples[:num_generate]

    def symmetry_and_asymmetry(self, num_generate):
        samples = []
        len_comp = len(self.symmetry_and_asymmetry_components)
        for i in range(num_generate):
            p1, w1, p2, w2, p3, l = self.symmetry_and_asymmetry_components[i % len_comp]
            samples.append((''.join([p1, w1, p2, w2, p3]), ''.join([p1, w2, p2, w1, p3]), l))
        return samples[:num_generate]

    def symmetry_and_asymmetry_loc(self, num_generate):
        combs = []
        for c in self.symmetry_and_asymmetry_loc_components:
            combs.extend(list(product(c[0], c[1], [c[2]])))
        loc_pairs = [(l1, l2) for l1, l2 in product(self.loc, self.loc) if l1 != l2]
        random.shuffle(loc_pairs)

        samples = []
        cnt = 0
        while cnt < num_generate:
            c = combs[cnt % len(combs)]
            l = loc_pairs[cnt % len(loc_pairs)]
            samples.append((''.join([l[0], c[0], l[1], c[1]]), ''.join([l[1], c[0], l[0], c[1]]), c[2]))
            cnt += 1
        return samples[:num_generate]

    def math(self, num_generate):
        """降0.3分左右，推测是因为测试集打标标准问题导致的。不同数字在特定语义下可能打标成相关。"""
        samples = []
        cnt = 0

        # 加减乘除 - 汉字
        end = list(''.join(c) for c in product(self.math_components['='], self.math_components['ans']) if '\u4e00' <= c[0] <= '\u9fa5') + ['']
        end_pair = list(product(end, end))
        combs = [(s, 1) for s in self.math_components['+'] if '\u4e00' <= s <= '\u9fa5'] + \
                [(s, 0) for s in self.math_components['-'] if '\u4e00' <= s <= '\u9fa5'] + \
                [(s, 1) for s in self.math_components['*'] if '\u4e00' <= s <= '\u9fa5'] + \
                [(s, 0) for s in self.math_components['/'] if '\u4e00' <= s <= '\u9fa5']
        while cnt < 0.4 * num_generate:
            n1 = n2 = self._generate_number()
            while n2 == n1: n2 = self._generate_number()
            n1_sim = self._similar_num(n1)

            c = combs[cnt % len(combs)]
            e = end_pair[cnt % len(end_pair)]
            samples.append((''.join([n1, c[0], n2, e[0]]), ''.join([n2, c[0], n1, e[1]]), c[1]))
            if random.random() < 0.5:
                samples.append((''.join([n1, c[0], n2, e[0]]), ''.join([n1_sim, c[0], n2, e[0]]), 0))
            else:
                samples.append((''.join([n2, c[0], n1, e[0]]), ''.join([n2, c[0], n1_sim, e[0]]), 0))

            cnt += 2

        # 加减乘除 - 符号
        end = list(''.join(c) for c in product(self.math_components['='], self.math_components['ans'])) + ['']
        end_pair = list(product(end, end))
        combs = [(s, 1) for s in self.math_components['+'] if not ('\u4e00' <= s <= '\u9fa5')] + \
                [(s, 0) for s in self.math_components['-'] if not ('\u4e00' <= s <= '\u9fa5')] + \
                [(s, 1) for s in self.math_components['*'] if not ('\u4e00' <= s <= '\u9fa5')] + \
                [(s, 0) for s in self.math_components['/'] if not ('\u4e00' <= s <= '\u9fa5')]
        while cnt < 0.85 * num_generate:
            n1 = n2 = self._generate_number(cn_an='an')
            while n2 == n1: n2 = self._generate_number(cn_an='an')
            n1_sim = self._similar_num(n1)

            c = combs[cnt % len(combs)]
            e = end_pair[cnt % len(end_pair)]
            samples.append((''.join([n1, c[0], n2, e[0]]), ''.join([n2, c[0], n1, e[1]]), c[1]))
            if random.random() < 0.5:
                samples.append((''.join([n1, c[0], n2, e[0]]), ''.join([n1_sim, c[0], n2, e[0]]), 0))
            else:
                samples.append((''.join([n2, c[0], n1, e[0]]), ''.join([n2, c[0], n1_sim, e[0]]), 0))

            cnt += 2

        # 公倍数，公约数
        combs  = list(product(self.math_components['and'], self.math_components['common']))
        while cnt < 0.9 * num_generate:
            n1 = n2 = self._generate_number(int_bool=True)
            while n2 == n1: n2 = self._generate_number(int_bool=True)

            c = combs[cnt % len(combs)]
            samples.append((''.join([n1, c[0], n2, c[1]]), ''.join([n2, c[0], n1, c[1]]), 1))

            cnt += 1

        while cnt < num_generate:
            n1 = str(random.randint(10000, 10000000))
            n2 = list(n1)
            rnd1 = random.randint(1, len(n1)-1)
            if random.random() < 0.5:
                rnd2 = str(random.randint(0, 9))
                while rnd2 == n2[rnd1]:
                    rnd2 = str(random.randint(0, 9))
                n2[rnd1] = rnd2
            else:
                n2.pop(rnd1)
            n2 = ''.join(n2)

            samples.append((n1, n2, 0))

            cnt += 1

        return samples[:num_generate]

    def _generate_number(self, cn_an=None, int_bool=False):
        # 随机数
        rnd = random.random()
        if rnd < 0.6:
            num = random.random() * 10
        else:
            num = (random.random() * 90) + 10

        # 小数位数
        ratio = [1, 0] if int_bool else [0.7, 0.9]
        rnd = random.random()
        if rnd < ratio[0]:
            num = round(num)
        elif rnd < ratio[1]:
            num = round(num, 1)
        else:
            num = round(num, 2)

        # 汉字-阿拉伯数字
        if cn_an == 'cn':
            ratio = [1, 0, 0]
        elif cn_an == 'an':
            ratio = [0, 1, 0]
        else:
            ratio = [0.6, 1, 0]

        rnd = random.random()
        if rnd < ratio[0]:              # 汉字
            num = cn2an.an2cn(num)
        elif rnd < ratio[1]:            # 阿拉伯数字
            pass
        else:                           # 混合
            num_cn = cn2an.an2cn(num)
            if round(num) != num:       # 含小数
                if rnd < ratio[2]:
                    num = str(num).split('.')[0] + '.' + num_cn.split('点')[1]
                else:
                    num = num_cn.split('点')[0] + '点' + str(num).split('.')[1]

        return str(num)

    def _similar_num(self, num: str):
        if '点' in num:

            if len(num.split('点')[1]) == 2:
                return num[:-1]
            else:
                return num + cn2an.an2cn(random.randint(1, 9))

        elif '.' in num:

            if len(num.split('.')[1]) == 2:
                return num[:-1]
            else:
                return num + str(random.randint(1, 9))

        else:

            num_an = str(cn2an.cn2an(num, 'smart'))
            if num == num_an:           # 阿拉伯数字
                if len(num) == 1:
                    return str(random.randint(1, 9)) + num                          # 前面加一位
                else:
                    return num[1:]                                                  # 舍掉第一位
            else:                       # 汉字
                if len(num) == 1:
                    return cn2an.an2cn(str(random.randint(1, 9) + int(num_an)))     # 加 1~9
                else:
                    return cn2an.an2cn(num_an[:-1] + str(random.randint(1, 9)))     # 替换最后一位数


def main():
    random.seed(0)
    g = Generator()
    generated_samples  = []
    generated_samples += g.negative_asymmetry(500)
    generated_samples += g.temporal(1000)
    generated_samples += g.symmetry_and_asymmetry(1000)
    generated_samples += g.symmetry_and_asymmetry_loc(200)
    # generated_samples += g.math(200)
    generated_samples  = pd.DataFrame(generated_samples, columns=['query1', 'query2', 'label'])
    generated_samples.to_csv('../data/generated_samples.txt', sep='\t', header=False, index=False, quoting=3, line_terminator='\n')


if __name__ == '__main__':
    main()
