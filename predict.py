from functools import partial
import argparse
import sys
import os
import random
import time

import numpy as np
import paddle
import paddle.nn.functional as F
import paddlenlp as ppnlp
from paddlenlp.datasets import load_dataset
from paddlenlp.data import Stack, Tuple, Pad

from data import create_dataloader, read_text_pair, convert_example
from model import QuestionMatching

# yapf: disable
parser = argparse.ArgumentParser()

# 路径
parser.add_argument("--input_file",     type=str, required=True, help="The full path of input file")
parser.add_argument("--result_file",    type=str, required=True, help="The result file name")
parser.add_argument("--params_dir",     type=str, required=True, help="The directory to model parameters to be loaded.")
parser.add_argument("--params_number",  type=str, nargs="+", required=True, help="使用的模型参数编号。输入多个则做参数平均。")

# 数据
parser.add_argument("--max_seq_length", default=64, type=int, help="序列对最大长度和")
parser.add_argument("--batch_size",     default=128, type=int, help="Batch size per GPU/CPU for training.")
parser.add_argument("--use_pinyin",     action='store_true', default=False, help="是否使用拼音特征")

# 模型
parser.add_argument("--model",          default='ErnieGram', type=str, help="选择模型")
parser.add_argument("--dropout_coef",   default=0.1, type=float, help="模型倒数第二个 FC 层的 dropout 比率")
parser.add_argument("--absolute_position_aware", action='store_true', default=False, help="绝对位置感知")

# 输出
parser.add_argument("--output_info",    default=[], type=str, nargs='+', help="可以输出 query, logit, probability, label")
parser.add_argument("--score",          type=str, help="可以输出 accuracy")

# 硬件
parser.add_argument('--device',         choices=['cpu', 'gpu'], default="gpu", help="Select which device to train model, defaults to gpu.")

args = parser.parse_args()
# yapf: enable


def predict(model, data_loader):
    batch_logits = []

    model.eval()

    with paddle.no_grad():
        for batch in data_loader:
            input_ids, token_type_ids, position_ids, pinyin_ids, pinyin_type_ids = batch

            # input_ids = paddle.to_tensor(input_ids)
            # token_type_ids = paddle.to_tensor(token_type_ids)

            batch_logit, _ = model(input_ids=input_ids, token_type_ids=token_type_ids, position_ids=position_ids, pinyin_ids=pinyin_ids, pinyin_type_ids=pinyin_type_ids)

            batch_logits.append(batch_logit.numpy())

        batch_logits = np.concatenate(batch_logits, axis=0)

        return batch_logits


def load_model(model):
    params_paths = [os.path.join(args.params_dir, f"model_{num}/model_state.pdparams") for num in args.params_number]
    if params_paths and all(os.path.isfile(p) for p in params_paths):
        # 单个路径直接读取预测，多个路径就参数平均，然后预测。
        n_ps = len(params_paths)
        coefs = np.array([0.1 * 0.9 ** (n_ps - i - 1) for i in range(n_ps)])
        coefs /= coefs.sum()
        state_dict = dict()
        for i, p in enumerate(params_paths):
            state_dict_tmp = paddle.load(p)
            for param_name in state_dict_tmp.keys():
                state_dict[param_name] = state_dict.get(param_name, 0) + coefs[i] * state_dict_tmp[param_name]
            del state_dict_tmp

        model.set_dict(state_dict)
        print("Loaded parameters from %s" % params_paths)
    else:
        raise ValueError(
            "Please set --params_paths with correct pretrained model file")


def collect_and_output(y_logits):
    output = []
    y_preds = np.argmax(y_logits, axis=1)

    if 'query' in args.output_info:
        with open(args.input_file, 'r', encoding='utf-8') as f:
            query_pair = [line.strip().split('\t')[:2] for line in f]
        output.extend(list(zip(*query_pair)))

    if 'logit' in args.output_info:
        output.append(list(y_logits[:, 0]))
        output.append(list(y_logits[:, 1]))

    if 'probability' in args.output_info:
        logit_exp = np.exp(y_logits)
        prob = logit_exp / np.sum(logit_exp, axis=1, keepdims=True)
        output.append(list(prob[:, 0]))
        output.append(list(prob[:, 1]))

    output.append(list(y_preds))

    if 'label' in args.output_info or args.score == 'accuracy':
        with open(args.input_file, 'r', encoding='utf-8') as f:
            labels = [int(line.split('\t')[2]) for line in f]

    if 'label' in args.output_info:
        output.append(labels)

    # 打印准确率
    if args.score == 'accuracy':
        acc = (np.array(y_preds) == np.array(labels)).mean()
        print(f"准确率：{acc}")

    # 写入磁盘
    with open(args.result_file, 'w', encoding="utf-8") as f:
        for d in zip(*output):
            f.write('\t'.join(map(str, d)) + "\n")


if __name__ == "__main__":
    paddle.set_device(args.device)

    if args.model == 'ErnieGram':
        pretrained_model = ppnlp.transformers.ErnieGramModel.from_pretrained('ernie-gram-zh')
        tokenizer        = ppnlp.transformers.ErnieGramTokenizer.from_pretrained('ernie-gram-zh')
    elif args.model == 'RobertaLarge':
        pretrained_model = ppnlp.transformers.RobertaModel.from_pretrained('roberta-wwm-ext-large')
        tokenizer        = ppnlp.transformers.RobertaTokenizer.from_pretrained('roberta-wwm-ext-large')
    else:
        raise Exception()

    trans_func = partial(
        convert_example,
        tokenizer=tokenizer,
        max_seq_length=args.max_seq_length,
        absolute_position_aware=args.absolute_position_aware,
        mode='test')

    batchify_fn = lambda samples, fn=Tuple(
        Pad(axis=0, pad_val=tokenizer.pad_token_id),        # input ids
        Pad(axis=0, pad_val=tokenizer.pad_token_type_id),   # input type ids
        Pad(axis=0, pad_val=tokenizer.pad_token_id),        # position ids
        Pad(axis=0, pad_val=tokenizer.pad_token_id),        # pinyin ids
        Pad(axis=0, pad_val=tokenizer.pad_token_type_id),   # pinyin type ids
    ): [data for data in fn(samples)]

    test_ds = load_dataset(
        read_text_pair, data_path=args.input_file, mode='test', lazy=False)

    test_data_loader = create_dataloader(
        test_ds,
        mode='test',
        batch_size=args.batch_size,
        batchify_fn=batchify_fn,
        trans_fn=trans_func)

    model = QuestionMatching(pretrained_model, dropout=args.dropout_coef, use_pinyin=args.use_pinyin)
    load_model(model)

    y_logits = predict(model, test_data_loader)

    # 收集输出内容
    collect_and_output(y_logits)