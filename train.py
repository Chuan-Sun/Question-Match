from functools import partial
import argparse
import os
import math
import random
import time

import numpy as np
import paddle
import paddle.nn.functional as F

import paddlenlp as ppnlp
from paddlenlp.data import Stack, Tuple, Pad
from paddlenlp.datasets import load_dataset
from paddlenlp.transformers import LinearDecayWithWarmup
from paddlenlp.ops.optimizer import AdamWDL

from data import create_dataloader, read_text_pair, convert_example
from model import QuestionMatching, FGM, LazyEmbedding
from loss import Loss
from lookahead import Lookahead


# yapf: disable
parser = argparse.ArgumentParser()

# 路径
parser.add_argument("--train_set",         type=str, required=True, help="The full path of train_set_file")
parser.add_argument("--dev_set",           type=str, required=True, help="The full path of dev_set_file")
parser.add_argument("--save_dir",          default='./checkpoint', type=str, help="The output directory where the model checkpoints will be written.")

# 数据
parser.add_argument("--max_seq_length",    default=128,   type=int,   help="The maximum total input sequence length after tokenization. Sequences longer than this will be truncated, sequences shorter will be padded.")
parser.add_argument("--train_batch_size",  default=32,    type=int,   help="Batch size per GPU/CPU for training.")
parser.add_argument("--eval_batch_size",   default=128,   type=int,   help="Batch size per GPU/CPU for training.")
parser.add_argument("--aug_exchange",      default=0.0,   type=float, help="数据增强：调换句对概率")
parser.add_argument("--use_pinyin",        action='store_true', default=False, help="是否使用拼音特征")

# 模型
parser.add_argument("--model",             default='ErnieGram', type=str, help="选择模型")
parser.add_argument("--dropout_coef",      default=0.1,   type=float, help="模型倒数第二个 FC 层的 dropout 比率")
parser.add_argument("--absolute_position_aware", action='store_true', default=False, help="绝对位置感知")
parser.add_argument("--init_from_ckpt",    default=None,  type=str,   help="The path of checkpoint to be loaded.")

# 学习率
parser.add_argument("--learning_rate",     default=1e-4,  type=float, help="The initial learning rate for Adam.")
parser.add_argument("--layerwise_decay",   default=1.0,   type=float, help="The layer-wise decay ratio.")
parser.add_argument("--warmup_proportion", default=0.0,   type=float, help="Linear warmup proption over the training process.")

# Loss
parser.add_argument("--LSR_coef",          default=0.0,   type=float, help="Label smooth regularization")
parser.add_argument("--hinge_coef",        default=0.0,   type=float, help="Loss 低于这个值就归零，这个功能和 r_drop 的兼容性可能有问题")
parser.add_argument("--focal_coef",        default=0.0,   type=float, help="Focal Loss 的 gamma")
parser.add_argument("--rdrop_coef",        default=0.0,   type=float, help="The coefficient of KL-Divergence loss in R-Drop paper, for more detail please refer to https://arxiv.org/abs/2106.14448), if rdrop_coef > 0 then R-Drop works")
parser.add_argument("--rdrop_buffer",      action='store_true', default=False, help="是否乘上缓冲系数")
parser.add_argument("--contrastive_coef",  default=0.0,   type=float, help="对比损失函数比重")
parser.add_argument("--fgm_coef",          default=0.0,   type=float, help="对抗训练")
parser.add_argument("--fgm_alpha",         default=0.0,   type=float, help="对抗训练 Loss 比重")
parser.add_argument("--self_dist",         action='store_true', default=False, help='自蒸馏')

# Gradient
parser.add_argument("--weight_decay",      default=0.0,   type=float, help="Weight decay if we apply some.")
parser.add_argument("--clip_norm",         default=0.0,   type=float, help="梯度裁剪")
parser.add_argument("--lookahead_k",       default=0,     type=int,   help="lookahead 的参数 k")
parser.add_argument("--lookahead_alpha",   default=0.0,   type=float,   help="lookahead 的参数 alpha")
parser.add_argument("--lazy_embedding",    action='store_true', default=False, help="没有出现的 token 是否更新")
parser.add_argument("--radam",             action='store_true', default=False, help="是否使用 Rectified Adam")

# 训练
parser.add_argument("--epochs",            default=5,     type=int,   help="Total number of training epochs to perform.")
parser.add_argument("--max_steps",         default=-1,    type=int,   help="If > 0, set total number of training steps to perform.")
parser.add_argument('--amp',               action='store_true', default=False, help="混合精度训练")
parser.add_argument("--seed",              default=1000,  type=int,   help="Random seed for initialization.")

# 验证
parser.add_argument("--eval",              action='store_true', default=False, help="做不做验证")
parser.add_argument("--eval_step",         default=1000,  type=int,   help="Step interval for evaluation.")
parser.add_argument("--print_step",        default=1000,  type=int,   help="Step interval for print train information.")
parser.add_argument('--save_step',         default=100,  type=int,   help="Step interval for saving checkpoint.")

# 硬件
parser.add_argument('--device',            choices=['cpu', 'gpu'], default="gpu", help="Select which device to train model, defaults to gpu.")
parser.add_argument("--num_gpu",           default=1,     type=int,   help="GPU 数量")

args = parser.parse_args()
# yapf: enable


def set_seed(seed):
    """sets random seed"""
    random.seed(seed)
    np.random.seed(seed)
    paddle.seed(seed)


def get_pretrained_model():
    if args.model == 'ErnieGram':
        pretrained_model = ppnlp.transformers.ErnieGramModel.from_pretrained('ernie-gram-zh')
        tokenizer        = ppnlp.transformers.ErnieGramTokenizer.from_pretrained('ernie-gram-zh')
    elif args.model == 'RobertaLarge':
        pretrained_model = ppnlp.transformers.RobertaModel.from_pretrained('roberta-wwm-ext-large')
        tokenizer        = ppnlp.transformers.RobertaTokenizer.from_pretrained('roberta-wwm-ext-large')
    else:
        raise
    return pretrained_model, tokenizer


def get_data_loader(tokenizer):
    train_ds = load_dataset(read_text_pair, data_path=args.train_set, mode='train', lazy=False)
    dev_ds   = load_dataset(read_text_pair, data_path=args.dev_set,   mode='dev',   lazy=False)

    trans_func = partial(
        convert_example,
        tokenizer=tokenizer,
        max_seq_length=args.max_seq_length,
        absolute_position_aware=args.absolute_position_aware,
        aug_exchange=args.aug_exchange)

    batchify_fn = lambda samples, fn=Tuple(
        Pad(axis=0, pad_val=tokenizer.pad_token_id),        # text id
        Pad(axis=0, pad_val=tokenizer.pad_token_type_id),   # text type id
        Pad(axis=0, pad_val=tokenizer.pad_token_id),        # position id
        Pad(axis=0, pad_val=tokenizer.pad_token_id),        # pinyin id
        Pad(axis=0, pad_val=tokenizer.pad_token_type_id),   # pinyin type id
        Stack(dtype='float32'),         # logit
        Stack(dtype='float32'),         # prob
        Stack(dtype="int64"),           # label  0/1
        Stack(dtype="int64"),           # pseudo 0/1
        Stack(dtype="float32"),         # weight
    ): [data for data in fn(samples)]

    train_data_loader = create_dataloader(
        train_ds,
        mode='train',
        batch_size=args.train_batch_size,
        batchify_fn=batchify_fn,
        trans_fn=partial(trans_func, mode='train'))

    dev_data_loader = create_dataloader(
        dev_ds,
        mode='dev',
        batch_size=args.eval_batch_size,
        batchify_fn=batchify_fn,
        trans_fn=partial(trans_func, mode='dev'))
    return train_data_loader, dev_data_loader


def get_optimizer(model, lr_scheduler):
    # Generate parameter names needed to perform weight decay.
    # All bias and LayerNorm parameters are excluded.
    decay_params = [
        p.name for n, p in model.named_parameters()
        if not any(nd in n for nd in ["bias", "norm"])
    ]

    name_dict = dict()
    for n, p in model.named_parameters():
        name_dict[p.name] = n

    clip = paddle.nn.ClipGradByNorm(clip_norm=args.clip_norm) if args.clip_norm > 0 else None

    if args.model == 'ErnieGram':
        n_layers = 12
    elif args.model == 'RobertaLarge':
        n_layers = 24
    else:
        raise

    optim = RAdam if args.radam else AdamWDL

    optimizer = optim(
        learning_rate=lr_scheduler,
        parameters=model.parameters(),
        weight_decay=args.weight_decay,
        apply_decay_param_fun=lambda x: x in decay_params,
        grad_clip=clip,
        layerwise_decay=args.layerwise_decay,
        n_layers=n_layers,
        name_dict=name_dict)
    return optimizer


def train_one_batch(batch, model, criterion, metric, optimizer, look_ahead, lazy_embedding, lr_scheduler, scaler, fgm, global_step):
    input_ids, token_type_ids, position_ids, pinyin_ids, pinyin_type_ids, logit_pseudo, prob_pseudo, label, pseudo, weight = batch
    logit1, logit2, dist_logit = None, None, None

    # Logit1
    with paddle.amp.auto_cast(enable=args.amp):
        logit1, feat = model(input_ids=input_ids, token_type_ids=token_type_ids, position_ids=position_ids, pinyin_ids=pinyin_ids, pinyin_type_ids=pinyin_type_ids)

    # Logit2: R-Drop
    if args.rdrop_coef > 0:
        with paddle.amp.auto_cast(enable=args.amp):
            logit2, feat = model(input_ids=input_ids, token_type_ids=token_type_ids, position_ids=position_ids, pinyin_ids=pinyin_ids, pinyin_type_ids=pinyin_type_ids)

    # 记录指标
    correct = metric.compute(logit1, label)
    metric.update(correct)
    acc = metric.accumulate()

    # Loss
    with paddle.amp.auto_cast(enable=args.amp):
        loss, ce_loss, kl_loss = criterion(logit1, logit2, feat, label, dist_logit, global_step, weight)
        if args.fgm_coef > 0: loss *= (1 - args.fgm_alpha)
    scaled = scaler.scale(loss)
    scaled.backward()

    # 对抗训练
    if args.fgm_coef > 0:
        fgm.attack(args.fgm_coef)
        with paddle.amp.auto_cast(enable=args.amp):
            logit2, feat = model(input_ids=input_ids, token_type_ids=token_type_ids, position_ids=position_ids, pinyin_ids=pinyin_ids, pinyin_type_ids=pinyin_type_ids)
            loss_fgm, ce_loss_fgm, _ = criterion(logit2, None, feat, label, dist_logit, global_step, weight)
            loss_fgm *= args.fgm_alpha
        loss += loss_fgm
        scaled = scaler.scale(loss_fgm)
        scaled.backward()
        fgm.restore()

    scaler.minimize(optimizer, scaled)
    if args.lazy_embedding:
        lazy_embedding.stop_update()
    lr_scheduler.step()
    if args.lookahead_k > 1:
        look_ahead.step()
    optimizer.clear_grad()

    return loss, ce_loss, kl_loss, acc


@paddle.no_grad()
def evaluate(model, criterion, metric, data_loader):
    model.eval()
    metric.reset()
    losses = []
    total_num = 0

    for batch in data_loader:
        input_ids, token_type_ids, labels = batch
        total_num += len(labels)
        logits, _, _ = model(input_ids=input_ids, token_type_ids=token_type_ids, do_evaluate=True)
        correct = metric.compute(logits, labels)
        metric.update(correct)
        accu = metric.accumulate()

        if args.LSR_coef > 0:
            labels_smooth = F.one_hot(labels.squeeze(1), num_classes=2)
            labels_smooth = F.label_smooth(labels_smooth, epsilon=args.LSR_coef)
            labels = labels_smooth

        loss = criterion(logits, labels)
        if args.hinge_coef > 0: loss = loss.mean()
        losses.append(loss.numpy())

    print("dev_loss: {:.5}, accuracy: {:.5}, total_num:{}".format(np.mean(losses), accu, total_num))
    model.train()
    metric.reset()
    return accu


def do_train():
    paddle.set_device(args.device)
    rank = paddle.distributed.get_rank()
    if paddle.distributed.get_world_size() > 1:
        paddle.distributed.init_parallel_env()

    set_seed(args.seed)

    pretrained_model,  tokenizer       = get_pretrained_model()
    train_data_loader, dev_data_loader = get_data_loader(tokenizer)

    model = QuestionMatching(pretrained_model, dropout=args.dropout_coef, use_pinyin=args.use_pinyin)
    if args.init_from_ckpt and os.path.isfile(args.init_from_ckpt):
        state_dict = paddle.load(args.init_from_ckpt)
        model.set_dict(state_dict)
    model = paddle.DataParallel(model)
    fgm = FGM(model) if args.fgm_coef > 0 else None

    num_training_steps = len(train_data_loader) * args.epochs // args.num_gpu
    lr_scheduler       = LinearDecayWithWarmup(args.learning_rate, num_training_steps, args.warmup_proportion)
    optimizer          = get_optimizer(model, lr_scheduler)
    look_ahead         = Lookahead(model, k=args.lookahead_k, alpha=args.lookahead_alpha) if args.lookahead_k > 1 else None
    lazy_embedding     = LazyEmbedding(model) if args.lazy_embedding else None
    scaler             = paddle.amp.GradScaler(args.amp)
    criterion          = Loss(args, num_training_steps)
    metric             = paddle.metric.Accuracy()

    global_step, best_accuracy = 0, 0.0
    tic_train = time.time()
    for epoch in range(1, args.epochs + 1):
        for step, batch in enumerate(train_data_loader, start=1):
            loss, ce_loss, kl_loss, acc = train_one_batch(batch, model, criterion, metric, optimizer, look_ahead, lazy_embedding, lr_scheduler, scaler, fgm, global_step)

            # 打印训练信息
            global_step += 1
            if global_step % args.print_step == 0:
                print(f"global step {global_step}, epoch: {epoch}, batch: {step}, loss: {float(loss):.4f}, "
                      f"ce_loss: {float(ce_loss):.4f}, kl_loss: {float(kl_loss):.4f}, accu: {acc:.4f}, "
                      f"speed: {args.print_step / (time.time() - tic_train):.2f} step / s")
                tic_train = time.time()

            # Eval
            if args.eval and global_step % args.eval_step == 0:
                accuracy = evaluate(model, criterion, metric, dev_data_loader)
                if accuracy > best_accuracy:
                    best_accuracy = accuracy
                    print(f"best global step: {global_step}, best accuracy: {best_accuracy:.4f}")

            # 最大 Step 限制
            if global_step == args.max_steps:
                return

        metric.reset()

    save_dir = os.path.join(args.save_dir, f'model_last/')
    if not os.path.exists(save_dir): os.makedirs(save_dir)
    save_param_path = os.path.join(save_dir, 'model_state.pdparams')
    paddle.save(model.state_dict(), save_param_path)
    tokenizer.save_pretrained(save_dir)
    print(f'save model_last')


if __name__ == "__main__":
    do_train()
