# 整合数据
python data_preprocess.py \
    --train_set ../data/data118485/train.txt \
    --dev_set   ../data/data118485/dev.txt \
    --test_set  ../data/data118485/test.txt \
    --save_dir  ../data

# 初始训练
python train.py --train_set ../data/train.txt --dev_set ../data/dev.txt --save_dir ../data/checkpoints_new0 --max_seq_length 64 --train_batch_size 128 --model ErnieGram --learning_rate 1E-4 --layerwise_decay 0.8 --LSR_coef 0.1 --rdrop_coef 0 --clip_norm 1 --lookahead_k 5 --lookahead_alpha 0.5 --lazy_embedding --epochs 7 --amp --print_step 500
python predict.py --model ErnieGram --params_dir ../data/checkpoints_new0 --params_number last --output_info query logit probability label --score accuracy --input_file ../data/dev.txt --result_file ../user_data/dev_last.tsv

# 预测
python predict.py --model ErnieGram --output_info query logit probability --input_file ../data/test.txt --result_file ../user_data/test0.tsv --params_dir ../data/checkpoints_new0 --params_number last
python postprocess.py --input_path ../user_data/test0.tsv --input_cols query logit probability --output_path ../user_data/ccf_qianyan_qm_result_B0.csv
python postprocess.py --input_path ../user_data/test0.tsv --input_cols query logit probability --output_path ../user_data/test_A_pseudo0.txt --output_cols query logit probability


# 整合数据
python data_preprocess.py \
    --train_set ../data/data118485/train.txt \
    --pseudo_set ../user_data/test_A_pseudo0.txt \
    --dev_set   ../data/data118485/dev.txt \
    --test_set  ../data/data118485/test.txt \
    --save_dir  ../data

# 伪标签第一轮训练
python train.py --train_set ../data/train.txt --dev_set ../data/dev.txt --save_dir ../data/checkpoints_new1 --max_seq_length 64 --train_batch_size 128 --model ErnieGram --learning_rate 1E-4 --layerwise_decay 0.8 --LSR_coef 0.1 --rdrop_coef 6.4 --fgm_coef 0 --fgm_alpha 0 --clip_norm 1 --lookahead_k 5 --lookahead_alpha 0.5 --lazy_embedding --epochs 7 --amp --print_step 500
python predict.py --model ErnieGram --params_dir ../data/checkpoints_new1 --params_number last --output_info query logit probability label --score accuracy --input_file ../data/dev.txt --result_file ../user_data/dev_last.tsv

# 预测
python predict.py --model ErnieGram --output_info query logit probability --input_file ../data/test.txt --result_file ../user_data/test1.tsv --params_dir ../data/checkpoints_new1 --params_number last
python postprocess.py --input_path ../user_data/test1.tsv --input_cols query logit probability --output_path ../user_data/ccf_qianyan_qm_result_B1.csv
python postprocess.py --input_path ../user_data/test1.tsv --input_cols query logit probability --output_path ../user_data/test_A_pseudo1.txt --output_cols query logit probability


# 整合数据
python data_preprocess.py \
    --train_set ../data/data118485/train.txt \
    --pseudo_set ../user_data/test_A_pseudo1.txt \
    --dev_set   ../data/data118485/dev.txt \
    --test_set  ../data/data118485/test.txt \
    --save_dir  ../data

# 伪标签第二轮训练
python train.py --train_set ../data/train.txt --dev_set ../data/dev.txt --save_dir ../data/checkpoints_new2 --max_seq_length 64 --train_batch_size 128 --model ErnieGram --learning_rate 1E-4 --layerwise_decay 0.8 --LSR_coef 0.1 --rdrop_coef 0 --fgm_coef 2 --fgm_alpha 0.5 --clip_norm 1 --lookahead_k 5 --lookahead_alpha 0.5 --lazy_embedding --epochs 7 --amp --print_step 500
python predict.py --model ErnieGram --params_dir ../data/checkpoints_new2 --params_number last --output_info query logit probability label --score accuracy --input_file ../data/dev.txt --result_file ../user_data/dev_last.tsv

# 预测
python predict.py --model ErnieGram --output_info query logit probability --input_file ../data/test.txt --result_file ../user_data/test2.tsv --params_dir ../data/checkpoints_new2 --params_number last
python postprocess.py --input_path ../user_data/test2.tsv --input_cols query logit probability --output_path ../user_data/ccf_qianyan_qm_result_B2.csv
python postprocess.py --input_path ../user_data/test2.tsv --input_cols query logit probability --output_path ../user_data/test_A_pseudo2.txt --output_cols query logit probability


# 整合数据
python data_preprocess.py \
    --train_set ../data/data118485/train.txt \
    --pseudo_set ../user_data/test_A_pseudo2.txt \
    --dev_set   ../data/data118485/dev.txt \
    --test_set  ../data/data118485/test.txt \
    --save_dir  ../data

# 伪标签第三轮训练
python train.py --train_set ../data/train.txt --dev_set ../data/dev.txt --save_dir ../data/checkpoints_new3 --max_seq_length 64 --train_batch_size 128 --model ErnieGram --learning_rate 1E-4 --layerwise_decay 0.8 --LSR_coef 0.1 --rdrop_coef 0 --fgm_coef 2 --fgm_alpha 0.5 --clip_norm 1 --lookahead_k 5 --lookahead_alpha 0.5 --lazy_embedding --epochs 7 --amp --print_step 500
python predict.py --model ErnieGram --params_dir ../data/checkpoints_new3 --params_number last --output_info query logit probability label --score accuracy --input_file ../data/dev.txt --result_file ../user_data/dev_last.tsv

# 预测
python predict.py --model ErnieGram --output_info query logit probability --input_file ../data/test.txt --result_file ../user_data/test3.tsv --params_dir ../data/checkpoints_new3 --params_number last
python postprocess.py --input_path ../user_data/test3.tsv --input_cols query logit probability --output_path ../prediction_result/ccf_qianyan_qm_result_B.csv
python postprocess.py --input_path ../user_data/test3.tsv --input_cols query logit probability --output_path ../user_data/test_A_pseudo3.txt --output_cols query logit probability
