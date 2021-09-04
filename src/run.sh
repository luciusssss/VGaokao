python3 evidence_retrieval.py

python3 convert_to_training_data.py

python3 train_hinge.py \
--model_name_or_path ../models/roberta-large_ocnli \
--do_train \
--do_eval \
--train_file ../data/processed/train_nli_iterative-hinge.json \
--validation_file ../data/processed/test_nli_iterative-hinge.json \
--output_dir ../models/vgaokao_soft-masking \
--num_train_epochs 8 \
--max_seq_length 256 \
--per_gpu_train_batch_size 2 \
--per_gpu_eval_batch_size 2 \
--gradient_accumulation_steps 32 \
--learning_rate 2e-5 \
--save_steps 100000 \
--overwrite_output 


python3 inference.py \
--model_name_or_path ../models/vgaokao_soft-masking \
--do_predict \
--train_file ../data/processed/train_nli_iterative.json \
--validation_file ../data/processed/test_nli_iterative.json \
--output_dir ../models/vgaokao_soft-masking \
--num_train_epochs 4 \
--max_seq_length 256 \
--per_gpu_train_batch_size 4 \
--per_gpu_eval_batch_size 4 \
--gradient_accumulation_steps 16 \
--learning_rate 2e-5 \
--overwrite_output 

python3 calculate_mc_acc.py ../data/raw/test.json ../models/vgaokao_soft-masking/test_prob_None.txt ../models/vgaokao_soft-masking/test_mc_results.json



