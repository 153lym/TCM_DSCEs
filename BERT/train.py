import os

command = 'python bert_classifier.py --data_dir data ' \
          '--bert_model  bert-base-chinese ' \
          '--task_name tcm --output_dir result  ' \
          '--train_batch_size 1 --tag_space 0 --max_seq_length 32 --num_train_epochs 50 --rnn_hidden_size 128 '
#          '--do_resume --do_eval --do_lower_case --num_train_epochs 3 --rnn_hidden_size 128 --dropout 0.5 '

os.system(command)