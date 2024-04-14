# Discret


## Training a transformer model as baseline (the implementation of the transformer model follows [this paper](https://arxiv.org/abs/2106.11959v2)):
```
python classification_task/full_experiments/pretrain_main.py --dataset_name cardio --batch_size 32 --log_folder logs/ --model transformer --learning_rate 5e-4
```
"--dataset_name cardio" means that we used a dataset called "cardio"


## Training our self-interpretable model which also used the above transformer model for encoding data:
```
python classification_task/full_experiments/main.py --dataset_name cardio --batch_size 32 --log_folder logs/ --model transformer --learning_rate 5e-4 --method ours --log_folder logs/ --program_max_len 4 --model transformer --topk_act 3
```

