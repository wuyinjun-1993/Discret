# Discret


## Training a transformer model as baseline (the implementation of the transformer model follows [this paper](https://arxiv.org/abs/2106.11959v2)):
```
python classification_task/full_experiments/pretrain_main.py --dataset_name cardio --batch_size 32 --log_folder logs/ --model transformer --learning_rate 5e-4
```
"--dataset_name cardio" means that we used a dataset called "cardio" which is stored in the "data/" folder by default and is downloaded from [kaggle](https://www.kaggle.com/datasets/sulianova/cardiovascular-disease-dataset?resource=download)

if you want to use your own data, you should go to [this function](https://github.com/wuyinjun-1993/Discret/blob/a3c2f1a39d3bf4312dcda411ee211a22c8573a8d/classification_task/datasets/EHR_datasets.py#L76) to specify the file name of the input data and the column names that represent the sample indexes and the labels (the variable "label_column" and "id_column"). 

Also, you need to go to [this line of code](https://github.com/wuyinjun-1993/Discret/blob/a3c2f1a39d3bf4312dcda411ee211a22c8573a8d/classification_task/full_experiments/rule_lang.py#L13) to specify which columns are categorical variables.


## Training our self-interpretable model which also used the above transformer model for encoding data:
```
python classification_task/full_experiments/main.py --dataset_name cardio --batch_size 32 --log_folder logs/ --model transformer --learning_rate 5e-4 --method ours --log_folder logs/ --num_ands 4 --model transformer --num_ors 3
```
THe output explanation of our method looks like this: (feat1 > 1 and feat2 < 2 and feat3 = 3) or (feat4 > 0 and feat5 == 6 and feat7 > 8). So "--num_ands" and "--num_ors" represent the number of logical and and logic or respectively.


## After the training proceess, we can use the following command to output the explanations for each test sample:
```
python classification_task/full_experiments/main.py --is_log --dataset_name cardio --batch_size 32 --log_folder logs/ --model transformer --learning_rate 5e-4 --method ours --log_folder logs/ --num_ands 4 --model transformer --num_ors 3
```

Then the output explanations can be found in the log file "logs/ours/logs.txt"
