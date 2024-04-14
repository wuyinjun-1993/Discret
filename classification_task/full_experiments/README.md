We could use the following command to run experiments with current code:
```
python full_experiments/full_cancer_experiment.py --batch_size 32 --learning_rate 0.001 --method ours --log_folder $log_dir --program_max_len 6 --model $model --rl_algorithm $RL_algorithm
```

in which, the argument "--batch_size", "--learning_rata" and "--program_max_len" are all tunable hyper-parameters. The variable $model can be either "transformer" or "mlp", which denotes how to encode the tabular feature data and the variable $RL_algorithm can be either "dqn" or "ppo". The default configurations of the models and the reinforcement learning algorithm can be found in the file "configs/configs.yaml". But any adjustments to this file are possible and the newer version of the configuration file (suppose named as "configs/configs_new.yaml")  could be loaded with the following command:

```
mkdir logs/
python full_experiments/full_cancer_experiment.py --batch_size 32 --learning_rate 0.001 --method ours --log_folder logs/ --program_max_len 6 --model mlp --rl_algorithm dqn --model_config "configs/configs_new.yaml"
```
