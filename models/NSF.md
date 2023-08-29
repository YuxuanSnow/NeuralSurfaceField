### Learn Neural Surface Field on Fusion Shape

##### 1. Learn NSF
```
python buff_nsf_male.py --exp_id 11 --batch_size 2 --split_file ./assets/data_split/buff_male_train_val.pkl --mode train --epochs 301
python buff_nsf_female.py --exp_id 12 --batch_size 2 --split_file ./assets/data_split/buff_female_train_val.pkl --mode train --epochs 301
```

##### 2. Inference
```
python buff_nsf_male.py --exp_id 11 --batch_size 1 --split_file ./assets/data_split/buff_male_train_val.pkl --mode test 
python buff_nsf_female.py --exp_id 12 --batch_size 1 --split_file ./assets/data_split/buff_female_train_val.pkl --mode test
```
Please adjust the hyperparameter for learning rate decay given the amount of samples in your training data. Traing too many epochs could cause undesired effects.
