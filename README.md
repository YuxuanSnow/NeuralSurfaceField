# NSF: Neural Surface Field for Human Modeling from Monocular Depth
#### [Project Page](yuxuan-xue.com/nsf) | [Video](PlaceHolder) | [Paper](PlaceHolder)

In ICCV 2023, Paris

[Yuxuan Xue](https://yuxuan-xue.com/)<sup>1, *</sup>, [Bharat Lal Bhatnagar](https://virtualhumans.mpi-inf.mpg.de/people/Bhatnagar.html)<sup>2,*</sup>, [Riccardo Marin](https://ricma.netlify.app/)<sup>2</sup>, [Nikolaos Sarafianos](https://nsarafianos.github.io/)<sup>2</sup>, [Yuanlu Xu](https://web.cs.ucla.edu/~yuanluxu/)<sup>2</sup>, [Gerard Pons-Moll](https://virtualhumans.mpi-inf.mpg.de/people/pons-moll.html)<sup>1</sup>, [Tony Tung](https://sites.google.com/site/tony2ng/)<sup>2</sup>


<sup>1</sup>Real Virtual Human Group @ University of Tübingen & Tübingen AI Center & Max Planck Institute for Informatics \
<sup>2</sup>Meta Reality Lab Research 

## Preparation
##### 1. Dependencies
Please refer to [Dependencies](https://github.com/YuxuanSnow/NeuralSurfaceField/tree/main/libs#readme) for:
- install conda environment and required packages
- obtain SMPL model
- obtain prediffused SMPL skinning weights field
##### 2. Data
Please refer to [Data](https://github.com/YuxuanSnow/NeuralSurfaceField/blob/main/depth_renderer/README.md) for:
- render depth frames from scan
- preprocess data

## Running
### Learn Fusion Shape via SDF
##### 1. Canonicalize Input Partial Shape
canonicalize input partial shape by root finding, save to preprocessed file.
```
python buff_root_finding_male.py
python buff_root_finding_female.py
```

##### 2. Learn Canonical Fusion Shape
```
python buff_fusion_shape_male.py --exp_id 1 --batch_size 1 --split_file ./assets/data_split/buff_male_train_val.pkl --mode train --epochs 301
python buff_fusion_shape_female.py --exp_id 2 --batch_size 1 --split_file ./assets/data_split/buff_female_train_val.pkl --mode train --epochs 301
```

### Learn Neural Surface Field based on Fusion Shape
```
python buff_nsf_male.py --exp_id 11 --batch_size 2 --split_file ./assets/data_split/buff_male_train_val.pkl --mode train --epochs 301
python buff_nsf_female.py --exp_id 12 --batch_size 2 --split_file ./assets/data_split/buff_female_train_val.pkl --mode train --epochs 301
```
Please adjust the hyperparameter for learning rate decay given the amount of samples in your training data.

The Marching Cube extracted fusion shapes have inverted normals. Handle it in `Meshlab` by: `Filters - Normals, Curvatures and Orientation - Invert face orientation` 
