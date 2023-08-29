# NSF: Neural Surface Field for Human Modeling from Monocular Depth
#### [Project Page](https://yuxuan-xue.com/nsf) | [Video](PlaceHolder) | [Paper](PlaceHolder)

In ICCV 2023, Paris

[Yuxuan Xue](https://yuxuan-xue.com/)<sup>1, *</sup>, [Bharat Lal Bhatnagar](https://virtualhumans.mpi-inf.mpg.de/people/Bhatnagar.html)<sup>2,*</sup>, [Riccardo Marin](https://ricma.netlify.app/)<sup>1</sup>, [Nikolaos Sarafianos](https://nsarafianos.github.io/)<sup>2</sup>, [Yuanlu Xu](https://web.cs.ucla.edu/~yuanluxu/)<sup>2</sup>, [Gerard Pons-Moll](https://virtualhumans.mpi-inf.mpg.de/people/pons-moll.html)<sup>1</sup>, [Tony Tung](https://sites.google.com/site/tony2ng/)<sup>2</sup>


<sup>1</sup>Real Virtual Human Group @ University of Tübingen & Tübingen AI Center & Max Planck Institute for Informatics \
<sup>2</sup>Meta Reality Lab Research 

![](https://github.com/YuxuanSnow/NeuralSurfaceField/blob/main/assets/data_split/teaser.png)

## News :triangular_flag_on_post:
- [2023/08/30] NSF paper is available on ArXiv.
- [2023/08/29] Code for NSF is available.
- [2023/07/14] NSF is accepted to ICCV 2023, Paris.



## Key Insight :raised_hands:
- Define neural fields on top of continuous surface
- Learned NSF generalizes to arbitrary resolution or topology of the surface
- Why benefitial For clothed human modelling:
  - eliminates need for Marching Cube / Poisson Reconstruction per frame => **efficient**
  - can reconstruct / animate human mesh resolution / topology => **flexible**
  - keeps mesh coherency across different frames => **modelling**

![](https://github.com/YuxuanSnow/NeuralSurfaceField/blob/main/assets/data_split/nsf_idea.png)

  
## Instruction :blue_book:
### A. Preparation
##### 1. Dependencies
Please refer to [Dependencies](https://github.com/YuxuanSnow/NeuralSurfaceField/tree/main/libs#readme) for:
- install conda environment and required packages
- obtain SMPL model
- obtain prediffused SMPL skinning weights field
##### 2. Data
Please refer to [Data](https://github.com/YuxuanSnow/NeuralSurfaceField/blob/main/depth_renderer/README.md) for:
- render depth frames from scan
- preprocess data
##### 3. Path
Please specify `ROOT_DIR` and `DATA_DIR` [here](https://github.com/YuxuanSnow/NeuralSurfaceField/blob/main/libs/global_variable.py)

### B. Running
#### Learn Fusion Shape via SDF
Please refer to [Fusion Shape](https://github.com/YuxuanSnow/NeuralSurfaceField/blob/main/models/Fusion_shape.md) for:
- learn implicit fusion shape
- fit SMPL-D to fusion shape
- project off-surface points onto fusion shape

#### Learn Neural Surface Field based on Fusion Shape
Please refer to [NSF](https://github.com/YuxuanSnow/NeuralSurfaceField/blob/main/models/NSF.md) for:
- learn NSF to model clothed avatar
- infer clothed avatar at arbitrary resolution
- animate clothed avatar with desired pose sequences

### C. Pre-trained Model
Please refer to [Pretrained Models](https://github.com/YuxuanSnow/NeuralSurfaceField/blob/main/models/pretrained_model.md) for:
- available pretrained models for subjects in BuFF/CAPE dataset


## Citation :writing_hand:

```bibtex
@inproceedings{xue2023nsf,
  title     = {{NSF: Neural Surface Field for Human Modeling from Monocular Depth}},
  author    = {Xue, Yuxuan and Bhatnagar, Bharat Lal and Marin, Riccardo and Sarafianos, Nikolaos and Xu, Yuanlu and Pons-Moll, Gerard and Tung, Tony.},
  booktitle = {Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
  month     = {October},
  year      = {2023},
}


