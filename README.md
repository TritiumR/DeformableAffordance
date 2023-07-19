# Learning Foresightful Dense Visual Affordance for Deformable Object Manipulation [ ICCV 2023 ]
This is the official implementation of the paper "Learning Foresightful Dense Visual Affordance 
for Deformable Object Manipulation" (ICCV 2023).

## Environment Installation
Please refer to https://github.com/Xingyu-Lin/softgym for SoftGym Installation Instructions.
## Collect Data
Collecting data for task **SpreadCloth**, please run 
```
bash scripts/collect_cloth-flatten.sh
```
Collecting data for task **RopeConfiguration**, please run 
```
bash scripts/collect_rope-configuration.sh
```
## Train Models
Our models are trained in a reversed step-by-step manner. We prepare some bash files to handle the training process.

To train models for **SpreadCloth**, pleas run
```
bash scripts/train_cloth-flatten.sh
```

To train models for **RopeConfiguration**, pleas run
```
bash scripts/train_rope-configuration.sh
```
## Test Models
We prepare some bash files to test the final model. you can easily change the argument to test our other models mentioned in our paper.

To test models for **SpreadCloth**, pleas run
```
bash scripts/test-cloth-flatten.sh
```

To test models for **RopeConfiguration**, pleas run
```
bash scripts/test-rope-configuration.sh
```

The manipulation result (gif) will be stored in "test_video" directory.