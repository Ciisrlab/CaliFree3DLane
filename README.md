# CaliFree3DLane
<p align="justify">
Monocular 3D lane detection plays a crucial role in autonomous driving, assisting vehicles in safe navigation. Existing methods primarily utilize calibrated camera parameters in the dataset to conduct 3D lane detection from a single image. However, errors or sudden absence of camera parameters can pose significant challenges to safe driving. On one hand, this can lead to incorrect feature acquisition, which further affects the precision of lane detection. On the other hand, it renders methods relying on transformation matrices for temporal fusion ineffective. To address the above issue and achieve accurate 3D lane detection, we propose CaliFree3DLane, a calibration-free method for spatio-temporal 3D lane detection based on Transformer structure. Instead of using geometric projections to obtain static reference points on images, we propose a reference point refinement strategy that dynamically updates the reference points and finally generates appropriate sampling points for image feature extraction. To integrate multi-frame features, we generate sub-queries from the current scene query to focus on the image features of each frame independently. We then aggregate these sub-queries to form a more comprehensive scene query for 3D lane detection. Using these operations, CaliFree3DLane accurately transforms multi-frame image features into the current bird's-eye view (BEV) space, enabling precise 3D lane detection. Experimental results show that our CaliFree3DLane achieves state-of-the-art 3D lane detection performance in various datasets.
</p>
    
## Environment Setup

- We develop with PyTorch 1.8 and recommend you to use Anaconda to create a conda environment before installing the dependencies
```
pip install torch==1.8.0 torchvision==0.9.0 torchaudio==0.8.0
```
- For [Deformable Attention](https://github.com/fundamentalvision/Deformable-DETR#requirements)
```
cd models/ops/
bash make.sh
# unit test (should see all checking is True)
python test.py
```
- For other dependencies
```
pip install -r requirement.txt
```
### Training and evaluation on Apollo 3D Lane Synthetic

- How to train:
    1. Please download [Apollo 3D Lane Synthetic](https://github.com/yuliangguo/3D_Lane_Synthetic_Dataset) and modify the configuration in the /tools/apollo_config.py
    2. Execute the following code:
```
cd tools
python3 train_apollo.py
```
- How to evaluate:
    1. Please modify the configuration in the /tools/val_apollo.py
    2. Execute the following code:
```
python val_apollo.py
```

### Training and evaluation on OpenLane

- How to train:
    1. Please download [OpenLane](https://github.com/OpenDriveLab/OpenLane) and modify the configuration in the /tools/apollo_config.py
    2. Execute the following code:
```
cd tools
python3 train_openlane.py
```
- How to evaluate:
    1. Please modify the configuration in the /tools/val_apollo.py
    2. Execute the following code:
```
python val_openlane.py
```

## Benchmark

### Results of different models on Apollo 3D Lane Synthetic (Balanced Scence)
- In/Extrinsics

| Method | F-Score | X error  near | X error far | Z error near | Z error far|
| :----: | :----: | :----: | :----: | :----: | :----: |
| 3D-LaneNet | 86.4 |0.068|0.477|0.015|0.202|
| Gen-LaneNet | 88.1 |0.061|0.486|0.012|0.214|
| CLGO | 91.9 |0.061|0.361|0.029|0.25|
| PersFormer | 92.9 |0.054|0.356|0.010|0.234|
| Anchor3DLane | 95.4 | 0.045 |0.300|0.016|0.223|
| BEV-LaneDet | 98.7 | 0.016 |0.242|0.020|0.216|
| Ours | 98.9 | 0.019 |0.228|0.021|0.221|

- without In/Extrinsics

| Method | F-Score | X error  near | X error far | Z error near | Z error far|
| :----: | :----: | :----: | :----: | :----: | :----: |
| BEV-LaneDet | 96.8 |0.035|0.292|0.033|0.250|
| Ours | 97.9 |0.031|0.261|0.030|0.243|

### Results of different models on OpenLane dataset
- In/Extrinsics

| Method | F-Score | X error  near | X error far | Z error near | Z error far|
| :----: | :----: | :----: | :----: | :----: | :----: |
| 3D-LaneNet | 44.1 |0.479|0.572|0.367|0.443|
| Gen-LaneNet | 32.5 |0.591|0.684|0.411|0.521|
| PersFormer | 50.5 |0.485|0.553|0.364|0.431|
| Anchor3DLane | 54.3 | 0.275 |0.310|0.105|0.135|
| BEV-LaneDet | 58.4 | 0.309 |0.659|0.244|0.631|
| Ours | 59.2| 0.238|0.527 | 0.173| 0.387|

| Method     | All  | Up&<br>Down | Curve | Extreme<br>Weather | Night | Intersection | Merge&<br>Split |
| :----:     |:----:|:----:|:----:|:----:|:----:|:----:|:----:|
| PersFormer | 42.0 | 40.7 | 46.3 | 43.7 | 36.1 | 28.9 | 41.2 | 
| Anchor3DLane | 54.3 | 47.2 | 58.0 | 52.7 | 48.7 | 45.8 | 51.7 | 
| BEV-LaneDet | 58.4 | 48.7 | 63.1 | 53.4 | 53.4 | 50.3 | 53.7 | 
| Ours | 59.2| 51.8 | 64.3 | 57.3 | 54.2 | 50.9 | 54.3 | 

- without In/Extrinsics

| Method | F-Score | X error  near | X error far | Z error near | Z error far|
| :----: | :----: | :----: | :----: | :----: | :----: |
| BEV-LaneDet | 54.7 | 0.346 |0.769|0.253|0.709|
| Ours  | 57.0 | 0.306 |0.761 |0.227 |0.683|

| Method     | All  | Up&<br>Down | Curve | Extreme<br>Weather | Night | Intersection | Merge&<br>Split |
| :----:     |:----:|:----:|:----:|:----:|:----:|:----:|:----:|
| BEV-LaneDet |54.7 | 47.4 | 61.4 | 49.2 | 49.7 | 46.4 | 49.9 | 
| Ours | 57.0 | 48.9 | 62.3 | 54.8 | 52.0 | 50.2 | 52.1 | 

## Visualization
<p align="center"><img src="openlane.jpg" width="1000"/></p>

## Acknowledgement

This repo is based on following works:
- [BEV-LaneDet](https://github.com/gigo-team/bev_lane_det)
- [SeqFormer](https://github.com/wjf5203/SeqFormer)
- [Deformable DETR](https://github.com/fundamentalvision/Deformable-DETR) 

Thanks for their wonderful contribution.

