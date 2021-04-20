# HUMAN4D: A Human-Centric Multimodal Dataset for Motions & Immersive Media

HUMAN4D constitutes a large and multimodal 4D dataset that contains a variety of human activities simultaneously captured by a professional marker-based MoCap, a volumetric capture and an audio recording system.

![alt text](https://raw.githubusercontent.com/tofis/myurls/master/human4d/imgs/facilities.png)
Pictures taken during the preparation and capturing of the HUMAN4D dataset. The room was equipped with 24 Vicon MXT40S cameras rigidly placed on the walls, while a portable volumetric capturing system (https://github.com/VCL3D/VolumetricCapture) with 4 Intel RealSense D415 depth sensors was temporarily set up to capture the RGBD data cues.

![alt text](https://raw.githubusercontent.com/tofis/myurls/master/human4d/imgs/rgbd2.png)
HW-SYNCed multi-view RGBD samples (4 RGBD frames each) from "stretching_n_talking"(top) and "basket-ball_dribbling"(bottom) activities. 

![alt text](https://raw.githubusercontent.com/tofis/myurls/master/human4d/imgs/actor_bodyscan_s.png)
3D Scanning using a custom photogrammetry rig with 96 cameras, photos were taken of the actor (left) and reconstructed into a 3D textured mesh using Agisoft Metashape (right).

![alt text](https://raw.githubusercontent.com/tofis/myurls/master/human4d/imgs/meshreco2.png)
Reconstructed mesh-based volumetric  data  with  (Left)  color  per  vertex  visualization  in 3  voxel-grid resolutions, i.e. r= 5, r= 6 andr= 7 and (Right) textured 3D mesh sample in voxel-grid resolution for r= 6.

![alt text](https://raw.githubusercontent.com/tofis/myurls/master/human4d/imgs/pcloud.png)
Merged reconstructed point-cloud from one single mRGBD frame from various views.

If you used the dataset or found this work useful, please cite:
```
@article{chatzitofis2020human4d,
  title={HUMAN4D: A Human-Centric Multimodal Dataset for Motions and Immersive Media},
  author={Chatzitofis, Anargyros and Saroglou, Leonidas and Boutis, Prodromos and Drakoulis, Petros and Zioulis, Nikolaos and Subramanyam, Shishir and Kevelham, Bart and Charbonnier, Caecilia and Cesar, Pablo and Zarpalas, Dimitrios and others},
  journal={IEEE Access},
  volume={8},
  pages={176241--176262},
  year={2020},
  publisher={IEEE}
}
```
