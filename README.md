# VFace
This repository gives the official implementation of VFace: A Training-Free Approach for Diffusion-Based Video Face Swapping (WACV 2026)


<!-- ![Example](assets/teaser2.jpeg) -->
<!-- ### [Paper](https://arxiv.org/abs/2409.07269) -->
[Sanoojan Baliah](https://www.linkedin.com/in/sanoojan/), Yohan Abeysinghe, Rusiru Thushara, Khan Muhammad, Abhinav Dhall, Karthik Nandakumar, and Muhammad Haris Khan

## Abstract
>We present a training-free, plug-and-play method, namely VFace, for high-quality face swapping in videos. It
can be seamlessly integrated with image-based face swapping approaches built on diffusion models. First, we introduce a Frequency Spectrum Attention Interpolation technique to facilitate generation and intact key identity characteristics. Second, we achieve Target Structure Guidance via plug-and-play attention injection to better align the structural features from the target frame to the generation. Third, we present a Flow-Guided Attention Temporal Smoothening mechanism that enforces spatiotemporal coherence without modifying the underlying diffusion model to reduce temporal inconsistencies typically encountered in frame-wise generation. Our method requires no additional training or video-specific fine-tuning. Extensive experiments show that our method significantly enhances temporal consistency and visual fidelity, offering a practical and modular solution for video-based face swapping. Our code is available at (https://github.com/Sanoojan/VFace).



## News
- *2025-12-06* Release VFace for REFace


## Requirements
A suitable [conda](https://conda.io/) environment named `VFace` can be created
and activated with:

```
conda create -n "VFace" python=3.10.13 -y
conda activate VFace
cd REFace
sh setup.sh
```

Follow the instructions in the REFace repository (https://github.com/Sanoojan/REFace/tree/main) to download the model and install the required dependencies for setting up image-level inference.

## Testing

To test our model on a video dataset and another source images dataset, you can use `REFace/scripts/VFace_inference_batch.py`. For example, 

or simply run:
```
sh REFace/VFace_video_swap_batch.sh
```



## Video Test Benchmark




## Citing Us
If you find our work valuable, we kindly ask you to consider citing our paper and starring ⭐ our repository. Our implementation includes a standard metric code and we hope it make life easier for the generation research community.


```
@article{baliah2024realistic,
  title={Realistic and Efficient Face Swapping: A Unified Approach with Diffusion Models},
  author={Baliah, Sanoojan and Lin, Qinliang and Liao, Shengcai and Liang, Xiaodan and Khan, Muhammad Haris},
  journal={arXiv preprint arXiv:2409.07269},
  year={2024}
}
```

## Acknowledgements

This code borrows heavily from [Paint-By-Example](https://github.com/Fantasy-Studio/Paint-by-Example) and [REFace](https://github.com/Sanoojan/REFace).

## Maintenance

Please open a GitHub issue for any help. If you have any questions regarding the technical details, feel free to contact us. 

## License


This project is licensed under the MIT License. See LICENSE.txt for the full MIT license text.

Additional Notes:

Note 1: This project includes a derivative of [Paint-By-Example](https://github.com/Fantasy-Studio/Paint-by-Example) licensed under the CreativeML Open RAIL-M license. The original license terms and use-based restrictions of the CreativeML Open RAIL-M license still apply to the model and its derivatives. Please refer to https://github.com/Fantasy-Studio/Paint-by-Example?tab=License-1-ov-file for more details.

Note 2: This work includes a model that has been trained using the CelebAMask-HQ dataset. The CelebAMask-HQ dataset is available for non-commercial research purposes only. As a result, any use of this model must comply with the non-commercial usage restriction of the CelebAMask-HQ dataset. Use of this model for commercial purposes is strictly prohibited.
