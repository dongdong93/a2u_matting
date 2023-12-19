# A<sup>2</sup>U Matting

<p align="center">
  <img src="example.png" width="800" title="Example"/>
</p>

The official repository for paper:

> [**Learning Affinity-Aware Upsampling for Deep Image Matting**](https://openaccess.thecvf.com/content/CVPR2021/papers/Dai_Learning_Affinity-Aware_Upsampling_for_Deep_Image_Matting_CVPR_2021_paper.pdf),
> CVPR2021


## Prerequisite
```shell
git clone https://github.com/dongdong93/a2u_matting.git
cd a2u_matting/scripts
git clone https://github.com/mapillary/inplace_abn.git
cd inplace_abn
python setup.py install
cd ../Biupdownsample
python setup.py develop
```

## Installation
Our code has been tested on Python 3.6, Pytorch 1.3.1, CUDA 10.0 (inplace_abn only supports CUDA>=10.0). See other required packages in `requirements.txt`.


## Demo

    sh demo.sh

Our model trained on the Adobe Image Matting dataset can be downloaded from:

| Model | SAD | MSE | Grad | Conn | config |
| :--: | :--: | :--: | :--: | :--: | :--: |
| [Ours](https://drive.google.com/file/d/1hGe86w611FKl8YaTD3YWXlHDC0XIATyh/view?usp=sharing) | 32.10 | 0.0078 | 16.33 | 29.00 | ./config/adobe.yaml |
 
Disclaimer:
- This is a reimplementation. The metrics may be slightly different from the ones reported in our original paper.

Another pretrained model on the Adobe Image Matting dataset with 'unshare' setting can be downloaded from:
| Model | SAD | MSE | Grad | Conn | config |
| :--: | :--: | :--: | :--: | :--: | :--: |
| [Ours-unshare](https://drive.google.com/file/d/1CGunN8gMHDypUzWzsj5xfa0I9bM16kyJ/view?usp=sharing) | 31.87 | 0.0075 | 14.25 | 29.17 | ./config/adobe_unshare.yaml |

**As covered by ADOBE IMAGE MATTNG DATASET LICENSE AGREEMENT, the trained models included in this repository can only be used/distributed for non-commercial purposes.**


## Training
### Data Preparation
- Please contact [*Deep Image Matting*](https://openaccess.thecvf.com/content_cvpr_2017/papers/Xu_Deep_Image_Matting_CVPR_2017_paper.pdf) for the **Adobe Image Matting** dataset;
- The final structure for use:

````
$PATH_TO_DATASET/Combined_Dataset
├──── Training_set
│    ├──── alpha (431 images)
│    ├──── fg (431 images)
│    └──── train2014 (82783 images)
├──── Test_set
│    ├──── alpha (50 images)
│    ├──── merged (1000 images)
│    └──── trimaps (1000 images)
````

- Please contact [*HAttMatting*](https://openaccess.thecvf.com/content_CVPR_2020/papers/Qiao_Attention-Guided_Hierarchical_Structure_Aggregation_for_Image_Matting_CVPR_2020_paper.pdf) for the **Distinctions-646** dataset and organize it following the same structure; 


### Backbone
Download the pretrained ResNet34 from [*here*](https://github.com/mapillary/inplace_abn) and save to `./pretrained`. (It will be automatically downloaded)

### Train your model

    sh train.sh

Models will be saved to `./savemodel` by default. FYI, it can be trained with 2 1080Ti using the provided config files. 

## Inference

    sh test.sh
    
Results will be saved to `./results` by default. Quantitative metrics should be measured by `matlab_testcode/test_withpic.m`. The python version is for reference only.

## Citation
If you find this work or code useful for your research, please consider citing:
```
@inproceedings{dai2021learning,
  title={Learning Affinity-Aware Upsampling for Deep Image Matting},
  author={Dai, Yutong and Lu, Hao and Shen, Chunhua},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={6841--6850},
  year={2021}
}

```

## References
[*IndexNet Matting*](https://github.com/poppinace/indexnet_matting)

[*GCA Matting*](https://github.com/Yaoyi-Li/GCA-Matting)

[*CARAFE*](https://github.com/myownskyW7/CARAFE)


## Contact
yutong.dai@adelaide.edu.au


