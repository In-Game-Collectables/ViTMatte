## Get Started

* [Installation](docs/installation.md)



## Setup
### SAM Model
In order to generate trimap images, we utilise Segment Anything Model. To install Segment Anything follow the instructions in https://github.com/facebookresearch/segment-anything
Download the ViT-H SAM Model from Segment Anything Github repo and place it in the `pretrained` folder.

### VitMatte Checkpoints
Next, download the desired VitMatte model from any of the links below and place it in the `pretrained` folder.

Quantitative Results on [Composition-1k](https://paperswithcode.com/dataset/composition-1k)
| Model      | SAD   | MSE | Grad | Conn  | checkpoints |
| ---------- | ----- | --- | ---- | ----- | ----------- |
| ViTMatte-S | 21.46 | 3.3 | 7.24 | 16.21 | [GoogleDrive](https://drive.google.com/file/d/12VKhSwE_miF9lWQQCgK7mv83rJIls3Xe/view?usp=sharing) |
| ViTMatte-B | 20.33 | 3.0 | 6.74 | 14.78 | [GoogleDrive](https://drive.google.com/file/d/1mOO5MMU4kwhNX96AlfpwjAoMM4V5w3k-/view?usp=sharing) |

Quantitative Results on [Distinctions-646](https://paperswithcode.com/dataset/distinctions-646)
| Model      | SAD   | MSE | Grad | Conn  | checkpoints |
| ---------- | ----- | --- | ---- | ----- | ----------- |
| ViTMatte-S | 21.22 | 2.1 | 8.78 | 17.55 | [GoogleDrive](https://drive.google.com/file/d/18wIFlhFY9MPqyH0FGiB0PFk3Xp2xTHzx/view?usp=sharing) |
| ViTMatte-B | 17.05 | 1.5 | 7.03 | 12.95 | [GoogleDrive](https://drive.google.com/file/d/1d97oKuITCeWgai2Tf3iNilt6rMSSYzkW/view?usp=sharing) |

```

## Demo

You could try to matting the demo image with its corresponding trimap by run:
```
python run_one_image.py \
    --model vitmatte-s \
    --checkpoint-dir path/to/checkpoint
```
The demo images will be saved in ``./demo``.
You could also try with your own image and trimap with the same file.