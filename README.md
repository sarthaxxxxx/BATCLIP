<div align="center">

## $\texttt{BATCLIP}$: Bimodal Online Test-Time Adaptation for CLIP


[Sarthak Kumar Maharana<sup>1</sup>](https://sarthaxxxxx.github.io/), [Baoming Zhang<sup>1</sup>](https://www.linkedin.com/in/baoming-zhang-286083313/), [Leonid Karlinsky<sup>2</sup>](https://scholar.google.com/citations?user=WbO7tjYAAAAJ&hl=en), [Rogerio Feris<sup>2</sup>](https://www.rogerioferis.org/), and [Yunhui Guo<sup>1</sup>](https://yunhuiguo.github.io/) 
<br>
<sup>1</sup>The University of Texas at Dallas <sup>2</sup> MIT-IBM Watson AI Lab
<br>
ICCV 2025 

✍🏻 [Paper](https://arxiv.org/abs/2412.02837) 🔗 [Project](https://sarthaxxxxx.github.io/BATCLIP/index.html)
</div>


## Abstract 
Although open-vocabulary classification models like Contrastive Language Image Pretraining (CLIP) have demonstrated strong zero-shot learning capabilities, their robustness to common image corruptions remains poorly understood. Through extensive experiments, we show that zero-shot CLIP lacks robustness to common image corruptions during test-time, necessitating the adaptation of CLIP to unlabeled corrupted images using test-time adaptation (TTA). However, we found that existing TTA methods have severe limitations in adapting CLIP due to their unimodal nature. To address these limitations, we propose $\texttt{BATCLIP}$, a bimodal $\textbf{online}$ TTA method designed to improve CLIP's robustness to common image corruptions. The key insight of our approach is not only to adapt the visual encoders for improving image features but also to strengthen the alignment between image and text features by promoting a stronger association between the image class prototype, computed using pseudo-labels, and the corresponding text feature. We evaluate our approach on benchmark image corruption datasets and achieve state-of-the-art results in online TTA for CLIP. Furthermore, we evaluate our proposed TTA approach on various domain generalization datasets to demonstrate its generalization capabilities.


## Prerequisites
To use the repository, we provide a conda environment.
```bash
conda update conda
conda env create -f environment.yml
conda activate tta 
```

## Usage

$\texttt{BATCLIP}$ is heavily built upon [this](https://github.com/mariodoebler/test-time-adaptation). Thanks, Mario Doebler! 

<details open>
<summary>Features</summary>

- **Datasets**
  - `cifar10_c` [CIFAR10-C](https://zenodo.org/record/2535967#.ZBiI7NDMKUk)
  - `cifar100_c` [CIFAR100-C](https://zenodo.org/record/3555552#.ZBiJA9DMKUk)
  - `imagenet_c` [ImageNet-C](https://zenodo.org/record/2235448#.Yj2RO_co_mF)

- **Models**
  - It is also possible to use the models provided by [OpenCLIP](https://github.com/mlfoundations/open_clip/tree/main).
  
- **Settings**
  - `reset_each_shift` Reset the model state after the adaptation to a domain. We follow this setting.


- **Mixed Precision Training**
  - Almost all of the aforementioned methods (except SAR and GTTA) can be trained with mixed precision. This greatly 
  speeds up your experiments and requires less memory. However, all benchmark results are generated with fp32.

- **Modular Design**
  - Adding new methods should be rather simple, thanks to the modular design.

</details>

### Get Started
Once you’ve obtained any missing datasets, update the root data directory in `conf.py` by setting `_C.DATA_DIR = "./data"`. If your individual dataset folders use names other than those defined in the complete_data_dir_path mapping (also in `conf.py`), simply edit that dictionary to match your directory names.


### Run Experiments
Example run, 
```bash
python test_time.py --cfg cfgs/imagenet_c/ours.yaml MODEL.ARCH VIT-B-16 MODEL.WEIGHTS openai MODEL.USE_CLIP True SETTING reset_each_shift
```

You can head over to the config files to change the parameters.

## TODO
- [ ] Key results and viz.
- [ ] Framework pending


## Citation 

```bibtex
@inproceedings{maharana2025batclip,
  title={BATCLIP: Bimodal Online Test-Time Adaptation for CLIP},
  author={Maharana, Sarthak Kumar and Zhang, Baoming and Karlinsky, Leonid and Feris, Rogerio and Guo, Yunhui},
  journal={International Conference on Computer Vision (ICCV)},
  year={2025}
}


