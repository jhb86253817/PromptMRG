# PromptMRG

Code of AAAI 2024 paper: "PromptMRG: Diagnosis-Driven Prompts for Medical Report Generation".

## Installation
1. Clone this repository.
```Shell
git clone https://github.com/jhb86253817/PromptMRG.git
```
2. Create a new conda environment.
```Shell
conda create -n promptmrg python=3.10
conda activate promptmrg
```
3. Install the dependencies in requirements.txt.
```Shell
pip install -r requirements.txt
```
## Datasets Preparation
* **MIMIC-CXR**: The images can be downloaded from either [physionet](https://www.physionet.org/content/mimic-cxr-jpg/2.0.0/) or [R2Gen](https://github.com/zhjohnchan/R2Gen). The annotation file can be downloaded from the [Google Drive](https://drive.google.com/file/d/1qR7EJkiBdHPrskfikz2adL-p9BjMRXup/view?usp=sharing). Additionally, you need to download `clip_text_features.json` from [here](https://drive.google.com/file/d/1Zyq-84VOzc-TOZBzlhMyXLwHjDNTaN9A/view?usp=sharing), the extracted text features of the training database via MIMIC pretrained [CLIP](https://stanfordmedicine.app.box.com/s/dbebk0jr5651dj8x1cu6b6kqyuuvz3ml). Put all these under folder `data/mimic_cxr/`.
* **IU-Xray**: The images can be downloaded from [R2Gen](https://github.com/zhjohnchan/R2Gen) and the annotation file can be downloaded from the [Google Drive](https://drive.google.com/file/d/1zV5wgi5QsIp6OuC1U95xvOmeAAlBGkRS/view?usp=sharing). Put both images and annotation under folder `data/iu_xray/`.

Moreover, you need to download the `chexbert.pth` from [here](https://stanfordmedicine.app.box.com/s/c3stck6w6dol3h36grdc97xoydzxd7w9) for evaluating clinical efficacy and put it under `checkpoints/stanford/chexbert/`.

You will have the following structure:
````
PromptMRG
|--data
   |--mimic_cxr
      |--base_probs.json
      |--clip_text_features.json
      |--mimic_annotation_promptmrg.json
      |--images
         |--p10
         |--p11
         ...
   |--iu_xray
      |--iu_annotation_promptmrg.json
      |--images
         |--CXR1000_IM-0003
         |--CXR1001_IM-0004
         ...
|--checkpoints
   |--stanford
      |--chexbert
         |--chexbert.pth
...
````

## Training

## Testing

## Acknowledgment
* [R2Gen](https://github.com/zhjohnchan/R2Gen)
* [BLIP](https://github.com/salesforce/BLIP)
* [cvt2distilgpt2](https://github.com/aehrc/cvt2distilgpt2)
