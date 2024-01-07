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
* **MIMIC-CXR**: The images can be downloaded from [here](https://www.physionet.org/content/mimic-cxr-jpg/2.0.0/) and the annotation file can be downloaded from here. Put Both images and annotation under `data/mimic_cxr/`.
* **IU-Xray**: The images can be downloaded from [R2Gen](https://github.com/zhjohnchan/R2Gen) and the annotation file can be downloaded from here. Put both images and annotation under `data/iu_xray/`.

## Training

## Testing

## Acknowledgment
* [R2Gen](https://github.com/zhjohnchan/R2Gen)
* [BLIP](https://github.com/salesforce/BLIP)
* [cvt2distilgpt2](https://github.com/aehrc/cvt2distilgpt2)
