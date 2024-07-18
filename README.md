# BI-MDRG: Bridging Image History in Multimodal Dialogue Response Generation (ECCV 2024)

This repository provides the official implementation of our ECCV 2024 paper:
> BI-MDRG: Bridging Image History in Multimodal Dialogue Response Generation    
> Authors: Hee Suk Yoon*, Eunseop Yoon*, Joshua Tian Jin Tee*, Kang Zhang, Yu-Jung Heo, Du-Seong Chang, Chang D. Yoo

The implementation is built upon [openflamingo](https://github.com/mlfoundations/open_flamingo).

[[Paper Link]()]

## Installation
```bash
# Clone this repo
git clone https://github.com/hee-suk-yoon/BI-MDRG.git
cd BI-MDRG

# Create a conda enviroment
1. conda env create -f environment.yml
2. conda activate bimdrg
```

## Datasets
1. Download the [MMDialog](https://github.com/victorsungo/MMDialog) dataset and prepare using the following preprocessing code

2. Prepare Citation Augmented Data

3. Multimodal Dialogue Image Consistency (MDIC) Dataset

    To evaluate the image consistency in multimodal dialogue, we have created a curated set of 300 dialogues annotated to track object consistency across conversations based on the MMDialog dataset.

    You can find the dataset at: `mdic/mdic.pkl`

    The dataset format is: `{dialogue_id: [citation_tags]}`


## Training

## Evaluation


## Acknowledgement
This work was supported by a grant of the KAIST-KT joint research project through AI2X Lab., Tech Innovation Group, funded by KT (No. D23000019, Developing Visual and Language Capabilities for AI-Based Dialogue Systems), and by Institute for Information \& communications Technology Planning \& Evaluation (IITP) grant funded by the Korea government(MSIT) (No. 2021-0-01381, Development of Causal AI through Video Understanding and Reinforcement Learning, and Its Applications to Real Environments).

Also, we thank the authors of the [OpenFlamingo](https://github.com/mlfoundations/open_flamingo), [Subject-Diffusion](https://github.com/OPPO-Mente-Lab/Subject-Diffusion), [MMDialog](https://github.com/victorsungo/MMDialog) for their open-source contributions.


## Contact
If you have any questions, please feel free to email hskyoon@kaist.ac.kr