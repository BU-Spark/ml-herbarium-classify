# ML-Herbarium-Classify

# Overview
 
The changing climate increases stressors that weaken plant resilience, disrupting forest structure and ecosystem services. Rising temperatures lead to more frequent droughts, wildfires, and invasive pest outbreaks, leading to the loss of plant species. That has numerous detrimental effects, including lowered productivity, the spread of invasive plants, vulnerability to pests, altered ecosystem structure, etc. The project aims to aid climate scientists in capturing patterns in plant life concerning changing climate.

The herbarium specimens are pressed plant samples stored on paper. The specimen labels are handwritten and date back to the early 1900s. The labels contain the curator's name, their institution, the species and genus, and the date the specimen was collected. Since the labels are handwritten, they are not readily accessible from an analytical standpoint. The data, at this time, cannot be analyzed to study the impact of climate on plant life.

The digitized samples are an invaluable source of information for climate change scientists, and are providing key insights into biodiversity change over the last century. Digitized specimens will facilitate easier dissemination of information and allow more people access to data. The project, if successful, would enable users from various domains in environmental science to further studies pertaining to climate change and its effects on flora and even fauna.


## Project Description

The Harvard University Herbaria aims to digitize the valuable handwritten information on herbarium specimens, which contain crucial insights into biodiversity changes in the Anthropocene era. 

The main challenge is to develop a transformer-based optical character recognition (OCR) model using deep learning techniques to accurately **locate and extract the specimen labels on the samples to preserve them digitally**. The secondary task involves **building a plant classifier using taxon labels as ground truth, to supplement the OCR model as a source of a priori knowledge.** The tertiary goal involves **identifying the phenology of the plant specimen under consideration [will be updated after discussing with the client] and possibly predict the biological life cycle stage of the plant**. The successful completion of these objectives will showcase the importance of herbaria in storing and disseminating data for various biological research areas. The ultimate goal is to revive and digitize this valuable information to promote its accessibility to the public and future generations.


## For usage on SCC, first, create a conda environment using the following:
```
module load miniconda
```
```
conda create -n my_env -c conda-forge python=’version’
```
```
conda activate my_env
```

## The 'sec_task3.ipynb' file is used to download the images for the primary task dataset by filtering the GBIF data based on the New England dataset.
Run the cells in this notebook to download the images and save it in the desired path.

## The 'Mask_remove_text-3.ipynb' file is used to remove the labels from the image so that there is no bias influencing the classifier.
Add the following modules first(after creating a conda nevironment with python version > 3.9): 
```
module load tensorflow/2.11.0
```
```
module load gcc/12.2.0
```
```
module load bazel/4.1.0
```
```
module load cuda/11.2
```
```
module load cudnn/8.1.1
```
(conda install module_name – for all other packages)
Run the cells in this notebook to create a datset of the images with masked labels.

## 'IELT_data_prep.ipynb' is just used to convert the downloaded data to the format required by the IELT model(.tgz format)

To run the IELT model follow the following instructions:
## !! NOTE THAT THIS MODEL HAS BEEN SETUP TO RUN ON THEIR CUB DATASET, NOT ON THE DOWNLOADED HERBARIUM DATASET !!
## Requirements

python     >= 3.9

pytorch	>= 1.8.1
```
conda install -c conda-forge tqdm
```
```
conda install -c conda-forge timm
```

## Training

1. Put the pre-trained ViT model in `pretrained/`, and rename it to `ViT-B_16.npz`, you can download from [ViT pretrained](https://console.cloud.google.com/storage/browser/_details/vit_models/imagenet21k/ViT-B_16.npz).
2. Select a experiments setting file in `configs/`, and Modify the path of `dataset`.
3. Modify the path in `setup.py` in line 5, and you can change the log name and cuda visible by modify line 13,14.
4. Running the following code according to you pytorch version:

### Single GPU

```bash
python -m main.py
```

### Multiple GPUs

#### If pytorch < 1.12.0

```bash
python -m torch.distributed.launch --nproc_per_node 4 main.py 
```

#### If pytorch >= 1.12.0

```
torchrun --nproc_per_node 4 main.py
```

You need to change the number behind the `-nproc_per_node` to your number of GPUs

