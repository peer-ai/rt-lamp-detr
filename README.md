# RT-LAMP-DETR

This repository contains the code for RT-LAMP-DETR, an artificial intelligence (AI) operated-tool to enable a more precise and rapid result analysis in large scale testing presented in *One-step colorimetric isothermal detection of COVID-19 with AI-assisted automated result analysis: a platform model for future emerging point-of-care RNA/DNA diagnosis.* In this project, we developed a novel ultrasensitive and specific dual RT-LAMP assay targeting the Nsp9 segment of SARS-CoV-2 ORF1ab gene and human 18S rRNA gene (internal control) for decentralized COVID-19 screening with two modes of analysis: naked-eye observation for the highest convenience, and an AI-operated automated analysis to accommodate high-throughput testing.

*This repository is based on the original DETR code from https://github.com/facebookresearch/detr*

![RT-LAMP-DETR](/accompanying_image/RT-LAMP-DETR.png)

## RT-LAMP-DETR dataset

The dataset is under _data folder of this repository. The data is already split into 3 folders for training (train), validation (val) and testing (test).

## Install the required packages
    
    pip install -r requirements.txt

## Training RT-LAMP-DETR from scratch

    ./train_rt_lamp_detr.sh

## Evaluate RT-LAMP-DETR 

Please see eval_rt_lamp_detr.ipynb notebook for the code to evaluate RT-LAMP-DETR based on the trained model.

## Our pretrained RT-LAMP-DETR model

Due to the size of the file, our pretrained RT-LAMP-DETR model is only available upon request.

## Using docker
    
    git@github.com:peer-ai/rt-lamp-detr.git
    cd rt-lamp-det
    docker build . -t rt-lamp-detr
    
    # train 
    docker run -v `pwd`:/workspace --gpus 0 --shm-size=2g -it rt-lamp-detr ./train_rt_lamp_detr.sh

## Citation

Please cite this work if you find it useful.

    @journal{rt-lamp-detr,
      author = {...},
      title = {One-step colorimetric isothermal detection of COVID-19 with AI-assisted automated result analysis: a platform model for future emerging point-of-care RNA/DNA diagnosis},
      year = {2021},
      publisher = {...},
      journal = {...},
      howpublished = {...}
    }
