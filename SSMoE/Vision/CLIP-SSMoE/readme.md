# SSMoE Implementation

## Set up
To get started, create and activate a Conda environment, 
```
conda install --yes -c pytorch pytorch=1.7.1 torchvision cudatoolkit=<your cuda version>
pip install ftfy regex tqdm
pip install git+https://github.com/openai/CLIP.git
pip install -e .
```


## Evaluation

Here we follow the weights of the CLIP-MoE 4 experts with top-2 activation obtained from both ShareGPT4V. The model weights are available on [Hugging Face](https://huggingface.co/MajorDavidZhang/CLIP-MoE/tree/main). Please download the weights. After that you can modify and use the evaluation scripts in `CLIP-MoE/eval`.




### Run Retrieval Task
The Code for Retrieval task eval/retrieval. Data preparation follows by CLIP-MoE. Run Retrieval Task for Coco dataset:
```
bash eval/retrieval/run_coco.sh
```


### Run Image Classification Task
The Code for Classification task eval/classification. Data preparation follows by CLIP-MoE. Run Classification Task for Cifar10 dataset:
```
bash eval/classification/cifar/run_cifar10.sh
```



## Acknowledgement
- [CLIP-MoE](https://github.com/OpenSparseLLMs/CLIP-MoE) The codebase we built upon.

