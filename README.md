# Instance_Unique_Querying
<!-- [![NVIDIA Source Code License](https://img.shields.io/badge/license-NSCL-blue.svg)](https://github.com/NVlabs/SegFormer/blob/master/LICENSE) -->
![Python 3.7](https://img.shields.io/badge/python-3.7-green.svg)

# Learning Equivariant Segmentation with Instance-Unique Querying

<!-- ![image](resources/image.png) -->
<div align="center">
  <img src="./resources/Architecture.png">
</div>
<p align="center">
  Overview of our new training framework for query-based instance segmentation.
</p>

This is official repo for Learning Equivariant Segmentation with Instance-Unique Querying. Our full implementation will be availble at [mmdetection](https://github.com/open-mmlab/mmdetection) for easy-to-use, stay tuned!

## Abstract
Prevalent state-of-the-art instance segmentation methods fall into a query-based scheme, in which instance masks are derived by querying the image feature using a set of instance-aware embeddings. In this work, we devise a new training framework that boosts query-based models through discriminative query embedding learning. It explores two essential properties, namely dataset-level uniqueness and transformation equivariance, of the relation between queries and instances. First, our algorithm uses the queries to retrieve the corresponding instances from the whole training dataset, instead of only searching within individual scenes. As querying instances across scenes is more challenging, the segmenters are forced to learn more discriminative queries for effective instance separation. Second, our algorithm encourages both image (instance) representations and queries to be equivariant against geometric transformations, leading to more robust, instance-query matching. 

## Installation
This implementation is built on [mmdetection](https://github.com/open-mmlab/mmdetection) and [AdelaiDet](https://github.com/aim-uofa/AdelaiDet). Many thanks to the authors for the efforts.

```
conda create --name <env> --file requirements.txt
```

## Training

We use the slurm system to train our model. [Slurm](https://slurm.schedmd.com/) is a good job scheduling system for computing clusters.

On a cluster managed by Slurm, you can use `slurm_train.sh` to spawn training jobs. It supports both single-node and multi-node training.

The basic usage is as follows.

```shell
OMP_NUM_THREADS=1 [GPUS=${GPUS}] ./tools/slurm_train.sh ${PARTITION} ${JOB_NAME} ${CONFIG_FILE} ${WORK_DIR}
```

When using Slurm, the port option need to be set in one of the following ways:

1. Set the port through `--options`. This is more recommended since it does not change the original configs.

   ```shell
   OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=0,1,2,3 GPUS=4 ./tools/slurm_train.sh ${PARTITION} ${JOB_NAME} config1.py ${WORK_DIR} --options 'dist_params.port=29500'
   OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=4,5,6,7 GPUS=4 ./tools/slurm_train.sh ${PARTITION} ${JOB_NAME} config2.py ${WORK_DIR} --options 'dist_params.port=29501'
   ```

2. Modify the config files to set different communication ports.

   In `config1.py`, set

   ```python
   dist_params = dict(backend='nccl', port=29500)
   ```

   In `config2.py`, set

   ```python
   dist_params = dict(backend='nccl', port=29501)
   ```

   Then you can launch two jobs with `config1.py` and `config2.py`.

   ```shell
   OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=0,1,2,3 GPUS=4 ./tools/slurm_train.sh ${PARTITION} ${JOB_NAME} config1.py ${WORK_DIR}
   OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=4,5,6,7 GPUS=4 ./tools/slurm_train.sh ${PARTITION} ${JOB_NAME} config2.py ${WORK_DIR}
   ```

Note that:
- The configs are made for 8-GPU training. To train on another number of GPUs, change the `GPUS`.
- If you want to measure the inference time, please change the number of gpu to 1 for inference.
- We set `OMP_NUM_THREADS=1` by default, which achieves the best speed on our machines, please change it as needed.

## Citation
```
@inproceedings{wang2022learning,
  title={Learning Equivariant Segmentation with Instance-Unique Querying},
  author={Wang, Wenguan and Liang, James and Liu, Dongfang},
  booktitle={NeurIPS},
  year={2022}
}
```