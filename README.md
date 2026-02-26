# FreqTrajNet: Frequency-aware Trajectory Correlation Network for Continuous Sign Language Recognition

## Data Preparation

### PHOENIX2014 dataset
1. Download the RWTH-PHOENIX-Weather 2014 dataset from the official website: [[download link]](https://www-i6.informatik.rwth-aachen.de/~koller/RWTH-PHOENIX/). Our experiments are conducted based on the phoenix-2014.v3.tar.gz package. You may download it directly using the following command:

   ```bash
   wget https://www-i6.informatik.rwth-aachen.de/ftp/pub/rwth-phoenix/2016/phoenix-2014.v3.tar.gz
   ```

2. After downloading, extract the dataset using:

   ```bash
   tar -xvf phoenix-2014.v3.tar.gz
   ```

   It is recommended to create a symbolic link to the extracted directory for convenient access:

   ```bash
   ln -s PATH_TO_DATASET/phoenix2014-release ./dataset/phoenix2014
   ```

3. The original image resolution is 210×260. For data augmentation, the image sequences are resized to 256×256. Run the following command to generate the gloss dictionary and process the resized image sequences.

   ```bash
   cd ./preprocess
   python dataset_preprocess.py --process-image --multiprocessing
   ```


## Training

The priorities of configuration files are: command line > config file > default values of argparse. To train the SLR model, run the command below:

```bash
python main.py --device your_device
```
## Inference

### PHOENIX2014 dataset

We provide the pretrained models for inference, you can download them from:

| Backbone | WER on Dev | WER on Test | Pretrained model |
| -------- | ----- | ----- | ------------- |
| ResNet34 | 18.1% | 18.4 | [[Google Drive]](https://drive.google.com/drive/folders/1AqLRcXkNomaim2HkXl1XT4IpFnQesobE?usp=sharing) |

​ To evaluate the pretrained model, run the command below：

```bash
python main.py --config ./configs/baseline.yaml --device your_device --load-weights path_to_weight.pt --phase test
```
 
## Thanks

This repo is based on [VAC (ICCV 2021)](https://ieeexplore.ieee.org/abstract/document/9710385), [CorrNet (CVPR 2023)](https://ieeexplore.ieee.org/document/10205442) and [USTM (arxiv 2025)](https://arxiv.org/abs/2512.13415)！