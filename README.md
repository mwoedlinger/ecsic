# ECISC
Official code of our upcoming WACV paper "ECSIC: Epipolar Cross Attention for Stereo Image Compression" by Matthias Wödlinger, Jan Kotera, Manuel Keglevic, Jan Xu and Robert Sablatnig.

## Installation

Install the necessary packages from the `requirements.txt` file with pip:

```pip install -r requirements.txt```

## Training
Train a new model with train.py. Example:

```python train.py gpu_idx exp_name --config configs/rd_cs_ecsic_m48.json --log_dir log_dir [--options]```

`gpu_idx` and `exp_name` need to be specified. The model weights are saved under `log_dir/experiments/HASH_DATE_TIME` (where HASH is added to prevent collisons for experiments with the same EXP_NAME). 

## Testing
Test a model with test.py. Example:

```python test.py gpu_idx exp_name --resume_dir experiments/RD_curves/cs/0.01/```

where `gpu_idx` and `exp_name` need to be specified and `resume_dir` can be set to any path with a config file `config.json` and weights file `model.pt` in it.

## Rate distortion curves
Use generate_rd_curve.py to generate rate distortion curves. For this specifiy a list of lambda values in the command line as a string. E.g.:

```python generate_rd_curve.py gpu_dix exp_name --config configs/rd_cs_ecsic_m48.json --lmda "0.001, 0.01, 0.1"```

The program will then perform full train/test runs for all specified lmda values and store the results in a json file in `experiments/RD_curves/exp_name/results.json` and the model weights for the run with lambda=lmda in `experiments/RD_curves/exp_name/lmda`.

## Example
More examples can be found in our [paper](https://arxiv.org/abs/2307.10284).
![image](./assets/001860.png "Qualitative comparison for a sample from the InStereo2k dataset")

## Citation
If you use this project please cite our work

```
@article{wodlinger2023ecsic,
  title={ECSIC: Epipolar Cross Attention for Stereo Image Compression},
  author={W{\"o}dlinger, Matthias and Kotera, Jan and Keglevic, Manuel and Xu, Jan and Sablatnig, Robert},
  journal={arXiv preprint arXiv:2307.10284},
  year={2023}
}
```
