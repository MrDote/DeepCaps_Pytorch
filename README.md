# DeepCaps for Galaxy Morphology Classification

This implementation is done by referring to the official implementation of DeepCaps by [1], a PyTorch implementation [2] and the official paper at https://arxiv.org/abs/1904.09546. 

## How to Use with Your Own Custom Dataset
To train on your own custom dataset, simply change the required parameters in `cfg.py` and write your own class to load the dataset in `load_data.py`. Finally, replace line 19 in `train.py` appropriately to point to your custom class. No further changes should be required to train the model. The training can be executed with 
```
python train.py
```

## DeepCaps on Galaxy Zoo Images
As part of my Masters project, the DeepCaps network was trained on part of the SDSS DR7 dataset with corresponding labels taken from the Galaxy Zoo 2 project [3] and the Simard et al. structural parameter catalogue [4]. The network was trained on greyscale images for 200 epochs.


## References

[1] https://github.com/brjathu/deepcaps

[2] https://github.com/HopefulRational/DeepCaps-PyTorch

[3] https://data.galaxyzoo.org

[4] https://ui.adsabs.harvard.edu/abs/2011ApJS..196...11S/abstract