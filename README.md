# Maximum utility Based Arm Selection Strategy For Sequential Query Recommendations

The repo features the experimental analysis on [Maximum utility Based Arm Selection Strategy For Sequential Query Recommendations
](https://proceedings.mlr.press/v157/puthiya-parambath21a/puthiya-parambath21a.pdf).

```
@inproceedings{parambath2021max,
  title={Max-Utility Based Arm Selection Strategy For Sequential Query Recommendations},
  author={Parambath, Shameem Puthiya and Anagnostopoulos, Christos and Murray-Smith, Roderick and MacAvaney, Sean and others},
  booktitle={Asian Conference on Machine Learning},
  pages={564--579},
  year={2021},
  organization={PMLR}
}
```

## Experiment Dependency

* python3.8
* numpy
* scipy, with intel-mkl
* matplotlib
* sentence-transformers
* sklearn
* code is based on striatum package

## Experiment Data

* Data from an online literature discovery service

``` 
A non-original processed data (very small version) is provided just for the experimental purposes
```

## Experiment

* Before running, please modify the code *data.py* accordingly, then

```
python3 experiments.py
```

* Logs in human-understandable format are attached in *log*
