# PyRes
Python Package to calculate image resolution using decorrelation analysis

# How to cite us

Li, R., Della Maggiora, G., Andriasyan, V., Petkidis, A., Yushkevich, A., Deshpande, N., ... & Yakimovich, A. (2024). Microscopy image reconstruction with physics-informed denoising diffusion probabilistic model. Communications Engineering, 3(1), 186.

```
@article{li2024microscopy,
  title={Microscopy image reconstruction with physics-informed denoising diffusion probabilistic model},
  author={Li, Rui and Della Maggiora, Gabriel and Andriasyan, Vardan and Petkidis, Anthony and Yushkevich, Artsemi and Deshpande, Nikita and Kudryashev, Mikhail and Yakimovich, Artur},
  journal={Communications Engineering},
  volume={3},
  number={1},
  pages={186},
  year={2024},
  publisher={Nature Publishing Group UK London}
}
```


# Installation

```bash
pip install https://github.com/casus/pyres.git
```

# To try out the examples and understand the usage of PyRes
Clone the repository and run the following commands

```bash
git clone https://github.com/casus/pyres.git
```
    
```bash
cd examples
```
    
```python
python main_imageDecorr.py
```

## Note: 
Please check for the updated version of the package before using it. The package is still under development and the API might change in the future.

# About the package
This is a python package to calculate the resolution of an image using decorrelation analysis. The package is based on the paper at https://www.nature.com/articles/s41592-019-0515-7 and the code is based on the MATLAB implementation of the paper by https://github.com/Ades91/ImDecorr
