# LMPDnet
[Deep unrolled primal dual network for TOF-PET list-mode image reconstruction](https://iopscience.iop.org/article/10.1088/1361-6560/adf9b7)

## Set up

### Create a conda Environment
```
conda env create --name LMRecon python==3.8
conda activate LMRecon
pip install requirements.txt
```
###

### Set system environment variables
1. Download the file parallelproj.c_dll and create a new variable named PARALLELPROJ-C-LIB in the system variable, pointing to the path where parallelproj_c.dll is located
2. Download the file parallelproj_cuda.dll and create a new variable named PARALLELPROJ-CUDA-LIB in the system variable, pointing to the path where parallelproj_cuda.dll is located
3. Create a new variable named PYTHONPATH in the system variable to point to the current working directory.
###


### Acknowledgement

Our simulation and projection codes are based on [
parallelproj](https://github.com/gschramm/parallelproj). The simulated phantom is created based on the [FBSEM](https://github.com/Abolfazl-Mehranian/FBSEM), Many Thanks. 

### Citation
If you find our paper or repo useful, please consider citing our paper:
```
@article{Hu_2025,
    doi = {10.1088/1361-6560/adf9b7},
    url = {https://doi.org/10.1088/1361-6560/adf9b7},
    year = {2025},
    month = {aug},
    publisher = {IOP Publishing},
    volume = {70},
    number = {17},
    pages = {175012},
    author = {Hu, Rui and Li, Chenxu and Tian, Kun and Cui, Jianan and Chen, Yunmei and Liu, Huafeng},
    title = {Deep unrolled primal dual network for TOF-PET list-mode image reconstruction},
    journal = {Physics in Medicine & Biology},
}

```
