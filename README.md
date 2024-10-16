# LMPDnet
[Deep unrolled primal dual network for TOF-PET list-mode image reconstruction](https://arxiv.org/pdf/2410.11148)

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
