<img src="https://github.com/YahyaGamal/Schelling_ABM_assignment/blob/main/Logo/Logo.png?raw=true" alt="drawing" width="125"/>

## Schelling Agent Based Model (ABM) assignment

The repository includes the following relevant files:
1. `assignment.ipynb`: a Python Jupyter notebook including:
    - Schelling model description
    - Code block for runnin the Schelling model with different input parameters
    - Assignment questions to be addressed by the students
2. `schelling.py`: a Python file with the full code for the Schelling model. Understaning the details of this file is optional and is not required to answer the assignment questions.

To run the model in the `assignment.ipynb` Jupyter notebook, you need to install the following packages:
- `numpy` version 1.26.4
- `matplotlib` version 3.10.0
- `ipython` version 8.32.0

To install these libraries, run the following command (executed in a command prompt window at the directory where the GitHub repository files have been downloaded and saved):
```
pip install -r requirements.txt
```

Alternatively, you can run the following command to install the libraries without navigating the command prompt to the directory where the `requirements.txt` file is saved
```
pip install numpy==1.26.4 matplotlib==3.10.0 ipython==8.32.0
```

If you face issues with installing the Python libraries and cannot run the model in the Jupyter notebook, you can run an online version of the model through this [NetLogo Web link](https://www.netlogoweb.org/launch#https://www.netlogoweb.org/assets/modelslib/Sample%20Models/Social%20Science/Segregation.nlogo) (note that the `similarity_threshold` is labelled as `%-similar-wanted` in the NetLogo version).