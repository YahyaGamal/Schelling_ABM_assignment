<img src="https://github.com/YahyaGamal/Schelling_ABM_assignment/blob/main/Logo/Logo.png?raw=true" alt="drawing" width="125"/>

## Agent Based Models (ABMs) assignment and practical

The repository includes the following relevant files:
1. Assignment
- `\Assignment\assignment.ipynb`: a Python Jupyter notebook including:
    - Schelling model description
    - Code block for running the Schelling model with different input parameters
    - Assignment questions to be addressed by the students
- `\Assignment\schelling.py`: a Python file with the full code for the Schelling model. Understaning the details of this file is optional and is not required to answer the assignment questions.
2. Week 1 practical
- `\Practical_1\practical_1.ipynb`: a Python Jupyter notebook including:
    - Predator-prey model description
    - Code block for running the predator-prey model with different input parameters
    - Sample questions (to be familiar with the assignment questions)
- `\Practical_1\predator_prey.py`: a Python file with the full code for the predator-prey model. Understaning the details of this file is optional and is not required to answer the practical questions.

To run the model in the `assignment.ipynb` and `practical_1.ipynb` Jupyter notebooks, you need to install the following packages:
- `numpy` version 1.26.4
- `matplotlib` version 3.10.0
- `ipython` version 8.32.0

To install these libraries, run the following code block in the Jupyter notebooks:
```
! pip install numpy==1.26.4 matplotlib==3.10.0 ipython==8.32.0
```

Alternatively, run the following command line at the directory where the GitHub repository files have been downloaded and saved:
```
pip install -r requirements.txt
```

If you face issues with installing the Python libraries and cannot run the model in the Jupyter notebook, you can run an online version of the model through the following links:
1. Assignment: [Schelling NetLogo Web link](https://www.netlogoweb.org/launch#https://www.netlogoweb.org/assets/modelslib/Sample%20Models/Social%20Science/Segregation.nlogox)
2. Week 1 practical: [Predator-prey NetLogo Web link](https://www.netlogoweb.org/launch#https://www.netlogoweb.org/assets/modelslib/Sample%20Models/Biology/Wolf%20Sheep%20Predation.nlogox)