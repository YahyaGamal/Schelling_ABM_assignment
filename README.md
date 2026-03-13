<img src="https://github.com/YahyaGamal/Schelling_ABM_assignment/blob/main/Logo/Logo_1.png?raw=true" alt="drawing" width="125"/> <img src="https://github.com/YahyaGamal/Schelling_ABM_assignment/blob/main/Logo/Logo_2.png?raw=true" alt="drawing" width="125"/>

## Agent Based Models (ABMs) assignment and practical

The repository includes the following relevant folders and files:
1. Week 1 practical session
- `\Practical 1\Pandemic_SIR\`: a NetLogo pandemic model
- `\Practical 1\Predator_prey\`: a Python predator-prey model to practice the assignment
2. Week 2 practical session
- `\Practical_2\DogSim\`: a NetLogo model for pedestians movement inside parks, including dogwalkers
- `\Practical_2\Wealth_Boltzmann`: a Python wealth distribution model (using the Mesa library).
3. Assignment
- `\Assignment\assignment.ipynb`: a Jupyter notebook including a Schelling model and a set of assignment questions.

To run the models in the `assignment.ipynb` Jupyter notebooks, you need to install the following packages:
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