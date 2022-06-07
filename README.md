# Metareasoning Approaches to Thermal Management During Image Processing

The execution and visualization code associated with my Master's Thesis.

There are five main scripts used for data generation.
1. 'main.py' : A shell script which defines which algorithms should be run, as well as what paramters the tests should use.
2. 'program.py' : The script which performs an individual test run with a given set of initial conditions. This corresponds to the psuedocode presented in Algorithm 1.
3. 'metareasoning.py' : The script which contains the three metareasoning approaches.
4. 'q_learning.py' : The script which contains the Q-Learning approach.
5. 'auxiliary_functions.py' : The script which contains the auxiliary functions used in the other scripts.

There are three additional scripts which are used to visualize the results of the tests.
1. 'VisualizationV2.ipynb' : The script which visualizes the results of the tests.
2. 'ThesisVis.ipynb' : A modified version of 'VisualizationV2.ipynb' which was used for my thesis. This is the script which works with the current data storage format.
3. 'ThesisQVis.ipynb' : A modified version of 'ThesisVis.ipynb' which was used for visulaizing the Q-Learning approach in my thesis.


Raspberry Pi 4B case designed by [Manuel on Thingiverse](https://www.thingiverse.com/thing:3723561).


© 2022 Michael K. Dawson Jr.
