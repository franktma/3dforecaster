# 3dforecaster
Instant predictor for 3D printing time

The jupyter notebooks:
1. TwoStepModel_Results.ipynb # This is the master notebook
2. Datasets.ipynb # This is the script that merges and produces the datasets from the separate Trimesh, Cura datasets
3. Data_Exploration.ipynb, Data_Exploration2.ipynb # These are initial explorations of the data, produces instructive plots to understand the data
4. Basic_Benchmark.ipynb # basic example of the analysis
5. Basic_Train_Test.ipynb # example for training and testing models.

Helper classes
TDPPredictor.py # This is the helper classes that the jupyter notebooks use to fit, predict models

Example model
huber.pkl # This is the example output of the robust linear fit

The finished product can be found here: http://www.3dforecaster.com
