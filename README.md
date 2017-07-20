# 3dforecaster
Instant predictor for 3D printing time

The jupyter notebooks:
1. TwoStepModel_Results.ipynb

This is the final analysis for 3dforecaster that performs the model fit and cross-validation.
The final model consists of two stages
Run a classifier to separate the data into two sub-groups
Perform regression separately on each of the sub-groups
The notebook flows as follows
Initial data exploration
Initial simple modeling with only regression
The full two stage modeling
Cross-validation for both the single stage as well as the two stage modeling


2. Datasets.ipynb # This is the script that merges and produces the datasets from the separate Trimesh, Cura datasets

3. Data_Exploration.ipynb, Data_Exploration2.ipynb # These are initial explorations of the data, produces instructive plots to understand the data

4. Basic_Benchmark.ipynb # basic example of the analysis

5. Basic_Train_Test.ipynb # example for training and testing models.

Helper classes
TDPPredictor.py # This is the helper classes that the jupyter notebooks use to fit, predict models
Example use:

from TDPPredictor import TDPRegressor as tdr
lregtrain_ols = tdr(sd_train[['Vmes']].as_matrix(),sd_train['t'].as_matrix(),'ols')
lregtrain_ols.transform()
lregtrain_ols.fit()
lregtrain_ols.predict()
lregtrain_ols.PlotPerformance(1,1000,'log',False)

Example model
huber.pkl # This is the example output of the robust linear fit

The finished product can be found here: http://www.3dforecaster.com
