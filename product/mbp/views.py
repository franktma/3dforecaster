# flask imports
from flask import render_template, flash
from mbp import app
#from sqlalchemy import create_engine
#from sqlalchemy_utils import database_exists, create_database
import pandas as pd
#import psycopg2
from flask import Flask, request, redirect, url_for
from werkzeug.utils import secure_filename
import os

# tremesh
import trimesh

# analysis imports
import numpy as np
#import matplotlib.pyplot as plt

# sklearn
from sklearn.preprocessing import scale
import sklearn.linear_model as skl_lm
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import statsmodels.api as sm
import statsmodels.formula.api as smf
# persistence
from sklearn.externals import joblib

from TDPPredictor import TDPRegressor as tdr
def apply_model(Vmes, Vch, Vbb, Sa, X, Z):
	print 'apply model on:',Vmes,Vch,Vbb,Sa,X,Z
	scaler,regr=joblib.load('huber.pkl')
	X = np.array([[Vmes, Vch, Vbb, Sa, X, Z]])

	tdregr = tdr(features=X)
	tdregr.import_model(scaler,regr)
	tdregr.transform()
	tdregr.predict()	
	# get result protect against negative prediction
	t=tdregr.yhat[0] if tdregr.yhat[0] >0 else 1 # The minimum time for printing
	print 'prediction:',t
	return t;

#
# Home page
#
@app.route('/index')
def index():
	user1={ 'nickname': 'Frank' }
	return render_template("index.html",
	   title = 'Home',
	   user = user1)

#
# STL File based Input/Output pages
#
#@app.route('/upload_part')
#def upload_part():
#   return render_template('upload_part.html')

@app.route('/')
@app.route('/home')
def input_fancy():
   return render_template('input_fancy.html')

@app.route('/about')
def about():
   return render_template('about.html')

@app.route('/contact')
def contact():
   return render_template('contact.html')

ALLOWED_EXTENSIONS = set(['stl'])
app.config['UPLOAD_FOLDER'] = 'uploads/'

def allowed_file(filename):
	return '.' in filename and \
	filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/predict', methods = ['GET', 'POST'])
def predict():
	# get file from flask's file handler
   if request.method == 'POST':
		# check if the post request has the file part
		# if 'file' not in request.files:
		# 	flash('No file part')
		# 	return redirect(request.url)
		file = request.files['stl_input_file']
      # if user does not select file, browser also
      # submit a empty part without filename
		# if file.filename == '':
		# 	flash('No selected file')
		# 	return redirect(request.url)

		# now we are in business
		if file and allowed_file(file.filename):
			filename = secure_filename(file.filename)
			filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
			print 'output file:', filepath
			file.save(filepath)

			file.save(secure_filename(filepath))
			print 'input file:', filename
			mesh = trimesh.load_mesh(filepath)
			Vmes=mesh.volume*0.001
			Vbb=mesh.bounding_box.volume*0.001
			Vch=mesh.convex_hull.volume*0.001
			Sa=mesh.area*0.01
			X=mesh.extents[0]*0.1
			Z=mesh.extents[2]*0.1
			Euler=mesh.euler_number
			that=apply_model(Vmes, Vch, Vbb, Sa, X, Z)/60. # conversion to hours
			neg_err = that*0.256
			pos_err = that*0.256
                        # cleanup
                        os.system('rm '+filepath)

			# get the truth

			data=pd.read_csv('../data/round2/batch2_all_viable_merged_v2.csv')
			from TDPPredictor import setup_df
			setup_df(data)

			truetime=-1
			truetime_str=''
			truthdata=data.loc[data['File']==filename,'t']
			if truthdata.empty == False:
				truetime=truthdata.iloc[0]/60.
				truetime_str=("(True time: %.1f hours)" % truetime)
			print 'truetime:', truetime
		return render_template("output_fancy.html",
			the_result = ("%.1f"%that), the_truth=truetime_str,
			the_neg_err="%.1f"%neg_err, the_pos_err="%.1f"%pos_err,
			the_filename = filename
			)

# set the secret key.  keep this really secret:
#app.secret_key = 'support volume'
