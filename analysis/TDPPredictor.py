import numpy as np
import matplotlib.pyplot as plt

# sklearn transformers
from sklearn.preprocessing import scale
from sklearn.preprocessing import StandardScaler
# sklearn models
import sklearn.linear_model as skl_lm
from sklearn.linear_model import HuberRegressor, Ridge
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble.forest import RandomForestRegressor

# sklearn metrics
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import statsmodels.api as sm
import statsmodels.formula.api as smf
from sklearn.metrics import confusion_matrix
# persistence
from sklearn.externals import joblib

def mad(arr):
    """ Median Absolute Deviation: a "Robust" version of standard deviation.
        Indices variabililty of the sample.
        https://en.wikipedia.org/wiki/Median_absolute_deviation 
    """
    arr = np.ma.array(arr).compressed() # should be faster to not use masked arrays.
    med = np.median(arr)
    return np.median(np.abs(arr - med))

class TDPRegressor:
	def __init__(self, features=[], target=[], model='ols',tag='train'):
		self.tag = tag+'_'+model
		#self.outdir = 'fig/Results'
		self.outdir = 'fig/XChecks'
		self.model=model
		import os
		os.system('mkdir -p '+self.outdir)

		# setup analysis
		self.X = features
		self.y = target

		# Scale
		self.scaler = StandardScaler(with_mean=True,with_std=True).fit(self.X)

		if model=='ols':
			self.regr = skl_lm.LinearRegression()
		elif model=='huber':
			self.regr = HuberRegressor(fit_intercept=True, alpha=0.0, max_iter=100, epsilon=1.35)
		elif model=='tree':
			self.regr = DecisionTreeRegressor(max_depth=6)
		elif model=='forest':
			self.regr = RandomForestRegressor(n_estimators=10,bootstrap=True, criterion='mae', max_depth=10, max_features='auto',
				min_samples_leaf=5, min_samples_split=10, random_state=0)
		print self


	def __repr__(self):
		return "Regression " + self.tag + " --- %.3d entries" % len(self.X)

	def Add(self,b):
		self.X=np.append(self.X,b.X,axis=0)
		self.y=np.append(self.y,b.y,axis=0)
		self.X_scaled=np.append(self.X_scaled,b.X_scaled,axis=0)
		self.yhat=np.append(self.yhat,b.yhat,axis=0)

	def transform(self):
		self.X_scaled =  self.scaler.transform(self.X)

	def fit(self):
		# Fit
		X_scaled = self.X_scaled
		self.regr.fit(X_scaled,self.y)
		if self.model =='ols' or self.model=='hubert':
			print 'coeffecients:'
			print(self.regr.intercept_)
			print(self.regr.coef_)
		if self.model == 'tree':
			print 'tree feature importances'
			print self.regr.feature_importances_
		if self.model == 'forest':
			#print 'random forest estimators'
			#print self.regr.estimators_
			print 'random forest nfeatures:', self.regr.n_features_
			print 'random forest feature importances'
			print self.regr.feature_importances_

	def predict(self):
		X_scaled = self.X_scaled
		self.yhat = self.regr.predict(X_scaled)
		if len(self.y) > 0:
			self.CalcErrorMetric()

	def CalcErrorMetric(self):
		X=self.X
		X_scaled = self.X_scaled
		y=self.y
		lin_rmse = np.sqrt(mean_squared_error(y,self.yhat))
		lin_ame = mean_absolute_error(y,self.yhat)
		lin_mad = mad(y-self.yhat)
		ymean = np.mean(y)
		self.frac_ame=lin_ame/ymean
		self.frac_err=lin_ame/ymean
		self.R2=r2_score(self.y,self.yhat)
		print 'residual standard error (rse):', lin_rmse, 'residual mean_absolute_error:', lin_ame, 'residual mad',lin_mad,lin_ame, '<y>: ', ymean
		print 'ratio (err): ', self.frac_err
		print 'R^2 score: ', self.R2

	def PlotInputs(self,xmin=0.5,xmax=5000,xc='linear'):
		# convenient
		X=self.X
		X_scaled= self.X_scaled
		y=self.y

		# vars to fit
		fig, axes = plt.subplots(nrows=1, ncols=1,figsize=(7,7))
		axes.scatter(X[:,0],y, color='red',marker='o',alpha=0.2)
		axes.set_xlabel('x', fontsize='xx-large')
		axes.set_ylabel('Time (min)', fontsize='xx-large')
		plt.xscale(xc)
		plt.yscale(xc)
		plt.xlim(xmin,xmax)
		plt.ylim(ymin=10,ymax=100000)    
		plt.savefig(self.outdir+'/x_vs_t.png')

	def PlotPerformanceSingle(self,xmin=1,xmax=5000,xc='linear'):
		# convenient
		X=self.X
		y=self.y
		yhat=self.yhat

		fig, axarr = plt.subplots(nrows=1, ncols=1, figsize=(7,7), sharex=False)
		axarr.scatter(X[:,0],y, color='red',marker='o',alpha=0.2,label='data')
		axarr.scatter(X[:,0],yhat, color='blue',marker='s',alpha=0.5, s=5)
		#ax.set_xlabel(r'$\Delta_i$', fontsize=15)
		axarr.set_xlabel('Volume (cm^3)', fontsize=20)
		axarr.set_ylabel('Time (min)', fontsize=20)
		axarr.yaxis.set_tick_params(labelsize=20)
		axarr.set_xscale('linear')
		axarr.set_yscale('linear')
		axarr.set_xlim(xmin,xmax)
		axarr.set_ylim(ymin=10,ymax=15000)
		fig.savefig(self.outdir+'/'+self.tag+'_data_model_vs_x.png')

	def PlotPerformance(self,xmin=1,xmax=5000,xc='linear',plotLeg=True):
		# convenient
		X=self.X
		y=self.y/60
		yhat=self.yhat/60

		# plot residual
		fig, axarr = plt.subplots(nrows=1, ncols=1, figsize=(7,7), sharex=False)
		#fig.subplots_adjust(hspace=0)
		# Two subplots, the axes array is 1-d

		axarr.scatter(X[:,0],y, color='red',marker='o',alpha=0.2,label='data')
		axarr.scatter(X[:,0],yhat, color='blue',marker='s',alpha=0.5, s=5)
		#ax.set_xlabel(r'$\Delta_i$', fontsize=15)
		axarr.set_xscale('linear')
		axarr.set_yscale('linear')
		axarr.set_xlim(xmin,xmax)
		axarr.set_ylim(ymin=0,ymax=250)
		axarr.xaxis.set_tick_params(labelsize=20)
		axarr.yaxis.set_tick_params(labelsize=20)
		axarr.set_xlabel('Volume (cm^3)', fontsize=20)
		axarr.set_ylabel('Build Time (hours)', fontsize=20)
		fig.savefig(self.outdir+'/'+self.tag+'_data_model_data_vs_x.png')

		fig, axarr = plt.subplots(nrows=1, ncols=1, figsize=(7,7), sharex=False)
		axarr.scatter(X[:,0],y-yhat,color='red',alpha=0.2)
		axarr.set_xscale(xc)
		axarr.set_yscale('linear')
		axarr.set_xlim(xmin,xmax)
		axarr.set_ylim(-50,50)
		axarr.xaxis.set_tick_params(labelsize=20)
		axarr.yaxis.set_tick_params(labelsize=20)
		axarr.set_xlabel('Volume (cm^3)', fontsize=20)
		axarr.set_ylabel('Build Time (hours)', fontsize=20)

		axarr.scatter(X[:,0].T,X[:,0].T*0, color='blue',marker='s',alpha=0.5, s=5,label=self.tag + '\nFrac Error = ' + "%.3f" % self.frac_ame +'\n'+r'$R^2$ = ' + "%.3f" % self.R2)

		if plotLeg:
			axarr.legend(loc='lower left',framealpha=0,fontsize=16)

		fig.savefig(self.outdir+'/'+self.tag+'_data_model_residual_vs_x.png')

	def export_model(self):
		from sklearn.externals import joblib
		joblib.dump([self.scaler,self.regr], self.outdir+'/'+self.model+'.pkl')

	def import_model(self,scaler,regr):
		self.scaler=scaler
		self.regr=regr
		print 'scaler:',scaler.mean_
		print 'regr coefficients:', regr.intercept_, regr.coef_


class TDPClassifier:
	def __init__(self, features, target, tag='train'):
		self.tag = tag
		self.outdir = 'fig/'+self.tag

		# setup training
		self.X = features
		self.y = target

		# Scale
		self.scaler = StandardScaler(with_mean=True,with_std=True).fit(self.X)

		from sklearn.linear_model import LogisticRegression
		self.clf = LogisticRegression()
		print self

	def __repr__(self):
		return "Classifier " + self.tag + " --- %.3d entries" % len(self.X)

	def transform(self):
		self.X_scaled = self.scaler.transform(self.X)

	def fit(self):
		self.clf.fit(self.X_scaled,self.y)
		print(self.clf.intercept_)
		print(self.clf.coef_)

	def predict(self):
		self.yhat = self.clf.predict(self.X_scaled)
		self.CalcErrorMetric()

	def CalcErrorMetric(self):
		X=self.X
		y=self.y
		print confusion_matrix(y,self.yhat)

#
# Some helper functions
#
def correlation_matrix(df):
    from matplotlib import pyplot as plt
    from matplotlib import cm as cm

    fig = plt.figure(figsize=(10,10))
    ax1 = fig.add_subplot(111)
    cmap = cm.get_cmap('jet', 30)
    cax = ax1.imshow(df[['Vmes','Vbb','Vch','Sa','Z']].corr(), interpolation="nearest", cmap=cmap)
    ax1.grid(True)
    plt.title('3D Printing Feature Correlation')
    labels=['Volume','Bounding Box','Conv Hull','Surf. Area','Z',]
    ax1.set_xticklabels(labels,fontsize=10)
    ax1.set_yticklabels(labels,fontsize=10)
    # Add colorbar, make sure to specify tick locations to match desired ticklabels
    fig.colorbar(cax, ticks=[.75,.8,.85,.90,.95,1])
    plt.show()

def setup_df(data):
	# aliases for slicer info
	data.rename(columns={'Runtime (s)':'t','Fill Vol (mm^3)':'Vfil','Support Vol (mm^3)':'Vsup','Layer Height':'dh'},inplace=True)
	# aliases for basic mesh info
	data.rename(columns={'Volume':'Vmes','Surface Area':"Sa",'BB Vol':'Vbb','BC vol':'Vbc','BS vol':'Vbs','CVHull Vol':'Vch','Euler Number':'Euler'},inplace=True)
	# aliases for engineered features
	data.rename(columns={'DownArea':'Sdown','Adj DownArea':'Vdown','Ang DownArea':'Sangover','Magic Number':'Vangover'},inplace=True)

	# convert unit
	print 'before conversion:', data.t[1]
	volvars=['Vfil','Vsup','Vmes','Vbb','Vbc','Vbs','Vch','Vdown','Vangover']
	areavars=['Sa','Sdown','Sangover']
	lenvars=['X','Y','Z']
	factor=0.1
	for i in range(len(lenvars)):
	    data[lenvars[i]]=data[lenvars[i]].apply(lambda x: factor*x)
	for i in range(len(areavars)):
	    data[areavars[i]]=data[areavars[i]].apply(lambda x: np.power(factor,2)*x)
	for i in range(len(volvars)):
	    data[volvars[i]]=data[volvars[i]].apply(lambda x: np.power(factor,3)*x)
	data['t']=data['t'].apply(lambda x:np.divide(x,60.))
	print 'after conversion:', data.t[1]

	# add branches
	data['sumV']=data['Vfil'] + data['Vsup']
	data['diffVbbVmes']=data['Vbb'] - data['Vmes']
	data['diffVchVmes']=data['Vch'] - data['Vmes']
	data['rSaVmes']=data['Sa'].div(data['Vmes'])
	data['rVbbVmes']=data['Vbb'].div(data['Vmes'])
	data['rVbcVmes']=data['Vbc'].div(data['Vmes'])
	data['rVbsVmes']=data['Vbs'].div(data['Vmes'])
	data['rVchVmes']=data['Vch'].div(data['Vmes'])
	data['rXVmes']=data['X'].div(data['Vmes'])
	data['rZVmes']=data['Z'].div(data['Vmes'])                       
	data.info()