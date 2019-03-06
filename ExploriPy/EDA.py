import pandas as pd
import numpy as np
import io
from ExploriPy import FeatureType
from ExploriPy import WOE_IV
from ExploriPy import TargetAnalysisCategorical
from ExploriPy import TargetAnalysisContinuous
from jinja2 import Template
from jinja2 import Environment, FileSystemLoader
import random
from scipy.stats import chi2_contingency
import seaborn as sns
import matplotlib.pyplot as plt 
import scipy.stats as stats
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from statsmodels.stats.multicomp import MultiComparison
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.base import BaseEstimator, TransformerMixin
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from scipy.stats import kurtosis
from scipy.stats import skew
import time
import os
from sklearn.metrics import auc
from tqdm import tqdm


class EDA: 
	def __init__(self,df,CategoricalFeatures=[],OtherFeatures=[],filename="index.html",VIF_threshold=5,debug='NO',title='Exploratory Data Analysis'):
		''' 
		Constructor for this class. 
		'''
		self.df = df
		self.df.columns = [col.replace(" ", "_") for col in df.columns]
		self.df.columns = [col.replace("(", "_") for col in df.columns]
		self.df.columns = [col.replace(")", "_") for col in df.columns]
		self.SelectedColors = ["#5D535E", "#9A9EAB","#DFE166","#D9B44A","#4F6457","#4B7447","#8EBA43","#73605B","#D09683","#6E6702","#C05805","#DB9501","#50312F","#E4EA8C","#3F6C45","#B38867","#F69454","#A1BE95","#92AAC7","#FA6E59","#805A3B","#7F152E"]
		
		self.AllColors = ["#f2e1df","#ead6d5","#e3ccca","#dbc2c0","#d4b8b5","#ccaeaa","#c5a3a0","#bd9995","#b68f8b","#ae8480","#a77a75","#a0706b","#986660","#915c56","#89514b","#824740",
		"#7a3d36","#73322b","#6b2821","#641e16","#fdedec","#f5e3e1","#eed8d7","#e6cecc","#dec4c1","#d7b9b6","#cfafac","#c7a5a1","#c09a96","#b8908c","#b08681","#a97b76",
		"#a1716c","#9a6661","#925c56","#8a524c","#834741","#7b3d36","#73332b","#6c2821","#641e16","#ebe2ef","#e3d8e7","#dacedf","#d2c4d8","#cabad0","#c1b0c8","#b8a6c0",
		"#b09cb8","#a892b0","#9f88a8","#967da1","#8e7399","#866991","#7d5f89","#745581","#6c4b79","#644172","#5b376a","#522d62","#4a235a","#dfe9f0","#d5e0e9","#cad8e1",
		"#bfcfda","#b5c6d2","#aabdca","#9fb5c3","#95acbb","#8aa3b4","#809aac","#7592a4","#6a899d","#608095","#55788e","#4a6f86","#40667e","#355d77","#2a546f","#204c68",
		"#154360","#e1edf4","#d6e4ed","#ccdce6","#c1d4e0","#b7ccd9","#adc3d2","#a2bbcb","#98b3c4","#8daabd","#83a2b6","#799ab0","#6e91a9","#6489a2","#59819b","#4f7894",
		"#45708d","#3a6887","#306080","#255779","#1b4f72","#ddf0ed","#d2e9e5","#c7e1dc","#bcdad4","#b2d2cc","#a7cbc4","#9cc4bc","#91bcb3","#86b4ab","#7bada3","#70a69b",
		"#659e93","#5a968a","#4f8f82","#44887a","#3a8072","#2f786a","#247161","#196a59","#0e6251","#ddeeea","#d2e6e2","#c7ded9","#bcd5d0","#b1cdc8","#a6c5bf","#9bbdb6",
		"#90b5ad","#85ada5","#7aa49c","#6e9c93","#63948b","#588c82","#4d8479","#427c70","#377468","#2c6b5f","#216356","#165b4e","#0b5345","#deefe6","#d4e7dc","#c9dfd3",
		"#bed8c9","#b4d0c0","#a9c8b6","#9ec0ad","#94b8a3","#89b09a","#7ea890","#74a187","#69997e","#5f9174","#54896b","#498161","#3f7958","#34724e","#296a45","#1f623b",
		"#145a32","#dff3e8","#d5ecdf","#cae4d6","#c0ddcd","#b6d6c4","#abcfba","#a0c8b1","#96c0a8","#8cb99f","#81b296","#76ab8d","#6ca484","#629c7b","#579572","#4c8e68",
		"#42875f","#388056","#2d784d","#237144","#186a3b","#f9f3dc","#f4edd1","#efe6c6","#eae0bb","#e5dab0","#e0d4a5","#dbce9a","#d6c78f","#d1c184","#ccbb78","#c7b56d",
		"#c2af62","#bda857","#b8a24c","#b39c41","#ae9636","#a9902b","#a48920","#9f8315","#9a7d0a","#7D6608","#f9eedc","#f4e6d1","#efdfc6","#ead8bb","#e6d1b0","#e1caa5",
		"#dcc29a","#d7bb8f","#d2b484","#cdac7a","#c8a56f","#c39e64","#be9759","#b9904e","#b48843","#b08138","#ab7a2d","#a67222","#a16b17","#9c640c","#f6e9de","#f0e0d4",
		"#e9d8c9","#e2cfbe","#dcc6b3","#d5bda8","#ceb49e","#c8ac93","#c1a388","#ba9a7e","#b49173","#ad8868","#a7805d","#a07752","#996e48","#93653d","#8c5c32","#855427",
		"#7f4b1d","#784212","#f4e4da","#eddbcf","#e6d1c4","#dfc7b8","#d8beac","#d1b4a1","#caaa96","#c3a08a","#bc977f","#b48d73","#ad8367","#a67a5c","#9f7050","#986645",
		"#915c3a","#8a532e","#834923","#7c3f17","#75360c","#6e2c00","#e1e3e5","#d6d9dc","#cccfd2","#c1c5c9","#b7bbc0","#adb1b6","#a2a7ac","#989da3","#8d939a","#838a90",
		"#798086","#6e767d","#646c74","#59626a","#4f5860","#454e57","#3a444e","#303a44","#25303b","#1b2631","#dfe2e4","#d5d8da","#cacdd1","#c0c3c7","#b5b9bd","#abafb3",
		"#a0a5a9","#969aa0","#8b9096","#80868c","#767c82","#6b7278","#61676f","#565d65","#4c535b","#414951","#373f47","#2c343e","#222a34","#17202a"]
		
		start1 = time.time()
		featureType = FeatureType.FeatureType(df,CategoricalFeatures,OtherFeatures)		
		
		self.CategoricalFeatures = featureType.CategoricalFeatures()
		self.NonCategoricalFeatures = featureType.NonCategoricalFeatures()
		self.ContinuousFeatures = featureType.ContinuousFeatures()
		self.OtherFeatures = featureType.OtherFeatures()
		self.BinaryCategoricalFeatures = featureType.BinaryCategoricalFeatures()
		self.NonBinaryCategoricalFeatures = featureType.NonBinaryCategoricalFeatures()
		
		end1 = time.time()
		# print("Time to get the feature types = ",end1-start1)
		
		self.filename = filename
		self.VIF_threshold = VIF_threshold
		self.debug = debug
		self.title = title 
		
		# change the datatypes to str for all the categorical variables
		print("Converting Categorical Features to String...")
		for feature in tqdm(self.CategoricalFeatures):
			if self.df[feature].dtype == np.number:
				self.df[feature] = np.where(pd.isnull(self.df[feature]),self.df[feature],self.df[feature].astype(str))
			
			# self.df[feature] = self.df[feature].astype(str)
			# self.df[feature] = np.where(self.df[feature]=='nan',np.NaN,self.df[feature])
	
	def TargetAnalysis(self,target):
		'''
		Target Specific Analysis
		'''
		print("Initiating Target Specific Analysis...")
		start = time.time()
		target = target.replace(" ", "_")
		target = target.replace("(", "_")
		target = target.replace(")", "_")
		if target in self.CategoricalFeatures:
			targetAnalysis = TargetAnalysisCategorical.TargetAnalysisCategorical(self.df, self.CategoricalFeatures, self.ContinuousFeatures, self.OtherFeatures, target, self.title)
		elif target in self.ContinuousFeatures:
			targetAnalysis = TargetAnalysisContinuous.TargetAnalysisContinuous(self.df, self.CategoricalFeatures, self.ContinuousFeatures, self.OtherFeatures, target, self.title)
		end = time.time()
		# print("Time to initialize TargetAnalysis = ",end-start)
		targetAnalysis.TargetSpecificAnalysis()
		
		
		return

	def EDAToHTML(self,out=None):
		'''
		Main function to be called
		'''
		filename = 'HTMLTemplate\\dist\\HTMLTemplate_V2.html'
		this_dir, this_filename = os.path.split(__file__)
		
		Template_PATH = os.path.join(this_dir, filename)

		with open(Template_PATH) as file:
			template = Template(file.read())

		CorrList, ColumnNames = self.CorrList()
		WOEList,SummaryWOEList = self.WOEList()
		
		if SummaryWOEList.shape[0]>0:
			IVSummary = SummaryWOEList.sort_values('IV',ascending=False)
			IVStrongPredictor = IVSummary[IVSummary['IV']>=0.1]
			IVWeakPredictor = IVSummary[IVSummary['IV']<0.1]
			ChiSqSummary = SummaryWOEList.sort_values('PValue',ascending=True)
			ChiSqDependent = ChiSqSummary[ChiSqSummary['PValue']<=0.05]
			ChiSqIndependent = ChiSqSummary[ChiSqSummary['PValue']>0.05]
		else:
			IVSummary = pd.DataFrame()
			IVStrongPredictor = pd.DataFrame()
			IVWeakPredictor = pd.DataFrame()
			ChiSqSummary = pd.DataFrame()
			ChiSqDependent = pd.DataFrame()
			ChiSqIndependent = pd.DataFrame()
		
		AnovaList, SummaryAnovaList = self.Anova()
		if SummaryAnovaList.shape[0]>0:		
			SummaryAnovaList.sort_values('PValue',ascending=True,inplace=True)
			AnovaInfluencing = SummaryAnovaList[SummaryAnovaList['PValue']<=0.05]
			AnovaNonInfluencing = SummaryAnovaList[SummaryAnovaList['PValue']>0.05]
		else:
			SummaryAnovaList = pd.DataFrame()
			AnovaInfluencing = pd.DataFrame()
			AnovaNonInfluencing = pd.DataFrame()
		
		TTest = self.TTest()
		if TTest.shape[0]>0:
			TTest.sort_values('PValue',ascending=True,inplace=True)
			TTestDifferent = TTest[TTest['PValue']<=0.05]
			TTestNotDifferent = TTest[TTest['PValue']>0.05]
		else:
			TTest = pd.DataFrame()
			TTestDifferent = pd.DataFrame()
			TTestNotDifferent = pd.DataFrame()
		
			
		if(out):
			out_filename = out
		else:
			out_filename = os.path.join(this_dir, 'HTMLTemplate\\dist\\result.html')
			
		html = template.render(title = self.title
					   ,ListOfFields = self.ListOfFields()
					   ,CategoricalFeatures = self.CategoricalFeatures
					   ,OtherFeatures = self.OtherFeatures
					   ,ContinuousFeatures = self.ContinuousFeatures
					   ,BinaryCategoricalFeatures = self.BinaryCategoricalFeatures
					   ,NonBinaryCategoricalFeatures = self.NonBinaryCategoricalFeatures
					   ,FeatureTypes = self.CategoricalVsContinuous()
					   ,CategoriesCount = self.CategoriesCount()
					   ,WOEList = WOEList
					   ,IVStrongPredictor = IVStrongPredictor
					   ,IVWeakPredictor = IVWeakPredictor
					   ,ChiSqDependent = ChiSqDependent
					   ,ChiSqIndependent = ChiSqIndependent
					   ,ContinuousSummary = self.ContinuousSummary()
					   ,CorrList = CorrList
					   ,ColumnNames = ColumnNames
					   ,AnovaList = AnovaList
					   ,AnovaInfluencing = AnovaInfluencing
					   ,AnovaNonInfluencing = AnovaNonInfluencing
					   ,TTest = TTest
					   ,TTestDifferent = TTestDifferent
					   ,TTestNotDifferent = TTestNotDifferent
					   ,VIF_columns = self.VIF()
					   #,AUC_columns = self.AreaUnderCurve()
					   ,Variance = self.std_variance()
					   ,NullValue = pd.DataFrame(round((self.df.isnull().sum()/self.df.shape[0])*100)).reset_index().rename(columns={'index': 'Feature',0:'NullPercentage'})
					   ,ScatterImage = self.ScatterPlot()
					   )		   
		with io.open(out_filename, mode='w', encoding='utf-8') as f:
			f.write(html)
		
		import webbrowser
		url = 'file://'+out_filename
		webbrowser.open(url, new=2)
		return out_filename
		
	def ListOfFields(self):
		'''
		Get the list of fields from the DataFrame
		'''
		start = time.time()
		
		NameTypeDict = []
		for name in list(self.df.columns.values):
			item = dict(name = name, type=self.df[name].dtype)
			NameTypeDict.append(item)
			
		end = time.time()
		if self.debug == 'YES':
			print("ListOfFields",end - start)
		return NameTypeDict
		
	def CategoricalVsContinuous(self):
		'''
		Get statistics on the Feature Types and Assign colors for the Charts
		'''
		start = time.time()
		# Choose 3 random colors from Selected Colors
		indices = random.sample(range(len(self.SelectedColors)), 3)
		colors=[self.SelectedColors[i] for i in sorted(indices)]
		FeatureTypes = []
		FeatureTypes.append(dict(Name = 'Categorical', Value = len(self.CategoricalFeatures), Color=colors[0]))
		FeatureTypes.append(dict(Name = 'Continuous', Value = len(self.ContinuousFeatures), Color=colors[1]))
		FeatureTypes.append(dict(Name = 'Others', Value = len(self.OtherFeatures), Color=colors[2]))
		
		end = time.time()
		if self.debug == 'YES':
			print("CategoricalVsContinuous",end - start)
		return (FeatureTypes)
	
	def getRandomColors(self,no_of_colors):
		'''
		Generate Random Colors
		'''
		start = time.time()
		colors = []
		for i in range(0,no_of_colors):
			color = (random.randint(0,255),random.randint(0,255),random.randint(0,255))
			colors.append('#%02x%02x%02x' % color)
		end = time.time()
		if self.debug == 'YES':
			print('CategoricalVsContinuous',end-start)
		return colors 
	
	def CategoriesCount(self):
		'''
		Number of Categories in each of the Categorical Features
		'''
		start = time.time()
		CategoricalFeatures = self.CategoricalFeatures
		CategoriesCount = []

		for var in CategoricalFeatures:
			df = self.df[var].groupby(self.df[var]).agg(['count'])
			df.index.names = ['Name']
			df.columns = ['Value']
			
			if df.shape[0] > len(self.SelectedColors):
				if df.shape[0] > len(self.AllColors):
					colors = self.getRandomColors(df.shape[0])
				else:
					indices = random.sample(range(len(self.AllColors)), (df.shape[0]))
					colors=[self.AllColors[i] for i in sorted(indices)]				
			else:
				indices = random.sample(range(len(self.SelectedColors)), (df.shape[0]))
				colors=[self.SelectedColors[i] for i in sorted(indices)]
			
			df['Color'] = colors
			
			CategoriesCount.append(dict(Variable = var, Count = df))
		end = time.time()
		if self.debug == 'YES':
			print('CategoriesCount',end-start)
		return CategoriesCount
		
	
	def WOEList (self):
		'''
		Consolidate WOE, IV and ChiSq values
		'''
		start = time.time()
		woe = WOE_IV.WOE()
		WOEList = []
		SummaryWOEList = []
		InsightStat = "The variable \"{0}\" is {1} of the variable \"{2}\"."
		ChiSqInsight = "With the confidence limit of 0.05, the variable \"{0}\" is statistically {1} the variable \"{2}\""
		for DependentVar in self.CategoricalFeatures:
			for IndependentVar in self.CategoricalFeatures:
				if DependentVar != IndependentVar:
					# Update Weight Of Evidence(WOE) and Information Value (IV), only if the DependentVar is Binarys
					if DependentVar in self.BinaryCategoricalFeatures:
						df_independentVar = self.df[IndependentVar].dropna()
						df_dependentVar = self.df[DependentVar].dropna()
						WOE,IV = woe.woe_single_x(df_independentVar,df_dependentVar,event=df_dependentVar.unique()[0])
						if IV >= 0.3:
							IVInsight = InsightStat.format(IndependentVar,"strong predictor",DependentVar)
						elif IV >= 0.1:
							IVInsight = InsightStat.format(IndependentVar,"medium predictor",DependentVar)
						elif IV >= 0.02:
							IVInsight = InsightStat.format(IndependentVar,"weak predictor",DependentVar)
						else:
							IVInsight = InsightStat.format(IndependentVar,"very poor predictor",DependentVar)
					else:
						WOE = dict()
						IV = 0
						IVInsight = "NotBinary"
				
					# Calculate ChiSq and PValue. This is applicable even for binary variables
					
					ChiSq,PValue = self.ChiSquareOfDFCols(DependentVar,IndependentVar)
					
					if PValue <= 0.05:
						ChiSqInsight2 = ChiSqInsight.format(DependentVar, "dependent on", IndependentVar)
					else:
						ChiSqInsight2 = ChiSqInsight.format(DependentVar, "independent from", IndependentVar)
					
					# Add to the dictionary
					item = dict(DependentVar = DependentVar, IndependentVar = IndependentVar, WOE = WOE, IV = round(IV,2),
								IVInsight=IVInsight, ChiSq = ChiSq, PValue = PValue, ChiSqInsight = ChiSqInsight2)

					# Append to the list
					WOEList.append(item)

						
		for entry in WOEList:
			DependentVar = entry['DependentVar']
			IndependentVar = entry['IndependentVar']
			IV = entry['IV']
			ChiSq = entry['ChiSq']
			PValue = entry['PValue']
			SummaryWOEList.append(dict(DependentVar = DependentVar, IndependentVar = IndependentVar, IV = IV, ChiSq = ChiSq, PValue = PValue))
		
		end = time.time()
		if self.debug == 'YES':
			print('WOEList',end-start)	
			
		return WOEList,pd.DataFrame(SummaryWOEList)
		
	def ChiSquareOfDFCols(self, c1, c2):
		start = time.time()
		groupsizes = self.df.groupby([c1, c2]).size()
		ctsum = groupsizes.unstack(c1)
		end = time.time()
		if self.debug == 'YES':
			print('ChiSquareOfDFCols',end-start)
		
		return(list(chi2_contingency(ctsum.fillna(0)))[0:2])
		
	def ContinuousSummary(self):
		start = time.time()
		df = self.df[self.ContinuousFeatures]
		df = df.describe().transpose()
		VariableDetails = []
		for key,value in df.iterrows():
			Edges, EdgesValues, Hist, HistValues, PDF, Color1, Color2 = self.HistChart(key)
			VariableDetails.append(dict(Name = key
								,Count = value['count']
								,Mean = value['mean']
								,STD = value['std']
								,Min = value['min']
								,TwentyFive = value['25%']
								,Fifty = value['50%']
								,SeventyFive = value['75%']
								,Max = value['max']
								,Median = self.df[key].median()
								,ImageFileName = self.BoxPlot(key)								
								,Hist = Hist
								,HistValues = HistValues
								,Edges = Edges
								,EdgesValues = EdgesValues
								,PDF = PDF
								,Color1 = Color1
								,Color2 = Color2
								,Variance = np.var(self.df[key])
								,kurtosis = kurtosis(self.df[key])
								,skew = skew(self.df[key])
								))
		end = time.time()
		if self.debug == 'YES':
			print('ContinuousSummary',end-start)
		return VariableDetails
		
	def ScatterPlot(self):
		start = time.time()
		sns.set(style="ticks", color_codes=True)
		this_dir, this_filename = os.path.split(__file__)
		OutFileName = os.path.join(this_dir, 'HTMLTemplate/dist/output/Scatter.png')
		fig, ax = plt.subplots()
		ax = sns.pairplot(self.df[self.ContinuousFeatures].dropna(),markers="+",palette="husl",kind="reg", plot_kws={'line_kws':{'color':'orange'}})
		plt.savefig(OutFileName)
		end = time.time()
		if self.debug == 'YES':
			print('ScatterPlot',end-start)
		return OutFileName
		
	def BoxPlot(self,var):

		start = time.time()
		fig, ax = plt.subplots()
		ax = sns.boxplot(y=self.df[var], ax=ax)
		box = ax.artists[0]
		indices = random.sample(range(len(self.SelectedColors)), 2)
		colors=[self.SelectedColors[i] for i in sorted(indices)]
		box.set_facecolor(colors[0])
		box.set_edgecolor(colors[1])
		sns.despine(offset=10, trim=True)
		
		
		this_dir, this_filename = os.path.split(__file__)
		OutFileName = os.path.join(this_dir, 'HTMLTemplate/dist/output/'+var + '.png')
		
		plt.savefig(OutFileName)
		end = time.time()
		if self.debug == 'YES':
			print('BoxPlot',end-start)
		
		return OutFileName
		
	def HistChart (self, var):
		start = time.time()
		h = list(self.df[var].dropna())
		hist, edges = np.histogram(h, density=True, bins=35)
		histValues, edgesValues = np.histogram(h, density=False, bins=35)
		h.sort()
		hmean = np.mean(h)
		hstd = np.std(h)
		pdf = stats.norm.pdf(edges, hmean, hstd)
		
		hist = ','.join([str(round(x,5)) for x in hist])
		histValues = ','.join([str(x) for x in histValues])
		edges = ','.join([str(x) for x in edges])
		edgesValues = ','.join([str(x) for x in edgesValues])
		pdf = ','.join([str(round(x,5)) for x in pdf])
		indices = random.sample(range(len(self.SelectedColors)), 2)
		colors=[self.SelectedColors[i] for i in sorted(indices)]
		end = time.time()
		if self.debug == 'YES':
			print('HistChart',end-start)
		return edges,edgesValues, hist, histValues, pdf, colors[0], colors[1]
		
	def CorrList (self):
		start = time.time()
		df = self.df[self.ContinuousFeatures]
		CorrDf = df.corr()
		CorrList = []
		MasterList = []
		for col in CorrDf.columns:
			for index,row in CorrDf.iterrows():
				CorrList.append(row[col])
			MasterList.append(','.join([str(round(x,4)) for x in CorrList]))
			CorrList = []
		end = time.time()
		if self.debug == 'YES':
			print('CorrList',end-start)
		return MasterList, ','.join("'{0}'".format(x) for x in CorrDf.columns)
	
	def TTest(self):
		"""
		Calculate PValue based on T-Test. This is applicable only for Binary Categorical Variable with all the Continuous Variables
		"""
		temp_df = self.df.dropna()
		start = time.time()
		TList = []
		Insight1 = "With Confidence interval of 0.05, the distribution of variable - \"{0}\" varies significantly based on the categorical variable - \"{1}\". "
		Insight2 = "With Confidence interval of 0.05, the distribution of variable - \"{0}\" does not vary significantly based on the categorical variable - \"{1}\". "
		for CategoricalVar in self.BinaryCategoricalFeatures:
			for ContinuousVar in self.ContinuousFeatures:
				binary1 = temp_df[CategoricalVar].unique()[0]
				binary2 = temp_df[CategoricalVar].unique()[1]
				TStat,p = stats.ttest_ind(temp_df[temp_df[CategoricalVar]==binary1][ContinuousVar],temp_df[temp_df[CategoricalVar]==binary2][ContinuousVar])
				if p <= 0.05:
					Insight = Insight1.format(ContinuousVar,CategoricalVar)
				else:
					Insight = Insight2.format(ContinuousVar,CategoricalVar)
				TList.append(dict(Continuous=ContinuousVar,Categorical=CategoricalVar,TStat=TStat,PValue=p,Insight=Insight))
		
		end = time.time()
		if self.debug == 'YES':
			print('T-Test',end-start)
			print(pd.DataFrame(TList))
		return pd.DataFrame(TList)
		
	def Anova(self):
		"""		
		Calculate the F-Score (One Way Anova) for each of Categorical Variables with all the Continuous Variables
		"""
		# Drop records with Null values
		temp_df = self.df.dropna()
		start = time.time()
		AnovaList = []
		SummaryAnovaList = []
		Insight1 = "With Confidence interval of 0.05, the variable - \"{0}\" is influenced by the categorical variable - \"{1}\". "
		Insight2 = "As the Categorical variable - \"{0}\" is binary, Tukey's HSD test is not necessary. "
		Insight3 = "As the p-Value is higher than the Confidence Interval 0.05, the variable - \"{0}\" is not influenced by the categorical variable - \"{1}\". "
		for CategoricalVar in self.CategoricalFeatures:
			Binary = 'Yes' if CategoricalVar in self.BinaryCategoricalFeatures else 'No'
			for ContinuousVar in self.ContinuousFeatures:
				TukeyResult = None 
				f,p = stats.f_oneway(*[list(temp_df[temp_df[CategoricalVar]==name][ContinuousVar]) for name in set(temp_df[CategoricalVar])])
				if (p<0.05 and CategoricalVar in self.BinaryCategoricalFeatures):
					Insight = Insight1.format(ContinuousVar, CategoricalVar) + Insight2.format(CategoricalVar)
				elif p<0.05:
					TukeyResult = self.Tukey(CategoricalVar, ContinuousVar)
					Insight = Insight1.format(ContinuousVar, CategoricalVar)
				else:
					Insight = Insight3.format(ContinuousVar, CategoricalVar)
				AnovaList.append(dict(Categorical = CategoricalVar, Continuous = ContinuousVar, f = f, p = p, Binary = Binary, Insight = Insight,
				TukeyResult = TukeyResult))
		for entry in AnovaList:
			Categorical = entry['Categorical']
			Continuous = entry['Continuous']
			PValue = entry['p']			
			SummaryAnovaList.append(dict(Categorical=Categorical,Continuous=Continuous,PValue=PValue))
		
		end = time.time()
		if self.debug == 'YES':
			print('Anova',end-start)
		return AnovaList,pd.DataFrame(SummaryAnovaList)
		
	def Tukey(self,Categorical, Continuous):
		"""
		Calculate Tukey Honest Significance Difference (HSD) Test, to identify the groups whose
		distributions are significantly different
		"""
		temp_df = self.df.dropna()
		start = time.time()
		mc = MultiComparison(temp_df[Continuous], temp_df[Categorical])
		result = mc.tukeyhsd()
		reject = result.reject
		meandiffs = result.meandiffs
		UniqueGroup = mc.groupsunique
		group1 = [UniqueGroup[index] for index in mc.pairindices[0]]
		group2 = [UniqueGroup[index] for index in mc.pairindices[1]]
		reject = result.reject
		meandiffs = [round(float(meandiff),3) for meandiff in result.meandiffs]
		columns = ['Group 1', "Group 2", "Mean Difference", "Reject"]
		TukeyResult = pd.DataFrame(np.column_stack((group1, group2, meandiffs, reject)), columns=columns)
		
		end = time.time()
		if self.debug == 'YES':
			print('Tukey',end-start)
		return TukeyResult
		
	def std_variance(self):
		"""
		Scale the Continuous features with MinMaxScaler and then calculate variance
		"""
		start = time.time()
		scaler = MinMaxScaler()
		scaled = scaler.fit_transform(self.df[self.ContinuousFeatures].dropna())
		var_list = []
		i=0
		for column in self.ContinuousFeatures:
			var_list.append(dict(column=column,variance=np.var(scaled[:,i])))
			i=i+1
		end = time.time()
		if self.debug == 'YES':
			print('std_variance',end-start)
		return pd.DataFrame(var_list)
	
	def VIF(self):
		"""
		Drop the NaN's and calculate the VIF
		"""
		start = time.time()
		vif_list = []
		X = self.df[self.ContinuousFeatures].dropna()
		if len(list(self.ContinuousFeatures)) > 1:
			for var in X.columns:
				vif = variance_inflation_factor(X[X.columns].values,X.columns.get_loc(var))
				vif_list.append(dict(column=var,vif=vif))
			
		end = time.time()
		if self.debug == 'YES':
			print('VIF',end-start)
		return pd.DataFrame(vif_list)
	
	def AreaUnderCurve(self):
		"""
		Get the area under curve, if each scaled features are plotted against each other
		"""
		start = time.time()
		temp_df = self.df.copy()
		AUC_list = []
		le = LabelEncoder()
		scaler = MinMaxScaler()
		for feature in self.CategoricalFeatures:
			temp_df[feature] = le.fit_transform(pd.DataFrame(temp_df[feature]))
		
		cols_for_auc = self.CategoricalFeatures
		cols_for_auc.extend(self.ContinuousFeatures)
		for feature in cols_for_auc:
			temp_df[feature] = scaler.fit_transform(pd.DataFrame(temp_df[feature]))
			for feature_2 in cols_for_auc:
				if feature != feature_2:
					area_under_curve = auc(temp_df[feature],temp_df[feature_2],reorder=True)
					AUC_list.append(dict(dependent = feature, independent = feature_2, auc=area_under_curve))
					
		if self.debug == 'YES':
			print('AUC',end-start)		
			
		return pd.DataFrame(AUC_list)
		
		
			