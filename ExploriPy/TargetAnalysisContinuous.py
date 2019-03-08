import pandas as pd
import numpy as np
import os
from jinja2 import Template
from jinja2 import Environment, FileSystemLoader
import random
import io
from scipy.stats import chi2_contingency
import scipy.stats as stats
from tqdm import tqdm
from scipy.stats import kurtosis
from scipy.stats import skew
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.stats.multicomp import MultiComparison
import statsmodels
import statsmodels.api as sm
from statsmodels.formula.api import ols
import warnings
warnings.filterwarnings('ignore')

class TargetAnalysisContinuous: 
	def __init__(self, df, CategoricalFeatures, ContinuousFeatures, OtherFeatures, target, title):
		''' Constructor for this class. '''
		
		self.df = df
		self.CategoricalFeatures = CategoricalFeatures
		self.ContinuousFeatures = ContinuousFeatures
		self.OtherFeatures = OtherFeatures
		self.target = target.replace(" ", "_")
		self.target = self.target.replace("(", "_")
		self.target = self.target.replace(")", "_")
		print(self.target)
		self.title = title
		
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
		
		
						
	def TargetSpecificAnalysis(self):
		'''
		Perform Target Specific Analysis and Render the HTML file
		'''
		
		filename = 'HTMLTemplate\\dist\\HTMLTemplate_target_Continuous.html'
			
		this_dir, this_filename = os.path.split(__file__)
		
		Template_PATH = os.path.join(this_dir, filename)
		
		with open(Template_PATH) as file:
			template = Template(file.read())			

		corr_features_df, CorrList, ColumnNames = self.CorrList()
		html = template.render(title = self.title
							,ListOfFields = self.ListOfFields()
							,FeatureTypes = self.CategoricalVsContinuous()
							,number_of_records = self.df.shape[0]
							,number_of_nulls = self.df[self.target].isnull().sum()
							,percentage_of_nulls = round((self.df[self.target].isnull().sum()*100) / self.df.shape[0],2)
							,target = self.target
							,target_statistics_dict = self.TargetStatistics()
							,NullValue = pd.DataFrame(round((self.df.isnull().sum()/self.df.shape[0])*100)).reset_index().rename(columns={'index': 'Feature',0:'NullPercentage'})
							,box_plot_target_filename = self.BoxPlot(self.target)
							,ContinuousHistogram_list = self.ContinuousHistogram()
							,Anova_df = self.Anova()
							,target_distribution_list = self.TargetDistributions_Categorical()
							,ContinuousFeaturesHistChart_df = self.ContinuousHistChart()
							,CorrList = CorrList
							,ColumnNames = ColumnNames
							,corr_features_df = corr_features_df
							,CategoricalFeatures = self.CategoricalFeatures
							,OtherFeatures = self.OtherFeatures
							,ContinuousFeatures = self.ContinuousFeatures
							#,SummarizeTwoWayAnova_list = self.SummarizeTwoWayAnova()
					   )
			
		out_filename = os.path.join(this_dir, 'HTMLTemplate\\dist\\result.html')
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
		NameTypeDict = []
		for name in list(self.df.columns.values):
			item = dict(name = name, type=self.df[name].dtype)
			NameTypeDict.append(item)
			
		return NameTypeDict		
	
		
	def CategoricalVsContinuous(self):
		'''
		Get statistics on the Feature Types and Assign colors for the Charts
		'''		
		# Choose 3 random colors from Selected Colors
		indices = random.sample(range(len(self.SelectedColors)), 3)
		colors=[self.SelectedColors[i] for i in sorted(indices)]
		FeatureTypes = []
		FeatureTypes.append(dict(Name = 'Categorical', Value = len(self.CategoricalFeatures), Color=colors[0]))
		FeatureTypes.append(dict(Name = 'Continuous', Value = len(self.ContinuousFeatures), Color=colors[1]))
		FeatureTypes.append(dict(Name = 'Others', Value = len(self.OtherFeatures), Color=colors[2]))
				
		return (FeatureTypes)
		
	def TargetStatistics(self):
		g1 = pd.DataFrame(self.df[self.target].describe())
		return(dict(Count = g1.loc['count'][0],
					Mean = g1.loc['mean'][0],
					Median = self.df[self.target].median(),
					STD = g1.loc['std'][0],
					Min =  g1.loc['min'][0],
					TwentyFive = g1.loc['25%'][0],
					Fifty = g1.loc['50%'][0],
					SeventyFive = g1.loc['75%'][0],
					Max = g1.loc['max'][0],
					Variance = np.var(self.df[self.target]),
					kurtosis = kurtosis(self.df[self.target]),
					skew = skew(self.df[self.target])))
					
	def BoxPlot(self, feature):		
		fig, ax = plt.subplots()
		ax = sns.boxplot(y=self.df[feature], ax=ax)
		box = ax.artists[0]
		indices = random.sample(range(len(self.SelectedColors)), 2)
		colors=[self.SelectedColors[i] for i in sorted(indices)]
		box.set_facecolor(colors[0])
		box.set_edgecolor(colors[1])
		sns.despine(offset=10, trim=True)
		this_dir, this_filename = os.path.split(__file__)
		OutFileName = os.path.join(this_dir, 'HTMLTemplate/dist/output/'+feature + '.png')		
		plt.savefig(OutFileName)
		
		return OutFileName
		
	def RegPlot (self, feature, target):
		fig, ax = plt.subplots()		 
		#color=self.SelectedColors[random.sample(range(len(self.SelectedColors)), 1)] 
		ax = sns.regplot(x=feature, y=target, data=self.df, ax=ax, color=random.choice(self.SelectedColors))
		
		sns.despine(offset=10, trim=True)
		this_dir, this_filename = os.path.split(__file__)
		OutFileName = os.path.join(this_dir, 'HTMLTemplate/dist/output/'+feature + '_regPlot.png')		
		plt.savefig(OutFileName)
		
		return OutFileName
		
	def ContinuousHistogram(self):
		'''
			Accumulate Histogram details for each of the Continuous Variable
		'''
		ContinuousHistogram_list = []
		print('Generating Histogram...')
		for column in tqdm(self.ContinuousFeatures):
			Edges, EdgesValues, Hist, HistValues, PDF, Color1, Color2 = self.HistChart_cont(column)
			ContinuousHistogram_list.append(dict(Name = column								
									,ImageFileName = self.BoxPlot(column)								
									,Hist = Hist
									,HistValues = HistValues
									,Edges = Edges
									,EdgesValues = EdgesValues
									,PDF = PDF
									,Color1 = Color1
									,Color2 = Color2								
									))
		return ContinuousHistogram_list
		
	def HistChart_cont (self, var):
		'''
			Generate Values for Histogram
		'''		
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
		
		return edges,edgesValues, hist, histValues, pdf, colors[0], colors[1]
		
		
		
	def Anova(self):
		"""		
		Calculate the F-Score (One Way Anova) for each of Categorical Variables with all the Continuous Variables.
		Output --> List of Continuous Variables, whose pValue is < 0.05
		"""
		target = self.target		
		AnovaList = []
		print('Performing ANOVA...')
		for CategoricalVar in tqdm(self.CategoricalFeatures):
			temp_df = self.df[[CategoricalVar, target]].dropna()
			f,p = stats.f_oneway(*[list(temp_df[temp_df[CategoricalVar]==name][target]) for name in set(temp_df[CategoricalVar])])
			AnovaList.append(dict(Categorical = CategoricalVar, PValue = p))
		Anova_df = pd.DataFrame(AnovaList)
		if Anova_df.shape[0]>0:
			Anova_df = Anova_df[Anova_df['PValue']<=0.05]
			Anova_df.sort_values(['PValue'],ascending = True, inplace=True)
		
		return Anova_df
		
			
	def TargetDistributions_Categorical(self):
		'''
			Get the Target Distribution for each (top) category for all the CategoricalFeatures
		'''
		print('Calculate the Target Distributions...')
		target_distribution_list = []
		for feature in tqdm(self.CategoricalFeatures):
			# '''
				# Get Top 5 categories
			# '''
			# g1 = pd.DataFrame(df[feature].value_counts()).reset_index()
			# g1.columns = ['category','count']
			# g1.sort_values(['count'],ascending=False,inplace=True)
			# top_categories = list(g1.head(5)['category'])
			
			if self.target != feature:
				#feature_categories,target_d_list,category_colors = self.TargetDistribution(feature)
				target_null_distribution = self.TargetNullDistribution(feature)
				CategoriesCount_df = self.CategoriesCount(feature)
				GroupTukeyHSD_df = self.GroupTukeyHSD(self.df[self.target],self.df[feature].astype(str))
				tukey_histogram_list = self.TukeyHistogram(GroupTukeyHSD_df, feature)				
				
				target_distribution_list.append(dict(feature = feature, target_null_distribution = target_null_distribution, CategoriesCount_df = CategoriesCount_df, GroupTukeyHSD_df = GroupTukeyHSD_df, tukey_histogram_list = tukey_histogram_list))
		
		return target_distribution_list
		
		
	def CategoriesCount(self, var):
		'''
		Number of Categories in each of the Categorical Features
		'''		
		CategoricalFeatures = self.CategoricalFeatures
		CategoriesCount = []
		
		df = self.df[var].groupby(self.df[var]).agg(['count'])
		df.index.names = ['Name']
		df.columns = ['Value']
		df.sort_values(['Value'], ascending = False, inplace=True)
		df = df.head(30)
		
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
		# CategoriesCount.append(dict(Count = df))
		
		return df
		
	def getRandomColors(self,no_of_colors):
		'''
		Generate Random Colors
		'''
		colors = []
		for i in range(0,no_of_colors):
			color = (random.randint(0,255),random.randint(0,255),random.randint(0,255))
			colors.append('#%02x%02x%02x' % color)
		
		return colors 
		
	def TargetNullDistribution(self,feature):
		'''
			Get the percentage of each category, which corresponds to Null in Target
		'''
		target = self.target
		g1 = pd.DataFrame(self.df[self.df[target].isnull()][feature].fillna('NAN').value_counts()).reset_index()		
		g1.columns = ['category','null_count']
		
		df = self.df[[target, feature]]
		df[feature] = df[feature].fillna('NAN')
		g2 = pd.DataFrame(df.groupby([feature]).size().reset_index())
		g2.columns = ['category','total_count']
		g1 = g1.merge(g2, how='left', on='category')
		g1['null_percent'] = round((g1['null_count']*100)/g1['total_count'],2)
		g1.drop(['null_count','total_count'], axis=1,inplace = True)		
		g1.sort_values('null_percent',ascending = False, inplace = True)
		g1 = g1.head(10)		
		
		return g1.T.to_dict().values()
		
	def GroupTukeyHSD(self,continuous, categorical):
		
		mc = MultiComparison(continuous, categorical)
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
		'''
			Once Tukey HSD test is done. Select only those entries, with Reject=False. 
			This implies, only entries with similar distribution is selected.
			Once selected, group them into different distributions.
		'''		
		TukeyResult_false = TukeyResult[TukeyResult['Reject']=='False']
		overall_distribution_list = []
		same_distribution_list = []		
		if len(TukeyResult_false) > 0:
			for group1 in TukeyResult_false['Group 1'].unique():
				if group1 not in overall_distribution_list:
					temp_list=[]
					temp_result = TukeyResult_false[TukeyResult_false['Group 1']== group1]
					overall_distribution_list.append(group1)
					for entry in list(temp_result['Group 2'].unique()):
						if entry not in overall_distribution_list:
							overall_distribution_list.append(entry)
							temp_list.append(entry)
					temp_list.append(group1)
			#         if temp_result['Group 2'].nunique()>1:
			#             temp_list.extend((temp_result['Group 2'].unique()))
			#         else:
			#             temp_list.append((temp_result['Group 2'].unique()[0]))
					same_distribution_list.append(dict(list_name=group1.replace(" ", "_"), lists=temp_list, length=len(temp_list)))
			if len(set(categorical.unique())-set(overall_distribution_list)) >0:
				missing_categories = list(set(categorical.unique())-set(overall_distribution_list))
				for group1 in missing_categories:
					same_distribution_list.append(dict(list_name=group1.replace(" ", "_"), lists=[group1], length=1))

		else:
			for group1 in categorical.unique():
				same_distribution_list.append(dict(list_name=group1.replace(" ", "_"), lists=[group1], length=1))
		
		g1 = pd.DataFrame(same_distribution_list)
		return (g1.sort_values('length',ascending=False))
		
		
	def TukeyHistogram(self, GroupTukeyHSD_df, CategoricalFeature):
		target = self.target
		tukey_histogram_list = []
		i=0
		for index, row in GroupTukeyHSD_df.iterrows():
			cat_list = pd.DataFrame({'category':row['lists']})
			cat_name = row['list_name']
			df = self.df[[CategoricalFeature,target]]
			df[CategoricalFeature] = df[CategoricalFeature].astype(str)
			df = df.merge(cat_list, left_on=CategoricalFeature, right_on='category', how='inner')			
			edges,edgesValues, hist, histValues, pdf, color1, color2 = self.HistChart(list(df[target].dropna()))
			
			if i >= len(self.SelectedColors) - 2:
				tukey_histogram_list.append(dict(category = cat_name, edges = edges,edgesValues = edgesValues, hist = hist, histValues = histValues, pdf = pdf, color1 = self.AllColors[i], color2 = color2))
			else: 
				tukey_histogram_list.append(dict(category = cat_name, edges = edges,edgesValues = edgesValues, hist = hist, histValues = histValues, pdf = pdf, color1 = self.SelectedColors[i], color2 = color2))
			i = i+1
			
		return tukey_histogram_list
			
			
	def HistChart (self, var):
		h = var
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

		return edges,edgesValues, hist, histValues, pdf, colors[0], colors[1]		
			
	
	def ContinuousHistChart(self):
		target = self.target
		print('Generating Distribution Charts...')
		ContinuousFeaturesHistChart_list = []
		for column in tqdm(self.ContinuousFeatures):	
			if column != self.target:
				hist_list = []
				# Get the stats on the Continuous Variable
				g1 = pd.DataFrame(self.df[column].describe())						
				edges,edgesValues, hist, histValues, pdf, color1, color2 = self.HistChart(list(self.df[column].dropna()))
				hist_list.append(dict(category = column, edges = edges,edgesValues = edgesValues, hist = hist, histValues = histValues, pdf = pdf, color1 = color1, color2 = color2))
				boxPlotFileName = self.BoxPlot(column)
				regPlotFileName = self.RegPlot(column,self.target)
				
				ContinuousFeaturesHistChart_list.append(dict(ContinuousFeature=column, hist_values = hist_list,
																Count = g1.loc['count'][0],
																Mean = g1.loc['mean'][0],
																Median = self.df[column].median(),
																STD = g1.loc['std'][0],
																Min =  g1.loc['min'][0],
																TwentyFive = g1.loc['25%'][0],
																Fifty = g1.loc['50%'][0],
																SeventyFive = g1.loc['75%'][0],
																Max = g1.loc['max'][0],
																Variance = np.var(self.df[column]),
																kurtosis = kurtosis(self.df[column]),
																skew = skew(self.df[column]),
																boxPlotFileName = boxPlotFileName,
																regPlotFileName = regPlotFileName
																
																))
			
		return pd.DataFrame(ContinuousFeaturesHistChart_list)
			
		
	def CorrList (self):
		'''
			Show the correlation heatmap 
		'''
		print('Generating Correlation Heatmap...')
		target = self.target
		df = self.df[self.ContinuousFeatures]
		CorrDf = df.corr()
		corr_df = pd.DataFrame(CorrDf[(CorrDf[target]>=0.5)|(CorrDf[target]<=-0.5)][target]).reset_index()
		corr_df.columns = ['ContinuousFeature','Correlation']
		corr_df = corr_df[corr_df['ContinuousFeature']!=target]
		CorrList = []
		MasterList = []		
		for col in CorrDf.columns:
			for index,row in CorrDf.iterrows():
				CorrList.append(0 if (row[col] > -0.5) & (row[col] <0.5) else row[col])
			MasterList.append(','.join([str(round(x,4)) for x in CorrList]))
			CorrList = []
		
		return corr_df, MasterList, ','.join("'{0}'".format(x) for x in CorrDf.columns)
	
	def SummarizeTwoWayAnova(self):
		SummarizeTwoWayAnova_list = []
		print('Performing two-way ANOVA...')
		for i in tqdm(range(len(self.CategoricalFeatures) - 1)):
			for j in range(i+1,len(self.CategoricalFeatures)):
				if self.TwoWayAnova(self.CategoricalFeatures[i],self.CategoricalFeatures[j],self.target):
					SummarizeTwoWayAnova_list.append(dict(categorical1 = self.CategoricalFeatures[i],categorical2=self.CategoricalFeatures[j])) 
			
		return SummarizeTwoWayAnova_list
		
	def TwoWayAnova(self,categorical1, categorical2, continuous):
		df = self.df[[categorical1,categorical2,continuous]]
		df = df.dropna()
		
		function = continuous + ' ~ C(' + categorical1 + ')*C('+ categorical2 + ')' 
		print(function)
		lm = ols(function, data=df).fit(method='powell')
		table = sm.stats.anova_lm(lm, typ=3)
		return table.iloc[2]['PR(>F)']<0.05