import pandas as pd
import numpy as np

class FeatureType: 
	def __init__(self,df,CategoricalFeatures=[]):
		''' Constructor for this class. '''
		self.df = df
		self.likely_cat = {}
		if not CategoricalFeatures:
			for var in df.columns:
				self.likely_cat[var] = (self.df[var].nunique()>1) and \
				(1.*self.df[var].nunique()/self.df[var].count() < 0.01) and \
				(1.*self.df[var].nunique()/self.df[var].count() != 0)
		else:
			CategoricalFeatures = [col.replace(" ", "_") for col in CategoricalFeatures]
			CategoricalFeatures = [col.replace("(", "_") for col in CategoricalFeatures]
			CategoricalFeatures = [col.replace(")", "_") for col in CategoricalFeatures]
			for var in df.columns:
				self.likely_cat[var] = var in CategoricalFeatures
		
				
	def CategoricalFeatures(self):
		return [key for key, value in self.likely_cat.items() if value == True]
		
	def NonCategoricalFeatures(self):
		return [key for key, value in self.likely_cat.items() if value == False]
	
	def ContinuousFeatures(self):
		NonCatFeatures = self.NonCategoricalFeatures()
		return list(self.df[NonCatFeatures]._get_numeric_data().columns)
		#return [var for var in NonCatFeatures if self.df[var].dtype == np.number]
		
	def OtherFeatures(self):
		NonCatFeatures = self.NonCategoricalFeatures()
		return [var for var in NonCatFeatures if self.df[var].dtype != np.number]
		
	def BinaryCategoricalFeatures(self):
			return [name for name in self.df.columns if self.df[name].nunique() == 2]

	def NonBinaryCategoricalFeatures(self):
			return [name for name in self.CategoricalFeatures() if self.df[name].nunique() > 2]
			
