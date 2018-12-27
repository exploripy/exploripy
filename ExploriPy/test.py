import pandas as pd
from ExploriPy import EDA
german_credit_data =  pd.read_csv('ExploriPy/data/german_credit_data.csv')
CategoricalFeatures = ['Sex','Job','Housing','Saving accounts','Checking account','Credit amount','Purpose']
eda = EDA(german_credit_data,CategoricalFeatures,title='Testing the Package')
eda.EDAToHTML()

