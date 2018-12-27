# ExploriPy
Pre-Modelling Analysis of the data, by doing various exploratory data analysis and Statistical Test.

**Installation Steps**

```  
pip install exploripy
``` 

**Important Parameters**
* data : Complete Dataset, should be a Pandas DataFrame. 
* CategoricalFeatures : List of Categorical Features
* title : Title which should appear in the top of the HTML file

**Installation Steps**

```
from exploripy import EDA
df = pd.read_csv('LoanPrediction.csv',na_values = 'nan')
CategoricalFeatures = ['Gender','Married','Dependents','Education','Self_Employed','Credit_History','Property_Area','Loan_Status','Loan_Amount_Term']
eda = EDA(df,CategoricalFeatures,title='Data Analysis on a Public Data')
eda.EDAToHTML()
```

**Output**

**Categorical Variables**
<p>
<img src='doc_images/Categorical.png'>
