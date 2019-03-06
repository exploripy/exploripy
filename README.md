# ExploriPy
Pre-Modelling Analysis of the data, by doing various exploratory data analyses and Statistical Tests.

[Installation Steps](#installation-steps)
[Usage](#usage)
[Parameters](#parameters)
[Output](#output)
    [List of Features](#

#### Installation Steps

```  
pip install ExploriPy
``` 

**Important Parameters**
* data : Complete Dataset, should be a Pandas DataFrame. 
* CategoricalFeatures : List of Categorical Features
* title : Title which should appear in the top of the HTML file

**Installation Steps**

```
from ExploriPy import EDA
df = pd.read_csv('LoanPrediction.csv',na_values = 'nan')
CategoricalFeatures = ['Gender','Married','Dependents','Education','Self_Employed','Credit_History','Property_Area','Loan_Status','Loan_Amount_Term']
eda = EDA(df,CategoricalFeatures,title='Data Analysis on a Public Data')
eda.TargetAnalysis('Loan_Status') # For Target Specific Analysis
# eda.EDAToHTML() # For general analysis
```

**Output**


<p>
<img src='/ExploriPy/doc_images/Null.PNG'>


<p>
<img src='/ExploriPy/doc_images/Continuous.png'>


<p>
<img src='/ExploriPy/doc_images/Correlation Heatmap.PNG'>
  

<p>
<img src='/ExploriPy/doc_images/Scatter Plot.PNG'>


<p>
<img src='/ExploriPy/doc_images/Feature Reduction.PNG'>


<p>
<img src='/ExploriPy/doc_images/Categorical.PNG'>
  
**Categorical Vs Categorical**
<p>
Weight Of Evidence, Information Value, Chi-Sq Test of Independence
<img src='/ExploriPy/doc_images/WOE IV ChiSq.png'>

**Categorical Vs Continuous**
<p>
ANOVA and PostHoc Test (Tukey HSD)
<img src='/ExploriPy/doc_images/Anova TukeyHSD.png'>
