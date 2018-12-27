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

**Null Values**
<p>
<img src='/ExploriPy/doc_images/Null.PNG'>

**Continuous Variables**
<p>
<img src='/ExploriPy/doc_images/Continuous.png'>

**Correlation Heatmap**
<p>
<img src='/ExploriPy/doc_images/Correlation Heatmap.PNG'>
  
**Scatter Plot**
<p>
<img src='/ExploriPy/doc_images/Scatter Plot.PNG'>

**Feature Reduction**
<p>
<img src='/ExploriPy/doc_images/Feature Reduction.PNG'>

**Categorical Variables**
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
