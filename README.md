# ExploriPy
[![PyPI version](https://badge.fury.io/py/ExploriPy.svg)](https://badge.fury.io/py/exploripy)
[![Downloads](https://pepy.tech/badge/ExploriPy)](https://pepy.tech/project/ExploriPy) 
<a href="https://github.com/Vibish/exploripy/blob/master/LICENSE">
  <img src="https://img.shields.io/github/license/exploripy/exploripy">
</a>
<a href="https://github.com/Vibish/exploripy/stargazers">
  <img src="https://img.shields.io/github/stars/Vibish/exploripy.svg">
</a>
<a href="https://github.com/Vibish/exploripy/network/members">
  <img src="https://img.shields.io/github/forks/Vibish/exploripy.svg">
</a>
[![Build Status](https://travis-ci.org/quintshekhar/exploripy.svg?branch=master)](https://travis-ci.org/quintshekhar/exploripy)
[![Build Status](https://dev.azure.com/shashanksist/shashank_sist/_apis/build/status/bhagatsajan0073.dominance-analysis?branchName=master)](https://dev.azure.com/shashanksist/shashank_sist/_build?definitionId=3)

Exploratory Data Analysis (EDA) is one of the crucial steps in data science that facilitates generating insights and statistical measures which are essential for building predictive models. EDA is always a time-consuming activity and require a thorough analysis of datasets to summarize their main characteristics. It is always required to do an initial analysis on the data, and then deep dive on further domain specific analysis, based on the initial insights. Currently, there is no comprehensive library in Python, which could do the initial Data Analysis and statistical tests, and present in an output, which could be easily interpreted shared across the stakeholders. Though there are several individual packages available for statistical tests, interpretation of the output requires certain level of statistical knowledge. <br>
ExploriPy reduces a data analyst’s efforts significantly in the initial EDA. It is designed in a way to perform automated EDA, and statistical tests including Analysis of Variance, Chi Square Test of Independence, Weight of Evidence, Information Value and Tukey Honest Significance Difference. It provides easy interpretation on these statistical test results, based on industry standard assumptions. It expects a Pandas DataFrame, along with a list of categorical variables, as input. Output will be a presentable HTML document, with the result of analysis and statistical tests, represented through several interactive charts, and tables (with option to download as CSV). The ExploriPy package is available in the Python Package Index.


[Installation Steps](#installation-steps) <br>
[Usage](#usage) <br>
[Parameters](#parameters) <br>
[Output](#output) <br>
&nbsp;   &nbsp;  [List of Features](#list-of-features) <br>
&nbsp;   &nbsp;  [Null Values](#null-values) <br>
&nbsp;   &nbsp;  [Target Variable](#target-variable) <br>
&nbsp;   &nbsp;  [Categorical Vs Target](#categorical-vs-target)<br>
&nbsp;   &nbsp;  [Continuous Vs Target](#continuous-vs-target)<br>

#### Installation Steps

```  
pip install ExploriPy
``` 
#### Usage
``` 
from ExploriPy import EDA
import pandas as pd
df = pd.read_csv('BigMartSales_Train.csv',na_values = 'nan')
CategoricalFeatures = ['Item_Identifier','Outlet_Identifier','Item_Fat_Content','Item_Type','Outlet_Establishment_Year','Outlet_Size','Outlet_Location_Type','Outlet_Type']
eda = EDA(df,CategoricalFeatures,OtherFeatures=['Outlet_Establishment_Year'],title='Exploratory Data Analysis for Big Mart Sales III - Based on Item_Outlet_Sales')
eda.TargetAnalysis('Item_Outlet_Sales') # For Target Specific Analysis
``` 

#### Parameters
* df = pandas.DataFrame
* CategoricalFeatures (Optional) = List of Categorical Features. If not given, application will consider the features, with Object Datatype, and has less than 2% unique categories, when compared to the total number of records
* OtherFeatures (Optional) = List of Categorical or Continuous Features. These features will not be considered in the analysis. If not given, application will consider features with Object datatype, which are not part of the CategoricalFeatures
*title = Title to be given in the output HTML document

Parameter for TargetAnalysis: <br>
* Target = This will be the target variable, against which all the analysis and statistical testes will be performed

#### Output
The output of the package is a HTML file with the following features.

#### List of Features
* List of Features available and their datatypes. 
<p><img src='ExploriPy/doc_images/List of Features.png'></p>

* Percentage of Categorical, Continuous and Other Variables.
<p><img src='ExploriPy/doc_images/Percentage of each type of Feature.png'> </p>

#### Null Values
Percentage of null values in each column. Additionally, a bar chart is also populated with the data.
<p><img src='ExploriPy/doc_images/Null Values.png'> </p> 

#### Target Variable
Info displayed for Target Variable: <br>
* Total Number of Records <br>
* Total Number of Nulls in Target <br>
* Percentage of Nulls in Target <br>

**For Categorical Target Variable:** 
* List of Categories and Number of records for each category
* Pie Chart on Percentage of Categorical Variables 
<p><img src='ExploriPy/doc_images/Target Categorical.png'> </p>

**For Continuous Target Variable** <br>
* Statistics on the Continuous Target Variable <br>
* Distribution of the Continuous Target Variable <br>
<p><img src='ExploriPy/doc_images/Target Continuous.png'> </p>

#### Categorical Vs Target 
List of Top 30 categories along with their count and percentage, for every categorical variable <br>
<p><img src='ExploriPy/doc_images/Categorical Count.png'> </p>

#### For Target Categorical Feature 
* Dependent list of variables, based on ChiSquare Test of Independence <br>
<i> Chi square test is done to test the independence of two Categorical variable. The test is used when both the independent variable and the dependent variable are both categorical. When two variables are statistically independent then they are not related to each other i.e. which category an object falls in the independent variable will not determine the category it will fall in the dependent variable. <br>
The P values gives the probability that the null hypothesis is true. Usually the null hypothesis states that there is no relationship between the two variables. </i>
<p><img src='ExploriPy/doc_images/Dependent Based on ChiSquare.png'> </p>
<br> 

* Grouped Bar Charts, to represent percentage of each category in the categorical feature, against each of the categories in the Target Variable.
<p><img src='ExploriPy/doc_images/Categorical Vs Target Categorical.png'> </p>
<br> 

* Null Percentage for each of the category, when compared with number of null records in the Target variable. This will help to determine, if there is any stream of data, which has the problem.
 <p><img src='ExploriPy/doc_images/Categorical Vs Null Percentage in Target.png'> </p>

#### For Target Continuous Feature: 
* Influencing Categorical Variables based on ANOVA (Analysis of variance) <br>
<i>This test is done for a selected target variable and it is used for finding its most influencing and non-influencing categorical variable. A statistically influencing categorical variable is one where the group means are not the same and there is difference in at least two of them. </i>
 <p><img src='ExploriPy/doc_images/Influencing Categorical Variables - Based on ANOVA.png'> </p>
 
* Categories with Similar Distributions <br>
<i> Based on PostHoc Test (Tukey HSD), get the categories with similar distributions and plot the distributions as well.
Once we know that the categorical variable influences the target variable then the Tukey HSD - Post Hoc Test is done. The Null hypothesis is, there is no difference in the means of the distribution. This is done as the Anova test can only tell if the variables are significant and not where the significance lies. The test will compare all possible group of pairs. </i>
<p><img src='ExploriPy/doc_images/Categories with Similar Distributions.png'> </p>

    
#### Continuous Vs Target 
* Statistics for each of the Continuous Variables <br>
* Box Plot for each of the Continuous Variables <br>
* Histogram and Distribution for each of the Continuous Variables <br>
<p><img src='ExploriPy/doc_images/Stats on Continuous Variables.png'> </p>


#### Continuous Vs Target Categorical Feature 
* Influencing Categorical Variables based on ANOVA <br>
* Categories with Similar Distributions <br>
* Based on PostHoc Test (Tukey HSD), get the categories with similar distributions and plot the distributions as well. <br>

#### Continuous Vs Target Continuous Feature 

* Dependent Continuous Variables based on Correlation
<p><img src='ExploriPy/doc_images/Dependent variables based on Correlation.png'> </p><br> 
* Scatter Plot for Continuous with Continuous Target variable 
<p><img src='ExploriPy/doc_images/Scatter Plot.PNG'> </p>
* Correlation Heatmap, with all the continuous variables, considering only combinations with correlation >=0.5 or <=-0.5 
<p><img src='ExploriPy/doc_images/Correlation Heatmap.PNG'> </p>
                                                                                                                        
<hr>

### Authors & License
The ExploriPy package is released under a <a href="https://github.com/exploripy/exploripy/blob/master/LICENSE"> MIT License</a>. ExploriPy Python package has been developed by <a href="https://github.com/Vibish"> Kunjithapatham Sivakumar</a>, <a href="https://github.com/quintshekhar"> Shashank Shekhar</a> and <a href="https://github.com/bhagatsajan0073"> Sajan Bhagat</a>, . Pull requests submitted to the GitHub Repo are highly encouraged!

<hr>


 
