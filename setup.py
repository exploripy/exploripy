from setuptools import setup

setup(name='ExploriPy',
      version='1.0.0',
      description='Pre-Modelling Analysis of the data by doing various exploratory data analysis and Statistical Test',
      url='https://github.com/Vibish/exploripy',
      author="Sajan Kumar Bhagat, Kunjithapatham Sivakumar, Shashank Shekhar",
	  author_email='bhagat.sajan0073@gmail.com, s.vibish@gmail.com, quintshekhar@gmail.com',
	  maintainer="Sajan Kumar Bhagat, Kunjithapatham Sivakumar, Shashank Shekhar",
	  maintainer_email='bhagat.sajan0073@gmail.com, s.vibish@gmail.com, quintshekhar@gmail.com',
      license='MIT',
      packages=['ExploriPy'],
      install_requires=[
          'pandas',
          'numpy',
          'Jinja2',
          'scipy',
          'seaborn',
          'matplotlib',
          'statsmodels',
          'sklearn'
      ],
      include_package_data=True,
      zip_safe=False)
