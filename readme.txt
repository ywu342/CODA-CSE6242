This package has mainly source code for loading data, cleaning data and SVM ranking of the data. And a demo code to show how to integrate the whole thing. All the code should be under sources/ directory. In it, data_loader.py is the demo code that prints top 10 counties from an example svm ranking.

-----------------------------------------
This code is for Python 2.7. You will need to put the Data sources (you can download data and db from this link: https://github.com/ywu342/CODA-CSE6242) and county.db in the same root folder as sources.
Necessary packages to run the code:

pip install pandas
pip install flask
pip install sklearn
pip install scipy
pip install numpy

-- This should already be inside your code
pip install sqlite3

-----------------------------------------
how to run:
You should be in root directory, i.e. in the folder where you have 'county.db', 'Data sources', and 'sources'
From there, run:

python sources/data_loader.py

This will run a demo of the ranking svm. It will output the top 10 counties based on SVM ranking.

NOTE: The code takes ~30 sec to startup to load data.
