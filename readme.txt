This is for Python 2.7.
Necessary packages:

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

This will run everything. It will output the top 10 counties based on SVM ranking.

NOTE: The code takes ~30 sec to startup to load data.
