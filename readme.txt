necessary packages:

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

This will run everything.
Your code will be in data_loader.py
There are explanations in there.

Take a look at data_utils.py for detaisl on classes and helper functions.


Also, this is for Python 2.7. Let me know if there are issues for 3.x and I'll try to fix them. There shouldn't be any major problems.

NOTE: The code takes ~30 sec to startup.
