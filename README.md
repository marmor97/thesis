This repository contains the code resources for my Master Thesis at Copenhagen University.

The repository consists of a notebook to visualize the results as well as scripts to reproduce results.

``plot_results.ipynb``:  The notebook that shows results that have been collected and saved in the results folder.


``run.py``:  The script that produces the rouge values for a given system, data set and condition. One can adjust the parameters in the model in the configuration file run.yml

``calculate_perplexity.py``:  The script that produces perplexity values for a given system, data set and condition. One can adjust the parameters for the script in the configuration file calc_perp.yml

``check_prompts.py``:  The script that uses different prompt templates for the NQ dataset. Again, one can adjust the parameters for the script in the configuration file check_prompts.yml

Running the scripts require a Python version => 3.9. Make a local environment and install the packages by running the following commands: 

``python -m venv .env``


``source .env/bin/activate``

``pip install -r requirements.txt``

Now it is possible to run each of the scripts by typing:

``python script_name.py 'path/to/config'  'path/to/data'``

An example is 
``run.py 'config/run.yml'  'data'``
