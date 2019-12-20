# research.lpca

This project analyzes the results of various models for Link Prediction on Knowledge Graphs using Knowledge Graph Embeddings.
It allows to replicate the results in our work "Knowledge Graph Embeddings for Link Prediction: A Comparative Analysis".

#### Language
The project is completely written in Python 3.

#### Dependencies
- numpy
- matplotlib
- seaborn

### Structure
The project is structured as a set of Python scripts, each of which can be run separately from the others:
- folder `efficiency` contains the scripts to visualize our results on efficiency of LP models.
  - Our findings for training times can be replicated by running script `barchart_training_times.py`
  - Our findings for prediction times can be replicated by running script `barchart_prediction_times.py` 
- folder `effectiveness` contains the scripts to obtain our results on the effectiveness:
  - folder `performances_by_peers` contains various scripts that show how the predictive performances of LP models vary, depending on the number of source and target peers of test facts.
  - folder `performances_by_paths` contains various scripts that show how the predictive performances of LP models vary, depending on the Relational Path Support of test facts.
  - folder `performances_by_relation_properties` contains various scripts that show how the predictive performances of LP models vary, depending on the properties of the relations of test facts.
  - folder `performances_by_reified_relation_degree` contains various scripts that show how the predictive performances of LP models vary, depending on the degree of the original reified relation in FreeBase.
- folder `dataset_analysis` contains various scripts to analyze the structural properties of the original datasets featured in our analysis (e.g. for computing the source peers and target peers for each test fact, or its Relational Path Support, etc).
We share the results we obtained using these scripts in ...

In each of these folders, the scripts to run in order to replicate the results of our paper are contained in the folders named `papers`.
 
The experiments we report in 

### How to run the project (Linux/MacOS)
- Open a terminal shell;
- Create a new folder "comparative_analysis" in your filesystem by running command: 
  ```bash
  mkdir comparative_analysis
  ```
- Download and unzip the datasets and the results for the LP models in our project from.... You can do this by running the following commands:
  ```bash
  wget ...
  unzip...
  ```
- Move the resulting folders "datasets" and "results" under "comparative_analysis"
  ```bash
  mv ...
  mv ...
  ```

- Clone this repository under the same "comparative_analysis" folder with command:
  ```bash
  git clone https://github.com/merialdo/research.lpca.git
  ```

- Open the project in folder ```comparative_analysis/analysis``` with any Python IDE. 
  - Access file ```comparative_analysis/analysis/config.json``` and update ```ROOT``` variable with the absolute path of your "comparative_analysis" folder.
  - In order to replicate the plots and experiments performed in our work, just run the corresponding Python scripts in the ```paper``` mentioned above.
    By default, these experiments will be run on dataset `FB15K`
    In order to change the dataset on which to run the experiment, just change the value of variable `dataset_name` in the script you wish to launch.
    Acceptable values are `FB15K`, `FB15K_237`, `WN18`, `WN18RR` and `YAGO3_10`.

Please note that the data in folders `datasets` and `results` are required in order to launch most scripts in this repository.
Those data can also be obtained by running the various scripts in folder `dataset_analysis`, that we include for the sake of completeness.

The global performances of all models on both `min` and `avg` tie policies can be printed on screen by running the the script `print_global_performances.py`.
