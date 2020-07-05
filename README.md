# research.lpca

This project analyzes the results of various models for Link Prediction on Knowledge Graphs using Knowledge Graph Embeddings.
It allows to replicate the results in our work "Knowledge Graph Embeddings for Link Prediction: A Comparative Analysis".



## Models
We include 16 models representative of various families of architectural choices.
For each model we used the best-performing implementation available.

* DistMult: 
    - [Paper](https://arxiv.org/pdf/1412.6575) 
    - [Implementation](https://github.com/Accenture/AmpliGraph)
* ComplEx-N3: 
    - [ComplEX Paper](http://proceedings.mlr.press/v48/trouillon16.pdf) 
    - [ComplEX-N3 Paper](https://arxiv.org/pdf/1806.07297.pdf)
    - [Implementation](https://github.com/facebookresearch/kbc)
* ANALOGY:
    - [Paper](https://arxiv.org/pdf/1705.02426.pdf)
    - [Implementation](https://github.com/quark0/ANALOGY)
* SimplE:
    - [Paper](https://www.cs.ubc.ca/~poole/papers/Kazemi_Poole_SimplE_NIPS_2018.pdf)
    - [Implementation](https://github.com/baharefatemi/SimplE)
* HolE:
    - [Paper](http://www.aaai.org/ocs/index.php/AAAI/AAAI16/paper/download/12484/11828)       - [Implementation](https://github.com/Accenture/AmpliGraph)
* TuckER:
    - [Paper](https://arxiv.org/pdf/1901.09590.pdf)
    - [Implementation](https://github.com/ibalazevic/TuckER)
* TransE: 
    - [Paper](http://papers.nips.cc/paper/5071-translating-embeddings-for-modeling-multi-relational-data.pdf)
    - [Implementation](https://github.com/Accenture/AmpliGraph)
* STransE:
    - [Paper](https://arxiv.org/pdf/1606.08140)
    - [Implementation](https://github.com/datquocnguyen/STransE)
* CrossE:
    - [Paper](https://arxiv.org/pdf/1903.04750.pdf)
    - [Implementation](https://github.com/wencolani/CrossE)
* TorusE:
    - [Paper](https://aaai.org/ocs/index.php/AAAI/AAAI18/paper/viewPDFInterstitial/16227/15885)
    - [Implementation](https://github.com/TakumaE/TorusE)
* RotatE:
    - [Paper](https://openreview.net/pdf?id=HkgEQnRqYQ)
    - [Implementation](https://github.com/DeepGraphLearning/KnowledgeGraphEmbedding)
* ConvE:
    - [Paper](https://aaai.org/ocs/index.php/AAAI/AAAI18/paper/viewPDFInterstitial/17366/15884)
    - [Implementation](https://github.com/TimDettmers/ConvE)
* ConvKB:
    - [Paper](http://aclweb.org/anthology/N18-2053)
    - [Implementation](https://github.com/daiquocnguyen/ConvKB)
* ConvR: 
    - [Paper](https://www.aclweb.org/anthology/N19-1103.pdf) 
    - (implementation kindly shared by the authors privately)
* CapsE: 
    - [Paper](https://www.aclweb.org/anthology/N19-1226) 
    - [Implementation](https://github.com/daiquocnguyen/CapsE)
* RSN: 
    - [Paper](http://proceedings.mlr.press/v97/guo19c/guo19c.pdf) 
    - [Implementation](https://github.com/nju-websoft/RSN)


* We also employ the rule-based model AnyBURL as a baseline.
    - [Paper](http://web.informatik.uni-mannheim.de/AnyBURL/meilicke19anyburl.pdf)
    - [Implementation](http://web.informatik.uni-mannheim.de/AnyBURL/)



## Project

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
 


We note that
- In WN18RR, as reported by the authors of the dataset, a small percentage of test facts feature entities not included in the training set, so no meaningful predictions can be obtained for these facts. A few implementations (e.g. Ampligraph, ComplEx-N3) would actively skip such facts in their evaluation pipelines. Since the large majority of systems would keep them, we have all models include them in order to provide the fairest possible setting.
- In YAGO3-10 we observe that a few entities appear in two different versions depending on HTML escaping policies or on capitalisation. In these cases, odels would handle each version as a separate, independent entity; to solve this issue we have performed deduplication manually. The duplicate entities we have identified are:
    - Brighton_&\_Hove_Albion_F.C. and Brighton_\&amp;\_Hove_Albion_F.C.
    - College_of_William_&\_Mary and College_of_William_\&amp;\_Mary
    - Maldon_&\_Tiptree_F.C. and Maldon_\&amp;\_Tiptree_F.C. 
    - Alaska_Department_of_Transportation_&\_Public_Facilities and Alaska_Department_of_Transportation_\&amp;\_Public_Facilities
    - Turing_award and Turing_Award
     




### How to run the project (Linux/MacOS)
- Open a terminal shell;
- Create a new folder named `comparative_analysis` in your filesystem by running command: 
  ```bash
  mkdir comparative_analysis
  ```
- Download and the `datasets` folder and the `results` folder from [our storage](https://uniroma3-my.sharepoint.com/:f:/g/personal/pmerialdo_os_uniroma3_it/Ehhvyg1JQ7NDvhqCWVUWQT0Bj9N12I7C6-C3WwcaBHIw6g?e=hoRcS4), and move them into the `comparative_analysis` folder. Be aware that the files to download occupy around 100GB overall.


- Clone this repository under the same `comparative_analysis` folder with command:
  ```bash
  git clone https://github.com/merialdo/research.lpca.git analysis
  ```
  
- Open the project in folder `comparative_analysis/analysis` (using a Python IDE is suggested). 
  - Access file ```comparative_analysis/analysis/config.py``` and update ```ROOT``` variable with the absolute path of your "comparative_analysis" folder.
  - In order to replicate the plots and experiments performed in our work, just run the corresponding Python scripts in the `paper` folders mentioned above.
    By default, these experiments will be run on dataset `FB15K`.
    In order to change the dataset on which to run the experiment, just change the value of variable `dataset_name` in the script you wish to launch.
    Acceptable values are `FB15K`, `FB15K_237`, `WN18`, `WN18RR` and `YAGO3_10`.

Please note that the data in folders `datasets` and `results` are required in order to launch most scripts in this repository.
Those data can also be obtained by running the various scripts in folder `dataset_analysis`, that we include for the sake of completeness.

The global performances of all models on both `min` and `avg` tie policies can be printed on screen by running the the script `print_global_performances.py`.
