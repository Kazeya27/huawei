## Environment
The code has been tested in the following environment:
* python=3.10.11
* numpy=1.24.3
* scipy=1.10.1
* pandas=1.5.3
* torch=2.0.1
* pyscamp=4.0.0

## Procedure
To reproduce the experiments, please follow the steps below:
1. Download the LargeST dataset from https://github.com/liuxu77/LargeST/
2. Download the LTSF-Benchmark dataset from https://github.com/cure-lab/LTSF-Linear/
3. In ```process_dataset.ipynb```, change the path on line 1 of the second block to the directory containing the LTSF-Benchmark dataset.
4. In ```process_dataset.ipynb```, change the path on line 4 of the second block to your desired output directory.
5. In ```process_dataset.ipynb```, change the path on line 1 of the fourth block to the directory containing the LargeST dataset.
6. In ```process_dataset.ipynb```, change the path on line 3 of the fourth block to your desired output directory.
7. Run ```process_dataset.ipynb``` to process the datasets.
8. In ```script_config.py```, change the paths on lines 14 and 15 to the directories containing the output datasets.
9. Run ```script_config.py``` to generate config files.
10. Run ```script_all_exp.py``` to execute the experiments.
11. Run ```script_result.py``` to print the experiment results.
