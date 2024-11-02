# Goal:
Using ZORRO as test measure to determine the robustness of a dataset in response to worst case data uncertainty. By fine tuning the error injection and robustness models, we can determine the susceptibility of a dataset towards providing an incorrect representation of the ground truth. The promise of this research is the ability to not only determine a measure of how robust a given dataset is, but also develop a method to reverse engineer error uncertainty in datasets to capture the ground truth as close as possible


### To do:
1. Add important index and sensitive label robust ratio function variant to "run.py"

2. Add dataset argument to parser

3. Adjust function differences across different datasets and run on other included datasets

4. Incorate one drop function to find "important values" of training data using different metrics

5. Organize repo for presentation

6. Fix tqdm not showing up in terminal run



### Minimum recommended system requirements: 
8 cpu, 16 GM RAM on dsmlp

### How to run 
1. Clone repository and change directory location to it (git clone -> cd)
2. run "pip install -r requirements.txt"
       a. May need to run "pip install --upgrade ipykernel" to be compatible for ipython 8.18.1
3. run "python run.py --test baseline"

Runs baseline code robustness measurements on randomized seed of indexes for error injection followed by creating directory called ".../outputs/graph/" containing a heat map plot of resulting "uncertainty range X unceratinty size" robustness tests on Meyer. and ZORRO


### References:
https://gopher-sys.github.io/index.html#papers - Gopher source implementation

https://arxiv.org/pdf/2405.18549 - ZORRO Basis and code reference paper




