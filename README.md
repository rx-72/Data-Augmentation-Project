# Goal:
Using ZORRO as test measure to determine the robustness of a dataset in response to worst case data uncertainty. By fine tuning the error injection and robustness models, we can determine the susceptibility of a dataset towards providing an incorrect representation of the ground truth. The promise of this research is the ability to not only determine a measure of how robust a given dataset is, but also develop a method to reverse engineer error uncertainty in datasets to capture the ground truth as close as possible


### To do:

1. Read on influencing functions https://christophm.github.io/interpretable-ml-book/influential.html (pattern mining section)

2. Write a new code that can compound all methods into a single plot based on method type

3. Run tests messing around with histogram model

4. Test environment.yml file

### Minimum recommended system requirements: 
8 cpu, 16 GM RAM on dsmlp

### How to run 
1. Clone repository and change directory location to it (git clone -> cd)
   
2. Run "conda env create -f environment.yml" -> "conda activate Robustness_test_dependencies"
   
3. run  python run.py --dataset {(uploaded datasets)} --metric {(chosen metric)}"
   

      dataset: {"mpg", "ins"}

      metric: {"mae", "mae"}


   The tests will print a heat map showcasing robustness ratio deterioration across the chosen method of running and a naive random indice error injection on Meyer and Zorro.

Runs baseline or complex code robustness measurements on randomized seed of indexes for error injection followed by creating directory called ".../outputs/graph/" containing a heat map plot of resulting "uncertainty range X unceratinty size" robustness tests on Meyer. and ZORRO.

You'll need to declare what test you want to run: (baseline or complex), what dataset to run it upon: (cancer, mpg or ins) and what metric if you're going to use a complex test (accuracy, mse, etc.)

Note that the baseline does not run any metric and that the metrics that should be used for the cancer dataset (accuracy, precision, etc.) differ to the metric that should be used on ins and mpg datasets (mse, mae, etc.)


### References:
https://gopher-sys.github.io/index.html#papers - Gopher source implementation

https://arxiv.org/pdf/2405.18549 - ZORRO Basis and code reference paper




