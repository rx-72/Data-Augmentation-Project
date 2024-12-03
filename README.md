# Goal:
Using ZORRO as test measure to determine the robustness of a dataset in response to worst case data uncertainty. By fine tuning the error injection and robustness models, we can determine the susceptibility of a dataset towards providing an incorrect representation of the ground truth. The promise of this research is the ability to not only determine a measure of how robust a given dataset is, but also develop a method to reverse engineer error uncertainty in datasets to capture the ground truth as close as possible.

### Minimum recommended system requirements: 
8 cpu, 16 GM RAM on dsmlp

### How to run 
1. Clone repository and change directory location to it (git clone -> cd)
   
2. Run "conda env create -f environment.yml" -> "conda activate Robustness_test_dependencies"
   
3. run  python run.py --dataset {(uploaded datasets)} --test {(chosen test)}"
   

      dataset: {"mpg", "ins"}

      metric: {"discretization", "leve_one_out"}


   The tests will print a heat map showcasing robustness ratio deterioration utilizing a specific chosen model and metric. Multiple heatmaps will be generated per hyperparameter combination including one heatmap based on a naive random indice error injection on Meyer and Zorro.

You'll need to declare what test you want to run: (discretization or leave_one_out) and what dataset to run it upon: (mpg or ins). Note there's no baseline test since it's automatically generated as a heatmap as well for comparison on both tests.


### References:
https://gopher-sys.github.io/index.html#papers - Gopher source implementation

https://arxiv.org/pdf/2405.18549 - ZORRO Basis and code reference paper




