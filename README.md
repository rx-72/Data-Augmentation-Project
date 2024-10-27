# Data-Augmentation-Project


### To do:
1. Edit baseline to be presentatable in running

2. Include output logging via running

3. import leafy spurge dataset

# Zorro download:
https://anonymous.4open.science/r/Zorro-C8F3/micro_bench_test.ipynb

Recommended system requirements: 8 cpu, 16 GM RAM on dsmlp
(Note all below runtimes are assuming these requirements, can run faster with better systems)

## mpg-robustness-certification
Runs ZORRO on mpg train dataset after injecting errors. Errors include uncertain labels (whose performance is compared to "Meyer et al.") and uncertain features. Performance testing is based off of robustnest measurement of the final dataset. Also runs with a regularization parameter for both uncertain labels and features comparing robustness ratio result and worst case loss under given regularization parameter.

MPG dataset shape: 398 rows by 7 columns, 318 rows go to training and 80 rows for testing

### Running uncertain labels (Zorro and "Meyer et al"):
Zorro: 10 min.
"Meyer et al": 3 min.
Total: 13 - 14 min.

### Running uncertain features (Zorro):
Zorro: 50 min.

### Regularization uncertain labels (Zorro):
Zorro: 1h 30 min.

### Regularization uncertain features (Zorro):
Zorro: 3h 39 min.

### Running the baseline code:

1. Navigate to your terminal and clone repo via "git clone", then navigate to the repo via "cd"

2. "run.sh" is need to create the script environment. Run "chmod +x run.sh" to get "execute" permissions and then "./run.sh"

3. (optional but recommended) create a virtual environment to run the code upon (similar to conda) by running the following code:
       python -m venv venv
       "source venv/bin/activate" (Macbook) / "venv\Scripts\activate" (Windows)
       install -r requirements.txt

4. Run the baseline code via "python run_baseline.py"



