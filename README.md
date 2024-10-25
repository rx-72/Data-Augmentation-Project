# Data-Augmentation-Project


### To do:
1. Edit baseline to be presentatable in running

2. Include output logging via running

3. import leafy spurge dataset


### Running the baseline code:

1. Navigate to your terminal and clone repo via "git clone", then navigate to the repo via "cd"

2. "run.sh" is need to create the script environment. Run "chmod +x run.sh" to get "execute" permissions and then "./run.sh"

3. (optional but recommended) create a virtual environment to run the code upon (similar to conda) by running the following code:
       python -m venv venv
       "source venv/bin/activate" (Macbook) / "venv\Scripts\activate" (Windows)
       install -r requirements.txt

4. Run the baselien code via "python run_baseline.py"

### Downloading the VOC dataset
1. Navigate into the datasets folder within the project using the cd command
cd Data_Augmentation_Project/datasets

2. Download the VOC 2012 dataset into the datasets folder.
wget http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar