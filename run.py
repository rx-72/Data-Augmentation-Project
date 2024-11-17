import os
import json
import pandas as pd
import argparse
import matplotlib.pyplot as plt
import warnings
from sklearn.exceptions import DataConversionWarning
from etl import *

warnings.filterwarnings('ignore')
warnings.filterwarnings(action='ignore', category=DataConversionWarning)

with open("data-params.json", "r") as f:
  params = json.load(f)

METRICS = {
    "accuracy": accuracy
}

def run_complex_test(X_train, y_train, X_test, y_test, output_dir, metric, modeltype, args, ratios, robustness_radius, max_uncertain_pct=10, maximize=True):
    def plot_heatmap(ax, heatmap_data, x_labels, y_labels, title):
        heatmap = ax.imshow(heatmap_data, cmap=cmap, interpolation='nearest', 
                        aspect='auto', alpha=0.8, vmin=0, vmax=100)
        ax.set_xticks(np.arange(len(x_labels)))
        ax.set_yticks(np.arange(len(y_labels)))
        ax.set_xticklabels(x_labels)
        ax.set_yticklabels(y_labels)
        ax.tick_params(axis='both', which='both', length=0)  # Remove tick marks
    
        # Add white lines by adjusting the linewidth for minor ticks
        ax.set_xticks(np.arange(len(x_labels)) - 0.5, minor=True)
        ax.set_yticks(np.arange(len(y_labels)) - 0.5, minor=True)
        ax.grid(which="minor", color="white", linestyle='-', linewidth=0.5)
        ax.tick_params(which="minor", size=0)
    
        # Remove external boundaries
        for spine in ax.spines.values():
            spine.set_visible(False)

        # Add text annotations
        for i in range(len(y_labels)):
            for j in range(len(x_labels)):
                if heatmap_data[i][j] == 100:
                    text = ax.text(j, i, '100', ha='center', va='center', color='black')
                elif heatmap_data[i][j] == 0:
                    text = ax.text(j, i, '0', ha='center', va='center', color='black')
                else:
                    text = ax.text(j, i, f'{heatmap_data[i][j]:.1f}', ha='center', va='center', color='black')

        ax.set_title(title, fontsize=12)
        ax.set_xlabel('Percentage of Uncertain Data', fontsize=12)
        ax.set_ylabel('Uncertain Radius (%)', fontsize=12)
    
    
    print("")
    print(f"Generating important indices based on {args.metric}:")
    print("")

    X_train, X_test, y_train, y_test = X_train.reset_index(drop=True) , X_test.reset_index(drop=True) , y_train.reset_index(drop=True) , y_test.reset_index(drop=True)
    boundary_indices = leave_one_out(X_train, y_train, X_test, y_test, modeltype, metric, maximize)
  

    print("")
    print(f"Running Leave One Out (using {args.metric}) on ZORRO.")
    print("")
    
    robustness_dicts = []
    for seed in range(5):
        # mpg +- 2 is robust
        label_range = (y_train.max()-y_train.min())
        uncertain_radiuses = [ratio*label_range for ratio in ratios]
        uncertain_pcts = list(np.arange(1, max_uncertain_pct + 1)/100)
        robustness_dict = dict()
        robustness_dict['uncertain_radius'] = uncertain_radiuses
        robustness_dict['uncertain_radius_ratios'] = ratios
        for uncertain_pct in tqdm(uncertain_pcts, desc='Progess'):
            robustness_dict[uncertain_pct] = list()
            uncertain_num = int(uncertain_pct*len(y_train))
            for uncertain_radius in tqdm(uncertain_radiuses, desc=f'Varying Uncertain Radius'):

                robustness_ratio = compute_robustness_ratio_sensitive_label_error(X_train, y_train, X_test, y_test, 
                                                                    uncertain_num=uncertain_num,
                                                                    boundary_indices=boundary_indices,
                                                                    uncertain_radius=uncertain_radius, 
                                                                    robustness_radius=robustness_radius, 
                                                                    interval=False, seed=seed)
                robustness_dict[uncertain_pct].append(robustness_ratio)
        robustness_dicts.append(robustness_dict)

    print("")
    print(f"Running Naive Approach on ZORRO.")
    print("")

    robustness_dicts_naive = []
    for seed in tqdm(range(5), desc=f'Progress'):
        label_range = (y_train.max()-y_train.min())
        uncertain_radiuses = [ratio*label_range for ratio in ratios]
        uncertain_pcts = list(np.arange(1, max_uncertain_pct + 1)/100)
        robustness_dict = dict()
        robustness_dict['uncertain_radius'] = uncertain_radiuses
        robustness_dict['uncertain_radius_ratios'] = ratios
        for uncertain_pct in tqdm(uncertain_pcts, desc=f'Rep {seed+1}', leave=False):
            robustness_dict[uncertain_pct] = list()
            uncertain_num = int(uncertain_pct*len(y_train))
            for uncertain_radius in tqdm(uncertain_radiuses, desc=f'Varying Uncertain Radius', leave=False):
            
                robustness_ratio = compute_robustness_ratio_label_error(X_train, y_train, X_test, y_test, 
                                                                    uncertain_num=uncertain_num, 
                                                                    uncertain_radius=uncertain_radius, 
                                                                    robustness_radius=robustness_radius, 
                                                                    interval=False, seed=seed)
                robustness_dict[uncertain_pct].append(robustness_ratio)
        robustness_dicts_naive.append(robustness_dict)

    print()
    print(f"Running Leave One Out (using {args.metric}) on Meyer.")
    print()
    
    robustness_dicts_interval = []
    for seed in range(5):
        # mpg +- 2 is robust
        label_range = (y_train.max()-y_train.min())
        uncertain_radiuses = [ratio*label_range for ratio in ratios]
        uncertain_pcts = list(np.arange(1, max_uncertain_pct + 1)/100)
        robustness_dict_interval = dict()
        robustness_dict_interval['uncertain_radius'] = uncertain_radiuses
        robustness_dict_interval['uncertain_radius_ratios'] = ratios
        for uncertain_pct in tqdm(uncertain_pcts, desc='Progess'):
            robustness_dict_interval[uncertain_pct] = list()
            uncertain_num = int(uncertain_pct*len(y_train))
            for uncertain_radius in tqdm(uncertain_radiuses, desc=f'Varying Uncertain Radius'):
                robustness_ratio = compute_robustness_ratio_sensitive_label_error(X_train, y_train, X_test, y_test, 
                                                                    uncertain_num=uncertain_num,
                                                                    boundary_indices=boundary_indices,
                                                                    uncertain_radius=uncertain_radius, 
                                                                    robustness_radius=robustness_radius, 
                                                                    interval=True, seed=seed)
                robustness_dict_interval[uncertain_pct].append(robustness_ratio)
        robustness_dicts_interval.append(robustness_dict_interval)

    
    print("")
    print(f"Running Naive Approach on Meyer.")
    print("")

    robustness_dicts_interval_naive = []
    for seed in tqdm(range(5), desc=f'Progress'):
        # mpg +- 2 is robust
        label_range = (y_train.max()-y_train.min())
        uncertain_radiuses = [ratio*label_range for ratio in ratios]
        uncertain_pcts = list(np.arange(1, max_uncertain_pct + 1)/100)
        robustness_dict_interval = dict()
        robustness_dict_interval['uncertain_radius'] = uncertain_radiuses
        robustness_dict_interval['uncertain_radius_ratios'] = ratios
        for uncertain_pct in tqdm(uncertain_pcts, desc=f'Rep {seed+1}', leave=False):
            robustness_dict_interval[uncertain_pct] = list()
            uncertain_num = int(uncertain_pct*len(y_train))
            for uncertain_radius in tqdm(uncertain_radiuses, desc=f'Varying Uncertain Radius', leave=False):
                robustness_ratio = compute_robustness_ratio_label_error(X_train, y_train, X_test, y_test, 
                                                                    uncertain_num=uncertain_num, 
                                                                    uncertain_radius=uncertain_radius, 
                                                                    robustness_radius=robustness_radius, 
                                                                    interval=True, seed=seed)
                robustness_dict_interval[uncertain_pct].append(robustness_ratio)
        robustness_dicts_interval_naive.append(robustness_dict_interval)

    print("Heatmaps:")

    # Create the heatmap plot with a 2x2 grid
    fig, axes = plt.subplots(2, 2, figsize=(15, 8), dpi=200)

    # Define colormap
    cmap = plt.get_cmap("autumn_r")

    print("Formatting data for the heatmaps")
    
    df1 = sum([pd.DataFrame(robustness_dicts_interval_naive[i]).iloc[:, 2:] for i in range(5)])/5
    df2 = sum([pd.DataFrame(robustness_dicts_interval[i]).iloc[:, 2:] for i in range(5)])/5
    df3 = sum([pd.DataFrame(robustness_dicts_naive[i]).iloc[:, 2:] for i in range(5)])/5  
    df4 = sum([pd.DataFrame(robustness_dicts[i]).iloc[:, 2:] for i in range(5)])/5  

    print("Converting fractions to percentages")
    heatmap_data1 = df1.multiply(100).values
    heatmap_data2 = df2.multiply(100).values
    heatmap_data3 = df3.multiply(100).values
    heatmap_data4 = df4.multiply(100).values

    # Labels
    x_labels = df1.columns.tolist()
    y_labels = ratios

    # Plot each heatmap
    plot_heatmap(axes[0, 0], heatmap_data1, x_labels, y_labels, 'Meyer et al. (Naive Approach)')
    plot_heatmap(axes[0, 1], heatmap_data2, x_labels, y_labels, 'Meyer et al. (Subset Approach)')
    plot_heatmap(axes[1, 0], heatmap_data3, x_labels, y_labels, 'ZORRO (Naive Approach)')
    plot_heatmap(axes[1, 1], heatmap_data4, x_labels, y_labels, 'ZORRO (Subset Approach)')

    # Adjust layout and add colorbar
    plt.subplots_adjust(wspace=0.2, hspace=0.4, bottom=0.1, left=0.1, right=0.9)
    cb = fig.colorbar(axes[0, 1].images[0], ax=axes, orientation='vertical', pad=0.02)
    cb.set_label('Robustness Ratio (%)', fontsize=12)
    
    plt.savefig(f"{output_dir}/{args.dataset}-LeaveOneOut(using {args.metric})-label-heatmap.pdf", bbox_inches='tight')
  
    print("")
    print("Leave One Out finished!")
    print("")

# Main function to parse arguments
def main():
    # Argument parsing
    print("")
    print("Grabbing Arguments...")
    print("")
    parser = argparse.ArgumentParser(description="Run robustness tests")
    #parser.add_argument('--test', choices=['baseline', 'leave_one_out'], help="Specify which test to run: (baseline, leave_one_out)")
    parser.add_argument("--dataset", choices=["mpg", "ins"], default="mpg", help="Filename of the dataset in the datasets' folder")
    parser.add_argument('--metric', choices=["mae", "mse"], default="mae", type=str, help="Metric to utilize")
    args = parser.parse_args()

    # Set parameters
    # set parameters
    output_dir = params["output_dir"]
    os.makedirs(output_dir, exist_ok=True)
    metric = args.metric

  
    if args.dataset == "mpg":
        ratios = [0.05, 0.10, 0.15, 0.2, 0.25]
        X_train, X_test, y_train, y_test = load_mpg_cleaned(random_seed=params["random_seed"])
        if metric == "mae":
            run_complex_test(X_train, y_train, X_test, y_test, output_dir, mae, LinearRegression, args, ratios, 2, maximize=False)
        elif metric == "mse":
            run_complex_test(X_train, y_train, X_test, y_test, output_dir, mse, LinearRegression, args, ratios, 2, maximize=False)
        else:
            print("Not a valid metric!")
    elif args.dataset == "ins":
        ratios = [0.02, 0.04, 0.06, 0.08]
        X_train, X_test, y_train, y_test = load_ins_cleaned(random_seed=params["random_seed"])
        if metric == "mae":
            run_complex_test(X_train, y_train, X_test, y_test, output_dir, mae, LinearRegression, args, ratios, 500, maximize=False)
        elif metric == "mse":
            run_complex_test(X_train, y_train, X_test, y_test, output_dir, mse, LinearRegression, args, ratios, 500, maximize=False)
        else:
            print("Not a valid metric!")
    else:
        print("")
        print("Dataset is not provided, please provided dataset.")


if __name__ == "__main__":
  main()


