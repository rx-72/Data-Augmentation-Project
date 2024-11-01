import os
import json
import pandas as pd
import argparse
import matplotlib.pyplot as plt
from etl import *

with open("data-params.json", "r") as f:
  params = json.load(f)


def run_baseline_test(X_train, y_train, output_dir):
  Print("Running baseline on ZORRO.")
  robustness_dicts = []
  for seed in range(5):
    # mpg +- 2 is robust
    robustness_radius = 2
    label_range = (y_train.max()-y_train.min())
    ratios = [0.05, 0.10, 0.15, 0.2, 0.25]
    uncertain_radiuses = [ratio*label_range for ratio in ratios]
    uncertain_pcts = list(np.arange(1, 11)/100)
    robustness_dict = dict()
    robustness_dict['uncertain_radius'] = uncertain_radiuses
    robustness_dict['uncertain_radius_ratios'] = ratios
    for uncertain_pct in tqdm(uncertain_pcts, desc='Progess'):
        robustness_dict[uncertain_pct] = list()
        uncertain_num = int(uncertain_pct*len(y_train))
        for uncertain_radius in tqdm(uncertain_radiuses, desc=f'Varying Uncertain Radius'):
            #print(uncertain_radius)
            robustness_ratio = compute_robustness_ratio_label_error(X_train, y_train, X_test, y_test, 
                                                                    uncertain_num=uncertain_num, 
                                                                    uncertain_radius=uncertain_radius, 
                                                                    robustness_radius=robustness_radius, 
                                                                    interval=False, seed=seed)
            robustness_dict[uncertain_pct].append(robustness_ratio)
    robustness_dicts.append(robustness_dict)

  robustness_zonotope_mean = sum([pd.DataFrame(robustness_dicts[i]).iloc[:, 2:] for i in range(5)])/5
  robustness_zonotope_std = (sum([(pd.DataFrame(robustness_dicts[i]).iloc[:, 2:]-robustness_zonotope_mean)**2 for i in range(5)])/5).apply(np.sqrt)

  Print("Running baseline on Meyer.")
  # Running results with parameter adjustments on Meyer
  robustness_dicts = []
  for seed in range(5):
    # mpg +- 2 is robust
    robustness_radius = 2
    label_range = (y_train.max()-y_train.min())
    ratios = [0.05, 0.10, 0.15, 0.2, 0.25]
    uncertain_radiuses = [ratio*label_range for ratio in ratios]
    uncertain_pcts = list(np.arange(1, 11)/100)
    robustness_dict = dict()
    robustness_dict['uncertain_radius'] = uncertain_radiuses
    robustness_dict['uncertain_radius_ratios'] = ratios
    for uncertain_pct in tqdm(uncertain_pcts, desc='Progess'):
        robustness_dict[uncertain_pct] = list()
        uncertain_num = int(uncertain_pct*len(y_train))
        for uncertain_radius in tqdm(uncertain_radiuses, desc=f'Varying Uncertain Radius'):
            #print(uncertain_radius)
            robustness_ratio = compute_robustness_ratio_label_error(X_train, y_train, X_test, y_test, 
                                                                    uncertain_num=uncertain_num, 
                                                                    uncertain_radius=uncertain_radius, 
                                                                    robustness_radius=robustness_radius, 
                                                                    interval=True, seed=seed)
            robustness_dict[uncertain_pct].append(robustness_ratio)
    robustness_dicts.append(robustness_dict)

  robustness_interval_mean = sum([pd.DataFrame(robustness_dicts[i]).iloc[:, 2:] for i in range(5)])/5
  robustness_interval_std = (sum([(pd.DataFrame(robustness_dicts[i]).iloc[:, 2:]-robustness_interval_mean)**2 for i in range(5)])/5).apply(np.sqrt)

  # Heatmaps
  df = robustness_interval_mean

  # Isolate the portion of the DataFrame for heatmap (exclude the first two columns)
  heatmap_data = df.multiply(100).values  # Convert fractions to percentages

  # Labels for x-axis and y-axis
  x_labels = df.columns.tolist()
  y_labels = [0.05, 0.10, 0.15, 0.2, 0.25]
  
  # Create the heatmap plot
  # fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 4), sharey=True, dpi=200)
  fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 3), dpi=200)
  # cmap = plt.get_cmap("coolwarm")
  cmap = plt.get_cmap("autumn_r")
  heatmap = ax1.imshow(heatmap_data, cmap=cmap, interpolation='nearest', aspect='auto', 
                       alpha=0.8, vmin=0, vmax=100)
  
  # Add color bar
  # cbar = plt.colorbar(heatmap, ax=ax1)
  # cbar.set_label('% Percentage')
  
  # Add white lines by adjusting the linewidth for minor ticks to create separation
  ax1.set_xticks(np.arange(len(x_labels)) - 0.5, minor=True)
  ax1.set_yticks(np.arange(len(y_labels)) - 0.5, minor=True)
  ax1.grid(which="minor", color="white", linestyle='-', linewidth=0.5)
  ax1.tick_params(which="minor", size=0)
  
  # Set major ticks for labels without ticks
  ax1.set_xticks(np.arange(len(x_labels)))
  ax1.set_yticks(np.arange(len(y_labels)))
  ax1.set_xticklabels(x_labels)
  ax1.set_yticklabels(y_labels)
  ax1.tick_params(axis='both', which='both', length=0)  # Remove tick marks
  
  # Remove external boundaries
  ax1.spines['top'].set_visible(False)
  ax1.spines['right'].set_visible(False)
  ax1.spines['left'].set_visible(False)
  ax1.spines['bottom'].set_visible(False)
  
  # Set axis labels
  ax1.set_xlabel('Percentage of Uncertain Data', fontsize=12)
  ax1.set_ylabel('Uncertain Radius (%)', fontsize=12)
  
  # Add text annotations
  for i in range(len(y_labels)):
      for j in range(len(x_labels)):
          if heatmap_data[i][j]==100:
              text = ax1.text(j, i, f'{heatmap_data[i][j]:.0f}', ha='center', va='center', color='black')
          elif heatmap_data[i][j]==0:
              text = ax1.text(j, i, '0', ha='center', va='center', color='black')
          else:
              text = ax1.text(j, i, f'{heatmap_data[i][j]:.1f}', ha='center', va='center', color='black')
  ax1.set_title('Meyer et al.')
  
  df = robustness_zonotope_mean
  
  # Isolate the portion of the DataFrame for heatmap (exclude the first two columns)
  heatmap_data = df.multiply(100).values  # Convert fractions to percentages
  heatmap2 = ax2.imshow(heatmap_data, cmap=cmap, interpolation='nearest', 
                        aspect='auto', alpha=0.8, vmin=0, vmax=100)
  
  # Add color bar
  # cbar.set_label('% Percentage')
  
  # Add white lines by adjusting the linewidth for minor ticks to create separation
  ax2.set_xticks(np.arange(len(x_labels)) - 0.5, minor=True)
  ax2.set_yticks(np.arange(len(y_labels)) - 0.5, minor=True)
  ax2.grid(which="minor", color="white", linestyle='-', linewidth=0.5)
  ax2.tick_params(which="minor", size=0)
  
  # Set major ticks for labels without ticks
  ax2.set_xticks(np.arange(len(x_labels)))
  ax2.set_yticks(np.arange(len(y_labels)))
  ax2.set_xticklabels(x_labels)
  ax2.set_yticklabels(y_labels)
  ax2.tick_params(axis='both', which='both', length=0)  # Remove tick marks
  
  # Remove external boundaries
  ax2.spines['top'].set_visible(False)
  ax2.spines['right'].set_visible(False)
  ax2.spines['left'].set_visible(False)
  ax2.spines['bottom'].set_visible(False)
  
  # Set axis labels
  ax2.set_xlabel('Percentage of Uncertain Data', fontsize=12)
  ax2.set_ylabel('Uncertain Radius (%)', fontsize=12)
  
  # Add text annotations
  for i in range(len(y_labels)):
      for j in range(len(x_labels)):
          if heatmap_data[i][j]==100:
              text = ax2.text(j, i, '100', ha='center', va='center', color='black')
          elif heatmap_data[i][j]==0:
              text = ax2.text(j, i, '0', ha='center', va='center', color='black')
          else:
              text = ax2.text(j, i, f'{np.around(heatmap_data[i][j], 1)}', ha='center', 
                              va='center', color='black')
  ax2.set_title('ZORRO')
  
  # fig.suptitle('Robustness Ratio (%)', fontsize=14)
  plt.subplots_adjust(wspace=0.2, bottom=0.2, left=0.1, right=0.9)
  cb = fig.colorbar(heatmap2, ax=(ax1, ax2), orientation='vertical', pad=0.02)
  cb.set_label('Robustness Ratio (%)', fontsize=12)
  plt.savefig(f"{output_dir}/breast-cancer-baseline-label-heatmap.pdf", bbox_inches='tight')
  print("Baseline finished")

# Main function to parse arguments
def main():
  # Argument parsing
  parser = argparse.ArgumentParser(description="Run robustness tests")
  parser.add_argument('--test', type=str, choices=['baseline', 'residuals', 'gopher'], default='baseline',
                      help="Specify which test to run: (baseline, residuals, gopher)")
  args = parser.parser_args()

  # Set parameters
  # set parameters
  output_dir = params["output_dir"]
  os.makedirs(output_dir, exist_ok=True)

  # Load data
  X_train, X_test, y_train, y_test = load_data(random_seed=params["random_seed"])

  # run chosen test
  if args.test == 'baseline':
    run_baseline_test(X_train, y_train, outputdir)
  else:
    print("Not a test baseline")

if __name__ == "__main__":
  main()


