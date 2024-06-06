# import libaries for data science and machine learning
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor, AdaBoostRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor


# Load the TPMS dataset into dataframe for dispersion curves
combined_TPMS_df = pd.read_csv('20240604 TPMS_data_github.csv') # the TPMS data added data (last update: 2024 06 04 )

# Reading 5 first rows of data and showing all features
combined_TPMS_df.head(5)


# ## Acoustic Feature Extraction
#Drop Elastic data (extracting acoustic data only)
combined_TPMS_df= combined_TPMS_df[combined_TPMS_df['Band Type'] == 'Acoustic']

# ## Data Analysis: Ploting VF variation to Observe Acoustic Bandgaps (ABG)

# Create subplots in a 5x4 grid with additional setup
fig, axs = plt.subplots(3, 4, figsize=(45, 28))

# Create a list of labels for subplot annotations
subplot_labels = ['a)', 'b)', 'c)', 'd)', 'e)', 'f)', 'g)', 'h)', 'i)', 'j)', 'k)', 'l)' ]
ratios = [24, 27, 30, 33, 36, 38, 41, 44, 47, 50, 53, 59 ]  # Replace with your desired ratios

for i, ratio in enumerate(ratios):
    row = i // 4
    col = i % 4
    ax = axs[row, col]

    subset_df = combined_TPMS_df[combined_TPMS_df['VF(%)'] == ratio]
    #subset_df = subset_df[subset_df['Band Type'] == 'Acoustic']  # selecting Acoustic Band Strcture

    # Plot BNF columns against "k"
    for col in subset_df.columns:
        if col.startswith("BNF="):
            ax.plot(subset_df['k'], subset_df[col], label=col,linewidth=2.5)

    ax.set_title(f"ABG, VF = {ratio}%", fontsize=35,fontweight='bold')
    ax.set_xticks([0, 1, 2, 3, 4, 5])  # Define the positions of the ticks
    ax.set_xticklabels(['R', 'M', 'Γ', 'X', 'M'], fontsize=40,fontweight='bold')  # Set custom labels
    ax.set_ylabel("Frequency (kHz)", fontsize=35,fontweight='bold')
    ax.tick_params(axis='y', labelsize=35, width=2, length=6)  # Make y-axis ticks bigger and bolder
    ax.legend(loc='upper right', fontsize=7.8, )  # Move the legend to the upper right and set font size

    # Add additional setup
    ax.tick_params(axis='y', labelsize=35, width=2, length=6)  # Make y-axis ticks bigger and bolder
    ax.set_xlim([0, 4])  # Set x limits
    ax.set_ylim([0, 25])  # Set y limits

    # Annotate the figure with subplot labels
    subplot_label = subplot_labels[i]
    ax.annotate(subplot_label, xy=(0.05, 0.85), xycoords='axes fraction', fontsize=48, fontweight='bold')

# Add the subplots_adjust setup
plt.subplots_adjust(hspace=0.3, wspace=0.2)

plt.tight_layout()
plt.show()
