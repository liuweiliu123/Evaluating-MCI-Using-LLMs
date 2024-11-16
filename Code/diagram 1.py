# Import necessary libraries
import matplotlib.pyplot as plt  # For creating visualizations
import pandas as pd  # For data manipulation and processing
import matplotlib.patches as mpatches  # For custom legend creation

# Step 1: Read the dataset
# Reads specific columns (0, 1, 2, and 3) from the CSV file at the given path
data = pd.read_csv('../../Desktop/baha1/Code and dataset/1234567.csv', usecols=[0, 1, 2, 3])

# Step 2: Assign temporary column names
# Assigns names for easier access: 'Color', 'Size', 'X', and 'Y'
data.columns = ['Color', 'Size', 'X', 'Y']

# Step 3: Convert 'X' column to numeric type
# Converts values in the 'X' column to numeric format, setting invalid values to NaN
data['X'] = pd.to_numeric(data['X'], errors='coerce')

# Step 4: Ensure the 'Y' column is numeric
# Converts values in the 'Y' column to numeric format, setting invalid values to NaN
data['Y'] = pd.to_numeric(data['Y'], errors='coerce')

# Step 5: Convert the 'Size' column to numeric type
# Ensures the 'Size' column values are numeric for setting scatter plot point sizes
data['Size'] = pd.to_numeric(data['Size'], errors='coerce')

# Step 6: Map color labels to specific color codes
# Maps categorical values in the 'Color' column to actual colors for plotting
color_mapping = {'Gemma': 'red', 'Qwen': 'blue', 'Qwen1.5': 'green', 'Llama3': 'yellow'}
data['Color'] = data['Color'].map(color_mapping)

# Step 7: Filter data by specific values in the 'X' column
# Retains only rows where the 'X' column has values 4, 8, or 12
data = data[data['X'].isin([4, 8, 12])]

# Step 8: Remove rows with NaN values
# Drops rows with missing values in 'X', 'Y', 'Size', or 'Color' columns to avoid errors in plotting
data.dropna(subset=['X', 'Y', 'Size', 'Color'], inplace=True)

# Step 9: Create a scatter plot
# Groups data by 'Color' and plots scatter points for each group
for label, df in data.groupby('Color'):
    # Assign a model name based on the color
    if label == 'red':
        model_name = 'Gemma'
    elif label == 'blue':
        model_name = 'Qwen'
    elif label == 'green':
        model_name = 'Qwen1.5'
    else:
        model_name = 'Llama3'
    # Plot the scatter points for the current group
    plt.scatter(df['X'], df['Y'], s=df['Size'] * 150, c=label, alpha=0.5, label=model_name)

# Step 10: Set axis labels
# Adds labels to the X and Y axes
plt.xlabel('Quantization level')
plt.ylabel('Accuracy')

# Step 11: Customize X-axis ticks
# Sets custom labels for the X-axis ticks, replacing 12 with 'None'
plt.xticks([4, 8, 12], ['4', '8', 'None'])

# Step 12: Set X-axis limits
# Restricts the range of X-axis values to between 3 and 17
plt.xlim(3, 17)

# Step 13: Create a legend
# Manually creates a legend with custom color patches and labels for each model
legend_handles = [
    mpatches.Patch(color='red', label='Gemma'),
    mpatches.Patch(color='blue', label='Qwen'),
    mpatches.Patch(color='green', label='Qwen1.5'),
    mpatches.Patch(color='yellow', label='Llama3')
]
# Adds the legend to the plot with a title and adjusts its position
plt.legend(handles=legend_handles, title='Model', bbox_to_anchor=(1.05, 1), loc='upper left', markerscale=1)

# Step 14: Adjust layout
# Ensures that the plot layout adjusts to prevent the legend from being clipped
plt.tight_layout()

# Step 15: Display the plot
# Renders the scatter plot on the screen
plt.show()
