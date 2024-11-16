# Import necessary libraries
import numpy as np  # For numerical operations and array handling
import matplotlib.pyplot as plt  # For data visualization
import pandas as pd  # For data manipulation and creating DataFrame
import seaborn as sns  # For advanced data visualization (e.g., heatmaps)

# Step 1: Define the hypothetical data
# List of model names representing various configurations
model_name = ['Gemma-2B', 'Gemma-7B', 'Llama-3-8B', 'Qwen-1.8B', 'Qwen-7B',
              'Qwen-14B', 'Qwen-1.5-0.5B', 'Qwen-1.5-1.8B', 'Qwen-1.5-4B', 'Qwen-1.5-7B']

# List of quantization levels representing parameter choices for the models
Quantization_level = [4, 8, None]  # None represents no quantization applied

# Hypothetical accuracy matrix (rows correspond to models, columns correspond to quantization levels)
accuracy = np.array([
    [0.5714, 0.5714, 0.5714],  # Accuracies for 'Gemma-2B'
    [0.6429, 0.2857, 0.6429],  # Accuracies for 'Gemma-7B'
    [0.8536, 0.6429, 0.5714],  # Accuracies for 'Llama-3-8B'
    [0.6429, 0.7143, 0.7143],  # Accuracies for 'Qwen-1.8B'
    [0.5000, 0.5714, 0.5714],  # Accuracies for 'Qwen-7B'
    [0.5714, 0.5714, 0.5714],  # Accuracies for 'Qwen-14B'
    [0.5714, 0.6429, 0.7857],  # Accuracies for 'Qwen-1.5-0.5B'
    [0.5714, 0.7143, 0.6429],  # Accuracies for 'Qwen-1.5-1.8B'
    [0.4286, 0.5714, 0.5714],  # Accuracies for 'Qwen-1.5-4B'
    [0.5714, 0.5714, 0.5714]   # Accuracies for 'Qwen-1.5-7B'
])

# Step 2: Convert the accuracy data into a pandas DataFrame
# The DataFrame structure makes it easier to handle and visualize the data
data_df = pd.DataFrame(accuracy, index=model_name, columns=Quantization_level)

# Step 3: Plot the heatmap using seaborn
# Create a heatmap to visualize the relationship between model accuracy, model name, and quantization level
plt.figure(figsize=(15, 8))  # Set the size of the figure
sns.heatmap(data_df, annot=True, cmap='coolwarm', fmt=".4f")  # Plot the heatmap with annotations and a color map
plt.title('Model Accuracy by Model Name and Quantization level')  # Add a title to the heatmap
plt.xlabel('Quantization level')  # Label for the x-axis
plt.ylabel('Model Name')  # Label for the y-axis
plt.show()  # Display the heatmap
