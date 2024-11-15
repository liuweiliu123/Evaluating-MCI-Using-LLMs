import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# Hypothetical data: impact of different combinations of learning rates and batch sizes on model accuracy
model_name = ['Gemma-2B', 'Gemma-7B','Llama-3-8B','Qwen-1.8B', 'Qwen-7B','Qwen-14B','Qwen-1.5-0.5B', 'Qwen-1.5-1.8B','Qwen-1.5-4B','Qwen-1.5-7B']
Quantization_level = [4, 8, None]
accuracy = np.array([
    [0.5714, 0.5714, 0.5714],
    [0.6429, 0.2857, 0.6429],
    [0.8536, 0.6429, 0.5714],
    [0.6429, 0.7143, 0.7143],
    [0.5000, 0.5714, 0.5714],
    [0.5714, 0.5714, 0.5714],
    [0.5714, 0.6429, 0.7857],
    [0.5714, 0.7143, 0.6429],
    [0.4286, 0.5714, 0.5714],
    [0.5714, 0.5714, 0.5714]
])
# Convert the data matrix into a DataFrame for easier plotting with seaborn
data_df = pd.DataFrame(accuracy, index=model_name, columns=Quantization_level)

# Plotting the heatmap
plt.figure(figsize=(15, 8))
sns.heatmap(data_df, annot=True, cmap='coolwarm', fmt=".4f")
plt.title('Model Accuracy by Model Name and Quantization level')
plt.xlabel('Quantization level')
plt.ylabel('Model Name')
plt.show()
