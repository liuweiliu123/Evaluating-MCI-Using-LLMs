
# Evaluating Mild Cognitive Impairment Using Large Language Models

## Overview

This project explores the application of **Large Language Models (LLMs)** for detecting **Mild Cognitive Impairment (MCI)** through speech analysis. The study involves fine-tuning advanced LLMs like Llama3, Gemma, and Qwen using **Supervised Fine-Tuning (SFT)** methods. The dataset consists of speech samples transcribed and preprocessed for MCI classification. This project aims to provide an efficient and scalable alternative to traditional diagnostic methods, enabling early detection of cognitive impairments.

---

## Features

- **LLM Fine-Tuning**: Implements advanced tuning techniques like Low-Rank Adaptation (LoRA) and Quantized LoRA (QLoRA) for resource-efficient model adaptation.
- **Speech-to-Text Preprocessing**: Converts audio recordings to text using Google Speech Recognition and pydub for improved transcription quality.
- **Multi-Metric Evaluation**: Evaluates models using metrics such as Accuracy, F1-Score, Precision, and Recall.
- **Quantization Impact Analysis**: Investigates the effects of model quantization on performance.

---

## Dataset

- **Source**: Speech recordings from the **TAUKA-DIAL Challenge** dataset, sourced via TalkBank (2002). https://sla.talkbank.org/TBB/dementia
- **Participants**: A total of 67 Chinese-speaking subjects aged 60–90 years with no history of neurological disorders.
- **Data Processing**:
  - Speech-to-text conversion using Google Speech Recognition API.
  - Dataset restructuring into `Instruction`, `Input`, and `Output` columns.
  - Balanced split of training (1:4) and testing (1:1 MCI to NC ratio) datasets.

---

## Model Fine-Tuning

### Base Models
- **Llama3**: Meta-LLaMA-3-8B-Instruct
- **Gemma**: Gemma-2B & 7B
- **Qwen**: Qwen-1.8B, 7B, and 14B
- **Qwen1.5**: Variants from 0.5B to 7B

### Fine-Tuning Techniques
- **Supervised Fine-Tuning (SFT)**:
  - Task-specific adaptation for MCI detection using binary classification (1 for MCI, 0 for NC).
  - Optimization with backpropagation to minimize loss.
- **Low-Rank Adaptation (LoRA)**:
  - Efficiently updates model parameters with minimal computational overhead.
- **Quantized LoRA (QLoRA)**:
  - Combines quantization and LoRA to handle memory constraints while maintaining performance.

---

## Code explanation

### diagram 1.py
- **Data Import and Preprocessing**:
  - Reads specific columns (`Color`, `Size`, `X`, `Y`) from a CSV file.
  - Converts `X`, `Y`, and `Size` to numeric format to ensure compatibility for plotting.
  - Maps the `Color` column's categorical values (e.g., 'Gemma', 'Qwen') to specific colors (e.g., red, blue) for visual distinction.

- **Data Filtering**:
  - Filters data to include only rows where `X` (quantization level) equals 4, 8, or 12.
  - Removes rows with missing values (`NaN`) in key columns.

- **Scatter Plot Creation**:
  - Groups data by color, representing different models.
  - Plots scatter points for each group, with sizes proportional to the `Size` column.
  - Adds X and Y axis labels and customizes X-axis tick labels (replacing `12` with `None`).

- **Legend Customization**:
  - Creates a custom legend with color patches representing each model (e.g., 'Gemma' in red, 'Qwen' in blue).
  - Positions the legend outside the plot for better readability.

- **Visualization Adjustments**:
  - Adjusts X-axis range and layout to ensure proper spacing and prevent overlapping elements.

- **Plot Display**:
  - Renders the finalized scatter plot, effectively illustrating the relationship between quantization levels and accuracy for various models.
#### Prerequisites
-**Python**: 3.8 or higher
- **Dependencies**:
  ```bash
  pip install -r requirements.txt
Ensure the necessary libraries are installed:
   ```bash
   pip install matplotlib pandas
### diagram 2.py
- **Data Definition**:
  - Creates a list of model names (`model_name`) representing different configurations.
  - Specifies quantization levels (`Quantization_level`), where `None` indicates no quantization applied.
  - Defines a hypothetical accuracy matrix (`accuracy`) showing the performance of each model under respective quantization levels.

- **Data Transformation**:
  - Converts the accuracy matrix into a pandas DataFrame (`data_df`) for easier handling and visualization.
  - Rows represent model names, and columns represent quantization levels.

- **Heatmap Visualization**:
  - Uses Seaborn to create a heatmap with:
    - Annotated cell values to display the accuracy.
    - A `coolwarm` color map to visually distinguish accuracy levels.
  - Adds axis labels and a title to provide context for the visualization.

- **Plot Display**:
  - Adjusts the plot size for better readability.
  - Displays the heatmap, allowing users to compare the performance of different models across quantization levels.
#### Prerequisites
-**Python**: 3.8 or higher
- **Dependencies**:
  ```bash
  pip install -r requirements.txt
Ensure the required libraries are installed:
   ```bash
   pip install numpy pandas matplotlib seaborn

[Click here to open in Google Colab](https://colab.research.google.com/drive/1EJaqxjigGaF2SbLdCH-v2TYgKX904a7j#scrollTo=1bK9NrEaR60U)
