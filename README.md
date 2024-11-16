
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

Code explanation

[Click here to open in Google Colab](https://colab.research.google.com/drive/1EJaqxjigGaF2SbLdCH-v2TYgKX904a7j#scrollTo=1bK9NrEaR60U)
