# Cyber-Physical Systems Intrusion Detection System with Explainable AI

This project implements an Intrusion Detection System (IDS) for Cyber-Physical Systems with comprehensive Explainable AI (XAI) capabilities to make the detection process transparent and interpretable.

## Project Overview

The CPS Intrusion Detection System analyzes network traffic data to detect various types of intrusions and provides detailed explanations of why specific traffic was flagged as an attack. The system combines machine learning with three different explainable AI techniques to help users understand the decision-making process.

## Key Features

- **Intrusion Detection**: Accurately identifies network attacks using machine learning models
- **Multi-method Explainability**: Provides three different types of explanations for each detected attack
- **Interactive Visualization**: Offers interactive plots and visualizations through a Streamlit interface
- **Counterfactual Analysis**: Suggests specific changes that would convert attack traffic to normal traffic
- **Exportable Reports**: Allows downloading explanation reports for further analysis

## Project Structure

The project is organized into the following Python modules:

- `streamlit_app.py`: Interactive Streamlit web application providing the main user interface
- `cps_xai_fixed.py`: Core implementation of XAI methods and model training
- `common.py`: Common imports, data loading functions, and utility functions
- `preprocessing.py`: Functions for data preprocessing and feature engineering
- `visualization.py`: Functions for data visualization and results plotting
- `model_training.py`: Functions for training and evaluating machine learning models
- `lime_explainer.py`: Implementation of LIME (Local Interpretable Model-agnostic Explanations)
- `shap_explainer.py`: Implementation of SHAP (SHapley Additive exPlanations)
- `main.py`: Script for command-line execution of the pipeline

## Setup and Installation

1. Install the required dependencies:

```bash
pip install -r requirements.txt
```

2. Make sure you have the KDDTrain1.csv and KDDTest1.csv datasets in the same directory
3. Run the model training script to generate necessary model files:

```bash
python cps_xai_fixed.py
```

## Usage

### Interactive Web Application

For an interactive exploration with full explainability features, run the Streamlit app:

```bash
streamlit run streamlit_app.py
```

The Streamlit app provides:

- Selection of instances by index or attack type
- Detection of network traffic attacks
- Detailed explanations through three XAI methods:
  - **LIME**: Shows which features most influence the prediction with local approximation
  - **SHAP**: Provides feature importance based on game theory with waterfall plots
  - **Counterfactual**: Suggests specific changes to convert attack traffic to normal
- Ability to download explanation reports for documentation
- Visualizations of feature importance and model predictions

### Command Line Analysis

For a basic analysis pipeline without the interactive interface:

```bash
python main.py
```

## Explainable AI Methods

This project implements three complementary explainable AI techniques:

1. **LIME (Local Interpretable Model-agnostic Explanations)**:
   - Explains individual predictions by learning a simple model around the prediction
   - Provides feature importances for specific predictions
   - Helps identify which features are most responsible for classifying an instance as an attack

2. **SHAP (SHapley Additive exPlanations)**:
   - Based on game theory concepts
   - Provides consistent and locally accurate feature attributions
   - Shows waterfall plots demonstrating how each feature pushes the prediction from baseline

3. **Counterfactual Explanations**:
   - Provides specific changes that would convert an attack instance to normal traffic
   - Hierarchical approach showing multiple levels of changes with different impacts
   - Offers actionable insights on how to prevent similar attacks

## Dataset

This project uses the KDD Cup 1999 dataset, which contains network traffic data with various types of intrusions. The dataset includes different attack categories:

- **DoS**: Denial of Service attacks
- **Probe**: Surveillance and port scanning
- **Privilege**: Unauthorized access to local superuser privileges
- **Access**: Unauthorized access from a remote machine

## Authors

- [Shailesh](https://github.com/Shailesh997)
- [Krishna Sampath](https://github.com/krishna-sampath)
- [Karthik Jonnalagadda](https://github.com/jkarthik-2004)
- [Yogesh Kanaparthi](https://github.com/yogeshkanaparthi)
