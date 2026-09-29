# ML Maestro — Implementation Guide

This document provides the technical instructions for installing, running,
and using ML Maestro V1 and V2.

It covers:

- [Environment setup & Dependency installation](#installation)
- [Running ML Maestro V2](#running-v2)
- [Using the V2 application](#using-ml-maestro)
- [Running the V1 application](#running-v1)
- [Input dataset requirements](#input-dataset)
- [Application outputs](#output)

The main project overview, publication information, architecture,
version history, and project description are available in the
[README](README.md).





# Installation

## Prerequisites

Before installing ML Maestro, make sure the following are available:

- Python
- Git
- pip
- A modern web browser
- PowerShell, Command Prompt, or another terminal

For V2, a Python 3.9–3.12 environment is recommended when creating a new environment because Machine Learning dependencies such as PyCaret can have version-specific compatibility requirements.

---

# Clone the Repository

Open PowerShell and run:

```powershell
git clone https://github.com/Ananyakavili/ML_Maestro.git
```

Move into the project directory:

```powershell
cd ML_Maestro
```

Verify the repository:

```powershell
git status
```

---

# Create a Virtual Environment

Creating a virtual environment is recommended so that ML Maestro dependencies remain isolated from other Python projects.

On Windows:

```powershell
python -m venv venv
```

Activate the environment:

```powershell
.\venv\Scripts\Activate.ps1
```

After activation, PowerShell should show something similar to:

```text
(venv) PS D:\ML_Maestro>
```

---

# PowerShell Execution Policy

If PowerShell prevents the virtual environment from being activated, you may see an execution-policy error.

Run:

```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

Then activate the environment again:

```powershell
.\venv\Scripts\Activate.ps1
```

---

# Install Dependencies

Upgrade pip:

```powershell
python -m pip install --upgrade pip
```

Install the dependencies required by V2:

```powershell
pip install -r requirements.txt
```

Depending on the Python version and package versions, PyCaret and its dependencies may require additional compatibility adjustments.

---

# Running V2

V2 is the current version of ML Maestro.

Make sure you are in the project root:

```powershell
cd D:\ML_Maestro
```

If the virtual environment is not already active:

```powershell
.\venv\Scripts\Activate.ps1
```

Start Streamlit:

```powershell
streamlit run ML_Maestro.py
```

Streamlit will start a local web server.

The application is normally available at:

```text
http://localhost:8501
```

Open the address in a web browser.

---

# Alternative V2 Start Command

If the `streamlit` command is not recognized, run:

```powershell
python -m streamlit run ML_Maestro.py
```

This ensures Streamlit is executed through the active Python environment.

---

# Running V2 on Another Port

If port `8501` is already being used, run:

```powershell
streamlit run ML_Maestro.py --server.port 8502
```

The application will then be available at:

```text
http://localhost:8502
```

---

# Using ML Maestro

Once the Streamlit application opens in the browser, follow the workflow below.

## Step 1 - Upload Dataset

Upload a CSV dataset through the Streamlit interface.

The dataset should contain:

- Meaningful column names
- Appropriate data types
- Rows representing observations
- Columns representing variables/features

---

## Step 2 - Explore the Dataset

Review the uploaded dataset.

Check:

- Number of rows
- Number of columns
- Data types
- Missing values
- General dataset structure

---

## Step 3 - Generate a Data Profile

Use the data profiling functionality to perform automated Exploratory Data Analysis.

Review:

- Statistics
- Distributions
- Missing values
- Correlations
- Variable information

---

## Step 4 - Visualize the Data

Use the visualization functionality to investigate patterns and relationships.

### Univariate

Select a single feature and create:

- Histogram
- Bar Chart

### Bivariate

Select two variables and create:

- Scatter Plot
- Line Plot
- Box Plot

### Multivariate

Use the correlation functionality to examine relationships between numerical variables.

---

## Step 5 - Select a Machine Learning Task

Choose one of:

```text
Classification
Regression
Clustering
```

Select the task according to the problem you are trying to solve.

---

## Step 6 - Select the Target

For supervised Machine Learning tasks:

### Classification

Select a categorical target column.

Example:

```text
Purchased
Fraud
Category
Status
```

### Regression

Select a numerical target column.

Example:

```text
Price
Sales
Revenue
Temperature
```

### Clustering

A target column is not required.

---

## Step 7 - Train and Compare Models

The application uses PyCaret to train and compare available models.

The exact models and evaluation metrics depend on the selected Machine Learning task and the dataset.

---

## Step 8 - Save the Model

After the Machine Learning workflow is completed, the model can be saved.

Example:

```text
best_model.pkl
```

---

## Step 9 - Download the Model

The trained model can be downloaded through the Streamlit interface.

The downloaded model can then be retained for future use.

---

# Running V1

V1 is preserved under:

```text
versions/v1/
```

The main V1 application is:

```text
versions/v1/app.py
```

To run V1, move into the V1 directory:

```powershell
cd versions\v1
```

Create a virtual environment if required:

```powershell
python -m venv venv
```

Activate it:

```powershell
.\venv\Scripts\Activate.ps1
```

Install the V1 dependencies:

```powershell
pip install -r requirements.txt
```

Run the original V1 application:

```powershell
streamlit run app.py
```

Alternatively:

```powershell
python -m streamlit run app.py
```

V1 is maintained primarily for historical reference, reproducibility, and comparison with V2.

---

# Input Dataset

ML Maestro is designed primarily to work with CSV datasets.

A dataset should generally contain:

- Rows representing observations
- Columns representing variables/features
- Meaningful column names
- Appropriate data types

## Supervised Machine Learning

For supervised Machine Learning:

### Classification

Classification requires a target column representing categories or classes.

### Regression

Regression requires a numerical target column.

## Unsupervised Machine Learning

### Clustering

Clustering does not require a target column.

The algorithm attempts to identify groups based on the available features.

---

# Example Dataset

A simple classification dataset could look like:

```text
Age,Income,Purchased
25,35000,No
31,52000,Yes
45,78000,Yes
22,29000,No
36,61000,Yes
```

In this example:

- `Age` is a feature
- `Income` is a feature
- `Purchased` is the target

The `Purchased` column can be used for classification.

---

# Output

Depending on the workflow, ML Maestro can produce:

- Dataset profiles
- Interactive visualizations
- Correlation analysis
- Model comparison results
- Trained Machine Learning models
- Downloadable `.pkl` model files

An example trained model file is:

```text
best_model.pkl
```
