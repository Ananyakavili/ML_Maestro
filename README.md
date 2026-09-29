# ML Maestro

### Zero-Code Machine Learning and Interactive Data Analysis Platform

ML Maestro is a **Streamlit-based, zero-code Machine Learning and interactive data analysis platform** designed to make common Machine Learning workflows accessible without requiring users to write Machine Learning code.

The application allows users to:

- Upload datasets
- Explore datasets
- Perform automated data profiling
- Create interactive visualizations
- Perform Machine Learning tasks
- Compare Machine Learning models
- Save trained models
- Download trained models

ML Maestro has evolved through **two major versions**:

- **V1 – Original ML Maestro:** The initial implementation focused primarily on classification and automated Machine Learning.
- **V2 – Current ML Maestro:** An expanded implementation supporting classification, regression, clustering, data profiling, and interactive data visualization.

The repository preserves the original V1 implementation under `versions/v1/`, while the current V2 implementation is maintained at the repository root.

---

# Table of Contents

- [Overview](#overview)
- [Project Evolution](#project-evolution)
- [V1 - Original ML Maestro](#v1---original-ml-maestro)
- [V2 - Current ML Maestro](#v2---current-ml-maestro)
- [V1 vs V2](#v1-vs-v2)
- [Key Features](#key-features)
- [How ML Maestro Works](#how-ml-maestro-works)
- [Application Workflow](#application-workflow)
- [Code Architecture](#code-architecture)
- [Machine Learning Tasks](#machine-learning-tasks)
- [Data Profiling](#data-profiling)
- [Interactive Data Visualization](#interactive-data-visualization)
- [Model Training and Comparison](#model-training-and-comparison)
- [Model Saving and Download](#model-saving-and-download)
- [Project Structure](#project-structure)
- [Technologies Used](#technologies-used)
- [Installation](#installation)
- [Running V2](#running-v2)
- [Using ML Maestro](#using-ml-maestro)
- [Running V1](#running-v1)
- [Input Dataset](#input-dataset)
- [Output](#output)
- [Version History](#version-history)
- [Limitations](#limitations)
- [Future Enhancements](#future-enhancements)
- [Contributors](#contributors)
- [License](#license)
- [Quick Start](#quick-start)

---

# Overview

Traditional Machine Learning workflows often require users to write Python code for:

1. Loading a dataset
2. Exploring the data
3. Performing data profiling
4. Visualizing relationships between variables
5. Selecting a Machine Learning algorithm
6. Preparing the data
7. Training the model
8. Comparing model performance
9. Saving the trained model
10. Downloading or deploying the model

ML Maestro provides a graphical web interface for these steps.

The application combines:

- **Streamlit** for the web interface
- **Pandas** for dataset handling and manipulation
- **YData Profiling** for automated Exploratory Data Analysis
- **Plotly** for interactive visualization
- **Matplotlib** for visualization
- **Seaborn** for statistical visualization and correlation analysis
- **PyCaret** for automated Machine Learning
- **Scikit-learn** through the Machine Learning workflow

The goal is to provide a simple workflow where users can focus on their dataset and Machine Learning task rather than implementing the underlying Machine Learning pipeline manually.

---

# Project Evolution

ML Maestro was developed through two major versions.

```text
                    ML MAESTRO
                         │
              ┌──────────┴──────────┐
              │                     │
              ▼                     ▼
             V1                    V2
       Original Version       Current Version
              │                     │
              ▼                     ▼
       Classification       Classification
       Data Profiling       Regression
       Model Comparison     Clustering
       Model Saving         Data Profiling
       Model Download       Visualization
                            Model Comparison
                            Model Saving
                            Model Download
```

## V1

V1 introduced the core ML Maestro concept as a Streamlit-based, zero-code Machine Learning application.

The main Machine Learning workflow was classification using PyCaret.

## V2

V2 expanded the original concept into a broader Machine Learning and interactive data analysis platform.

V2 introduced additional Machine Learning tasks and expanded visualization capabilities.

The repository preserves both versions so that the original implementation remains available while V2 serves as the current application.

---

# V1 - Original ML Maestro

V1 is the original implementation of ML Maestro.

It was developed as a Streamlit-based Machine Learning application that simplified the Machine Learning workflow for users who did not want to write ML code manually.

## V1 Main Capabilities

V1 provides:

- Dataset upload
- Dataset exploration
- Data profiling
- Classification Machine Learning
- Automated model comparison
- Best model selection
- Model saving
- Model download
- Sample datasets
- Streamlit interface

PyCaret is used to automate the classification Machine Learning workflow.

## V1 Workflow

```text
                 ┌──────────────────┐
                 │   Upload Dataset │
                 └────────┬─────────┘
                          │
                          ▼
                 ┌──────────────────┐
                 │ Explore Dataset  │
                 └────────┬─────────┘
                          │
                          ▼
                 ┌──────────────────┐
                 │  Data Profiling  │
                 └────────┬─────────┘
                          │
                          ▼
                 ┌──────────────────┐
                 │ Select Target    │
                 └────────┬─────────┘
                          │
                          ▼
                 ┌──────────────────┐
                 │ Classification   │
                 └────────┬─────────┘
                          │
                          ▼
                 ┌──────────────────┐
                 │ Model Comparison │
                 └────────┬─────────┘
                          │
                          ▼
                    Best Model
                          │
                          ▼
                    Save Model
                          │
                          ▼
                  Download Model
```

## V1 Application

The main V1 application is:

```text
versions/v1/app.py
```

The complete original V1 implementation is preserved under:

```text
versions/v1/
```

V1 is maintained for:

- Historical reference
- Reproducibility
- Comparison with V2
- Understanding the evolution of ML Maestro

V1 is not the primary application entry point for the current repository.

---

# V2 - Current ML Maestro

V2 is the **current version** of ML Maestro.

V2 builds upon the original V1 application and expands it into a broader zero-code Machine Learning and interactive data analysis platform.

The main V2 application is:

```text
ML_Maestro.py
```

## V2 Main Capabilities

V2 supports:

- CSV dataset upload
- Automated data profiling
- Exploratory Data Analysis
- Interactive data visualization
- Univariate visualization
- Bivariate visualization
- Multivariate visualization
- Correlation analysis
- Classification
- Regression
- Clustering
- Automated model comparison
- Model saving
- Trained model download

---

# V1 vs V2

The main evolution from V1 to V2 is the expansion from a primarily classification-focused Machine Learning application into a broader Machine Learning and interactive data analysis platform.

| Feature | V1 | V2 |
|---|---|---|
| Streamlit Interface | Yes | Yes |
| CSV Dataset Upload | Yes | Yes |
| Dataset Exploration | Yes | Yes |
| Data Profiling | Yes | Yes |
| Classification | Yes | Yes |
| Regression | No | Yes |
| Clustering | No | Yes |
| Model Comparison | Yes | Yes |
| Model Saving | Yes | Yes |
| Model Download | Yes | Yes |
| Univariate Visualization | Limited | Yes |
| Bivariate Visualization | Limited | Yes |
| Multivariate Visualization | No | Yes |
| Correlation Heatmap | No | Yes |
| Interactive Data Analysis | Limited | Expanded |
| Zero-Code ML Workflow | Yes | Yes |

## Evolution from V1 to V2

```text
                         ML Maestro V1
                              │
                              ▼
                       Dataset Upload
                              │
                              ▼
                       Data Exploration
                              │
                              ▼
                        Data Profiling
                              │
                              ▼
                        Classification
                              │
                              ▼
                       Model Comparison
                              │
                              ▼
                          Best Model
                              │
                              ▼
                       Model Download
                              │
                              │
                          Evolution
                              │
                              ▼
                         ML Maestro V2
                              │
                              ▼
                       Dataset Upload
                              │
                  ┌───────────┴───────────┐
                  │                       │
                  ▼                       ▼
           Data Profiling          Visualization
                                          │
                               ┌──────────┼──────────┐
                               │          │          │
                               ▼          ▼          ▼
                          Univariate  Bivariate  Multivariate
                               │          │          │
                               └──────────┼──────────┘
                                          │
                                          ▼
                                  Select ML Task
                                          │
                              ┌───────────┼───────────┐
                              │           │           │
                              ▼           ▼           ▼
                       Classification Regression Clustering
                              │           │           │
                              └───────────┼───────────┘
                                          │
                                          ▼
                                   Model Training
                                          │
                                          ▼
                                   Model Comparison
                                          │
                                          ▼
                                      Best Model
                                          │
                                          ▼
                                   Save / Download
```

---

# Key Features

## 1. Dataset Upload

Users can upload a CSV dataset directly through the Streamlit interface.

The uploaded dataset can then be used for:

- Data exploration
- Data profiling
- Visualization
- Machine Learning

Depending on the workflow, the dataset may be stored locally as:

```text
sourcedata.csv
```

---

## 2. Automated Data Profiling

ML Maestro provides automated Exploratory Data Analysis using YData Profiling.

The profiling functionality can provide information such as:

- Dataset statistics
- Column types
- Missing values
- Distributions
- Unique values
- Correlations
- Dataset structure
- Variable information

This allows users to understand their data before beginning model training.

---

## 3. Interactive Data Visualization

V2 provides three main visualization categories:

- Univariate
- Bivariate
- Multivariate

These visualizations allow users to investigate individual variables and relationships between multiple variables.

---

## 4. Classification

Classification is used when the target variable represents categories or classes.

Examples include:

- Yes / No
- Fraud / Not Fraud
- Positive / Negative
- Category A / Category B / Category C

---

## 5. Regression

Regression is used when the target variable is numerical.

Examples include:

- Price prediction
- Sales prediction
- Revenue prediction
- Temperature prediction

---

## 6. Clustering

Clustering is an unsupervised Machine Learning task.

Unlike classification and regression, clustering does not require a target variable.

Clustering can be used to group observations based on their available features.

---

## 7. Automated Model Comparison

ML Maestro uses PyCaret to simplify the process of training and comparing Machine Learning models.

Users do not need to manually implement every Machine Learning algorithm.

---

## 8. Model Saving

After training a Machine Learning model, the application allows the trained model to be saved as a pickle file.

Example:

```text
best_model.pkl
```

---

## 9. Model Download

The trained model can be downloaded through the Streamlit interface and retained for future use.

---

# How ML Maestro Works

The following flowchart represents the workflow of the **current V2 application**.

```text
                    ┌──────────────────┐
                    │   Upload Dataset │
                    └────────┬─────────┘
                             │
                             ▼
                   ┌────────────────────┐
                   │  Data Exploration  │
                   └─────────┬──────────┘
                             │
              ┌──────────────┼──────────────┐
              │              │              │
              ▼              ▼              ▼
       Data Profiling   Visualization   Statistics
              │              │              │
              └──────────────┼──────────────┘
                             │
                             ▼
                  ┌─────────────────────┐
                  │   Select ML Task    │
                  └──────────┬──────────┘
                             │
              ┌──────────────┼──────────────┐
              │              │              │
              ▼              ▼              ▼
       Classification    Regression     Clustering
              │              │              │
              └──────────────┼──────────────┘
                             │
                             ▼
                    Model Training
                             │
                             ▼
                   Model Comparison
                             │
                             ▼
                       Best Model
                             │
                             ▼
                    Save Model (.pkl)
                             │
                             ▼
                    Download Model
```

The V2 workflow allows users to move from dataset exploration and visualization to one of three Machine Learning tasks:

- **Classification** – for categorical target variables
- **Regression** – for numerical target variables
- **Clustering** – for grouping observations without a target variable

After the Machine Learning task is performed, models can be compared, the resulting model can be saved, and the trained model can be downloaded.

---

# Application Workflow

The application can be understood as five major stages.

## Stage 1 - Upload Dataset

The user uploads a CSV dataset through the Streamlit interface.

---

## Stage 2 - Explore the Dataset

The user examines the dataset and uses profiling functionality to understand:

- Rows
- Columns
- Data types
- Missing values
- Distributions
- Correlations
- General dataset statistics

---

## Stage 3 - Visualize the Dataset

The user can create visualizations to identify patterns and relationships in the data.

Available visualization categories include:

- Univariate
- Bivariate
- Multivariate

---

## Stage 4 - Select a Machine Learning Task

The user selects one of the available Machine Learning tasks:

```text
Classification
Regression
Clustering
```

The appropriate task depends on the problem and dataset.

---

## Stage 5 - Train and Download the Model

For the selected Machine Learning task, the application trains and compares models.

The resulting model can then be saved and downloaded.

---

# Code Architecture

The current V2 application is primarily implemented in:

```text
ML_Maestro.py
```

The application uses Streamlit to provide the user interface and integrates data analysis and Machine Learning functionality into the Streamlit workflow.

The high-level architecture is:

```text
                   ML Maestro V2
                         │
                         ▼
                  Dataset Input
                         │
                         ▼
                  Data Processing
                         │
              ┌──────────┼──────────┐
              │          │          │
              ▼          ▼          ▼
          Profiling  Visualization  Statistics
              │          │          │
              └──────────┼──────────┘
                         │
                         ▼
                  Machine Learning
                         │
             ┌───────────┼───────────┐
             │           │           │
             ▼           ▼           ▼
       Classification Regression Clustering
             │           │           │
             └───────────┼───────────┘
                         │
                         ▼
                  Model Comparison
                         │
                         ▼
                    Best Model
                         │
                         ▼
                 Model Save / Download
```

---

# Machine Learning Tasks

## Classification

Classification is used when the target variable represents categories or classes.

The general workflow is:

```text
Select Dataset
      │
      ▼
Select Target Column
      │
      ▼
Initialize Classification
      │
      ▼
Compare Models
      │
      ▼
Select Best Available Model
      │
      ▼
Save Model
```

Classification can be used for problems such as:

- Binary classification
- Multi-class classification
- Category prediction

---

## Regression

Regression is used when the target variable is numerical.

The general workflow is:

```text
Select Dataset
      │
      ▼
Select Numerical Target
      │
      ▼
Initialize Regression
      │
      ▼
Compare Models
      │
      ▼
Select Best Available Model
      │
      ▼
Save Model
```

Regression can be used for problems such as:

- Price prediction
- Sales prediction
- Revenue prediction
- Numerical value prediction

---

## Clustering

Clustering is an unsupervised Machine Learning technique.

Unlike classification and regression, clustering does not require a target variable.

The general workflow is:

```text
Select Dataset
      │
      ▼
Prepare Features
      │
      ▼
Initialize Clustering
      │
      ▼
Create Clusters
      │
      ▼
Analyze Groups
```

Clustering can be used to discover groups or patterns within a dataset.

---

# Data Profiling

ML Maestro uses YData Profiling to provide automated Exploratory Data Analysis.

The profiling process helps users identify:

- Number of rows
- Number of columns
- Data types
- Missing values
- Duplicate values
- Unique values
- Statistical distributions
- Correlations
- Variable information
- Potential data quality issues

Data profiling is useful before Machine Learning because it allows users to understand the structure and characteristics of the dataset before selecting a modeling approach.

---

# Interactive Data Visualization

V2 provides three major visualization categories.

## Univariate Visualization

Univariate visualization examines a single variable.

Available visualizations include:

- Histogram
- Bar Chart

Users can select a feature and customize visualization properties such as:

- Plot title
- Plot color

---

## Bivariate Visualization

Bivariate visualization examines relationships between two variables.

Available visualizations include:

- Scatter Plot
- Line Plot
- Box Plot

Users can select:

- X-axis
- Y-axis
- Optional grouping column
- Plot color
- Plot title

---

## Multivariate Visualization

Multivariate visualization examines relationships between multiple numerical variables.

The current V2 implementation provides:

- Correlation Matrix
- Correlation Heatmap

Only numerical columns are used for the correlation calculation.

---

# Model Training and Comparison

PyCaret is used to simplify the Machine Learning workflow.

Instead of manually implementing and testing multiple algorithms, the application can use PyCaret's automated model comparison functionality.

The general process is:

```text
Prepare Dataset
      │
      ▼
Initialize ML Environment
      │
      ▼
Train / Compare Models
      │
      ▼
Evaluate Models
      │
      ▼
Select Best Available Model
```

The available models and evaluation metrics depend on the selected Machine Learning task and the characteristics of the dataset.

---

# Model Saving and Download

After a Machine Learning model is trained, ML Maestro allows the resulting model to be saved.

The model can be represented as a Python pickle file:

```text
best_model.pkl
```

The trained model can then be downloaded through the Streamlit interface.

This allows users to retain the trained model for later use.

---

# Project Structure

The repository is organized to preserve both V1 and V2.

```text
ML_Maestro/
│
├── ML_Maestro.py
├── requirements.txt
├── README.md
│
└── versions/
    └── v1/
        ├── app.py
        ├── README.md
        ├── requirements.txt
        ├── best_model.pkl
        ├── Social_Network_Ads.csv
        ├── emg_data1.csv
        ├── sourcedata.csv
        ├── logs.log
        └── ML_Maestro.code-workspace
```

## V2 Files

The current V2 application is located at the repository root:

```text
ML_Maestro.py
```

The V2 dependencies are defined in:

```text
requirements.txt
```

The main project documentation is:

```text
README.md
```

## V1 Files

The original V1 implementation is preserved under:

```text
versions/v1/
```

The main V1 application is:

```text
versions/v1/app.py
```

V1 also contains its original:

- Requirements file
- Sample datasets
- Trained model
- Logs
- Workspace configuration
- Documentation

---

# Technologies Used

| Technology | Purpose |
|---|---|
| Python | Application and Machine Learning programming language |
| Streamlit | Web-based user interface |
| Pandas | Dataset loading and manipulation |
| PyCaret | Automated Machine Learning |
| YData Profiling | Automated Exploratory Data Analysis |
| Plotly | Interactive visualizations |
| Matplotlib | Data visualization |
| Seaborn | Statistical visualization and correlation analysis |
| Scikit-learn | Machine Learning functionality and supporting algorithms |
| Git | Version control |
| GitHub | Source code repository |

---

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

---

# Version History

## V1 - Original Implementation

V1 introduced the core ML Maestro concept:

- Streamlit interface
- Dataset upload
- Data exploration
- Data profiling
- Classification
- Automated model comparison
- Model saving
- Model download

The original V1 implementation is preserved under:

```text
versions/v1/
```

---

## V2 - Expanded Implementation

V2 expanded the project with:

- Classification
- Regression
- Clustering
- Interactive data visualization
- Univariate visualization
- Bivariate visualization
- Multivariate visualization
- Correlation analysis
- Automated data profiling
- Model comparison
- Model saving
- Model download

The V2 application is maintained at the repository root:

```text
ML_Maestro.py
```

---

# Repository Versioning

The repository intentionally maintains the two versions separately.

```text
ML_Maestro/
│
├── ML_Maestro.py          ← Current V2
├── requirements.txt       ← V2 dependencies
├── README.md
│
└── versions/
    └── v1/
        ├── app.py         ← Original V1
        ├── README.md
        ├── requirements.txt
        ├── best_model.pkl
        ├── Social_Network_Ads.csv
        ├── emg_data1.csv
        ├── sourcedata.csv
        ├── logs.log
        └── ML_Maestro.code-workspace
```

This structure allows users to:

- Run the current V2 application
- Access the original V1 implementation
- Compare V1 and V2
- Understand the development progression
- Preserve the original project implementation

---

# Limitations

ML Maestro is intended to simplify common Machine Learning and data analysis workflows.

Users should still understand the characteristics of their dataset and Machine Learning problem before interpreting model results.

Potential limitations include:

- Dataset size may affect application performance.
- PyCaret and its dependencies can require significant installation resources.
- Some datasets may require preprocessing before Machine Learning.
- Machine Learning results depend on the quality and characteristics of the input data.
- Automated model comparison does not replace domain knowledge.
- Not every dataset is suitable for every Machine Learning task.
- Clustering results require interpretation of the resulting groups.
- Different datasets may require different preprocessing and feature engineering strategies.

---

# Future Enhancements

Potential future enhancements include:

- Additional Machine Learning algorithms
- Advanced preprocessing options
- Feature engineering
- Feature selection
- Hyperparameter tuning
- Additional visualization types
- Model evaluation dashboards
- Model explainability
- Prediction interfaces
- Model deployment
- REST API integration
- Additional dataset formats
- Improved error handling
- Advanced experiment tracking
- Automated report generation

---

# Contributors

## Ananya Kavili

Developer and contributor to the ML Maestro project.

GitHub:

https://github.com/Ananyakavili

## Sheba Sulthana

Contributor to the original ML Maestro V1 implementation.

GitHub:

https://github.com/shebasulthana

---

# License

Please refer to the repository for the applicable license and usage terms.

If a specific license file is added to the repository in the future, the license information should be updated here accordingly.

---

# Project Status

**Current Version: V2**

V2 is the current version of ML Maestro and is located at the repository root.

The original V1 implementation is preserved under:

```text
versions/v1/
```

The repository maintains both versions to preserve the project's development history and provide access to the original implementation.

---

# Quick Start

For users who want to start the current V2 application quickly:

## 1. Clone the repository

```powershell
git clone https://github.com/Ananyakavili/ML_Maestro.git
```

## 2. Enter the project directory

```powershell
cd ML_Maestro
```

## 3. Create a virtual environment

```powershell
python -m venv venv
```

## 4. Activate the environment

```powershell
.\venv\Scripts\Activate.ps1
```

## 5. Upgrade pip

```powershell
python -m pip install --upgrade pip
```

## 6. Install dependencies

```powershell
pip install -r requirements.txt
```

## 7. Run ML Maestro V2

```powershell
streamlit run ML_Maestro.py
```

## 8. Open the application

Open:

```text
http://localhost:8501
```

---

# ML Maestro at a Glance

```text
                         ML MAESTRO
                              │
                 Zero-Code ML Platform
                              │
                 ┌────────────┴────────────┐
                 │                         │
                 ▼                         ▼
                V1                        V2
         Original Version           Current Version
                 │                         │
                 ▼                         ▼
          Dataset Upload           Dataset Upload
          Data Profiling           Data Profiling
          Classification           Visualization
          Model Comparison         Classification
          Model Saving             Regression
          Model Download           Clustering
                                   Model Comparison
                                   Model Saving
                                   Model Download
```

---

# Final Project Workflow

The overall ML Maestro concept can be summarized as:

```text
                    ┌──────────────────────┐
                    │      ML Maestro      │
                    └──────────┬───────────┘
                               │
                               ▼
                       Upload Dataset
                               │
                               ▼
                      Explore the Data
                               │
                 ┌─────────────┼─────────────┐
                 │             │             │
                 ▼             ▼             ▼
             Profiling    Visualization  Statistics
                 │             │             │
                 └─────────────┼─────────────┘
                               │
                               ▼
                       Select ML Task
                               │
              ┌────────────────┼────────────────┐
              │                │                │
              ▼                ▼                ▼
        Classification      Regression      Clustering
              │                │                │
              └────────────────┼────────────────┘
                               │
                               ▼
                        Train Models
                               │
                               ▼
                       Compare Models
                               │
                               ▼
                          Best Model
                               │
                               ▼
                       Save Model (.pkl)
                               │
                               ▼
                       Download Model
```

ML Maestro provides a unified interface for exploring datasets, visualizing data, performing Machine Learning tasks, comparing models, and downloading trained models without requiring users to implement the complete Machine Learning workflow manually.