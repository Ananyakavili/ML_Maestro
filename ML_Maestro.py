import streamlit as st
import pandas as pd
import os
import matplotlib.pyplot as plt
import plotly.express as px
import seaborn as sns
import numpy as np
from pycaret.classification import setup as setup_classification, compare_models as compare_models_classification, pull as pull_classification, save_model as save_model_classification, load_model as load_model_classification
from pycaret.regression import setup as setup_regression, compare_models as compare_models_regression, pull as pull_regression, save_model as save_model_regression, load_model as load_model_regression
from pycaret.clustering import setup as setup_clustering, create_model, pull as pull_clustering, save_model as save_model_clustering, load_model as load_model_clustering
from ydata_profiling import ProfileReport
from streamlit_pandas_profiling import st_profile_report

# Sidebar for navigation
with st.sidebar:
    st.image("https://i.pinimg.com/236x/73/cd/2a/73cd2a910f69d1fb3a61979364a22bff.jpg")
    st.title("ML Maestro")
    choice = st.radio("Navigation", ["Upload Dataset", "Data Profiling", "Data Visualization", "ML", "Download Model"])
    st.info("This is a Zero Code Web Application to build Automated ML pipeline.")

# Check if the dataset is already available
if os.path.exists("sourcedata.csv"):
    df = pd.read_csv("sourcedata.csv", index_col=None)

# Dataset Upload Section
if choice == "Upload Dataset":
    st.title("Upload Your Dataset For Modelling")
    file = st.file_uploader("Upload Your Dataset Here")
    if file:
        df = pd.read_csv(file, index_col=None)
        df.to_csv("sourcedata.csv", index=None)
        st.dataframe(df)

# Data Profiling Section (Can remain unchanged)
if choice == "Data Profiling":
    st.title("Automated Exploratory Data Analysis")
    if 'df' in locals():
        profile_report = ProfileReport(df)
        st_profile_report(profile_report)
    else:
        st.error("No data uploaded yet. Please upload a dataset.")

# Data Visualization Section
if choice == "Data Visualization":
    st.title("Interactive Data Visualization")

    if 'df' in locals():
        st.sidebar.write("Use the controls below to filter and visualize your data.")

        # Allow the user to filter data based on multiple columns
        filters = st.sidebar.multiselect("Filter data by", df.columns)
        filtered_df = df.copy()

        if filters:
            for filter_column in filters:
                unique_values = df[filter_column].unique()
                selected_values = st.sidebar.multiselect(f"Select values for {filter_column}", unique_values, default=unique_values)
                filtered_df = filtered_df[filtered_df[filter_column].isin(selected_values)]

        # Visualization type selection (Univariate, Bivariate, Multivariate)
        viz_type = st.sidebar.selectbox("Choose Visualization Type", ["Univariate", "Bivariate", "Multivariate"])

        # Plot Type Selection based on visualization type
        if viz_type == "Univariate":
            plot_type = st.sidebar.selectbox("Choose Plot Type", ["Bar Chart", "Histogram"])
            x_axis = st.sidebar.selectbox("Select Feature (X-axis)", filtered_df.columns)

            # Custom color selection
            color = st.sidebar.color_picker("Pick a color", "#00f900")
            plot_title = st.sidebar.text_input("Plot Title", f"{plot_type} of {x_axis}")

            # Plot rendering
            if plot_type == "Bar Chart":
                value_counts_df = filtered_df[x_axis].value_counts().reset_index()
                value_counts_df.columns = [x_axis, 'count']
                fig = px.bar(value_counts_df, x=x_axis, y='count', title=plot_title, color_discrete_sequence=[color])
                st.plotly_chart(fig)

            elif plot_type == "Histogram":
                fig = px.histogram(filtered_df, x=x_axis, title=plot_title, color_discrete_sequence=[color])
                st.plotly_chart(fig)

            # Summary Statistics
            st.write(f"Summary statistics for {x_axis}:")
            st.write(filtered_df[x_axis].describe())

        elif viz_type == "Bivariate":
            plot_type = st.sidebar.selectbox("Choose Plot Type", ["Scatter Plot", "Line Plot", "Box Plot"])
            x_axis = st.sidebar.selectbox("Select X-axis", filtered_df.columns)
            y_axis = st.sidebar.selectbox("Select Y-axis", filtered_df.columns)

            # Group By Category selection for color coding (optional)
            group_by = st.sidebar.selectbox("Group By (Optional)", [None] + list(filtered_df.columns[df.dtypes == 'object']))

            # Custom color selection
            color = st.sidebar.color_picker("Pick a color", "#00f900")
            plot_title = st.sidebar.text_input("Plot Title", f"{plot_type} of {x_axis} vs {y_axis}")

            # Plot rendering
            if plot_type == "Scatter Plot":
                fig = px.scatter(filtered_df, x=x_axis, y=y_axis, color=group_by, title=plot_title, color_discrete_sequence=[color])
                st.plotly_chart(fig)

            elif plot_type == "Line Plot":
                fig = px.line(filtered_df, x=x_axis, y=y_axis, color=group_by, title=plot_title, color_discrete_sequence=[color])
                st.plotly_chart(fig)

            elif plot_type == "Box Plot":
                fig = px.box(filtered_df, x=x_axis, y=y_axis, color=group_by, title=plot_title, color_discrete_sequence=[color])
                st.plotly_chart(fig)

            # Summary Statistics
            st.write(f"Summary statistics for {x_axis} vs {y_axis}:")
            st.write(filtered_df[[x_axis, y_axis]].describe())

        # Only Correlation Heatmap for multivariate
        elif viz_type == "Multivariate":
            st.write("Correlation Heatmap")

            # Select only numeric columns from the dataframe
            numeric_df = filtered_df.select_dtypes(include=['float64', 'int64'])

            if numeric_df.empty:
                st.error("No numeric columns available for correlation analysis.")
            else:
                # Compute correlation matrix on numeric data
                corr = numeric_df.corr()

                # Plot the heatmap using seaborn
                fig, ax = plt.subplots(figsize=(10, 6))
                sns.heatmap(corr, annot=True, cmap="coolwarm", ax=ax)
                st.pyplot(fig)

                # Display correlation matrix
                st.write("Correlation Matrix:")
                st.dataframe(corr)

    else:
        st.error("No data available for visualization. Please upload a dataset.")

# ML Model Training Section
if choice == "ML":
    st.title("ML Task Selection")
    if 'df' in locals():
        task = st.selectbox("Select Your ML Task", ["Classification", "Regression", "Clustering"])
        target = st.selectbox("Select Your Target", df.columns)

        if st.button("Train model"):
            st.info("Setting up ML Task...")

            # Classification Task
            if task == "Classification":
                setup_classification(df, target=target)
                setup_df = pull_classification()
                st.dataframe(setup_df)

                st.info("Comparing Models...")
                best_model = compare_models_classification()
                compare_df = pull_classification()
                st.dataframe(compare_df)
                save_model_classification(best_model, 'best_model.pkl')

            # Regression Task
            elif task == "Regression":
                setup_regression(df, target=target)
                setup_df = pull_regression()
                st.dataframe(setup_df)

                st.info("Comparing Models...")
                best_model = compare_models_regression()
                compare_df = pull_regression()
                st.dataframe(compare_df)
                save_model_regression(best_model, 'best_model.pkl')

            # Clustering Task
            elif task == "Clustering":
                setup_clustering(df)
                setup_df = pull_clustering()
                st.dataframe(setup_df)

                st.info("Creating Clustering Model...")
                best_model = create_model('kmeans')  # Use KMeans or any other clustering model
                save_model_clustering(best_model, 'best_model.pkl')
                st.success("Clustering Model Created!")

    else:
        st.error("No data available for model training. Please upload a dataset.")


# Download Model Section
if choice == "Download Model":
    if os.path.exists("best_model.pkl"):
        with open("best_model.pkl", 'rb') as f:
            st.download_button("Download the model", f, "trained_model.pkl")
    else:
        st.error("No trained model available for download. Please train a model first.")
