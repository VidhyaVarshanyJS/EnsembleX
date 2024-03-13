import streamlit as st
import pandas as pd
import numpy as np
import re
from joblib import Memory
from datasets import load_dataset
import plotly.express as px
from gradio_client import Client
import time
from ydata_profiling import ProfileReport
from streamlit_pandas_profiling import st_profile_report
import matplotlib.pyplot as plt


memory = Memory('./cachedir', verbose=0)
pd.set_option('display.max_colwidth', None)

@memory.cache
# @st.cache_data
def process_model_data(details):
    data = load_dataset("open-llm-leaderboard/"+details, data_files="results_*.json", split="train")
    config_general = data[0]['config_general']['model_name']
    results = data[0]['results']

    df = pd.DataFrame(results).T
    df['accuracy'] = df['acc_norm'].combine_first(df['mc2'])
    df['accuracy'] = df['accuracy'].fillna(df['acc'])
    df = df[['accuracy']]
    df = df.T.reset_index(drop=True)
    df.insert(0, 'model_name', config_general)

    def clean_column_name(column_name):
        clean_name = re.sub(r'^harness\|', '', column_name)
        clean_name = re.sub(r'\|.*?\|', '', clean_name)
        clean_name = re.sub(r'\|[0-9]+$', '', clean_name)
        clean_name = clean_name.replace('hendrycksTest', '')
        clean_name = clean_name.replace('-', '')
        return clean_name

    df.columns = df.columns.map(clean_column_name)
    return df

def knapsack(scores, W):
    n = len(scores)
    dp = np.zeros((n + 1, W + 1))

    for i in range(1, n + 1):
        for w in range(1, W + 1):
            if w >= 1:
                dp[i][w] = max(dp[i - 1][w], scores[i - 1] + dp[i - 1][w - 1])

    selected_index = []
    i, j = n, W
    while i > 0 and j > 0:
        if dp[i][j] != dp[i - 1][j]:
            selected_index.append(i - 1)
            j -= 1
        i -= 1

    return selected_index

def extract_model_name(row):
    model_url_pattern = r'href="([^"]+)"'
    
    model_url_match = re.search(model_url_pattern, row['Model'])

    if model_url_match:
        model_name = model_url_match.group(1).split("/")[-1]
        return model_name

    return row['model_name_for_query']

@memory.cache
# @st.cache_data
def get_api_data():
    # Function to get data from API
    client = Client("https://felixz-open-llm-leaderboard.hf.space/")
    json_data = client.predict("", "", api_name='/predict')
    df = pd.DataFrame(json_data['data'], columns=json_data['headers'])

    model_names_text = df['model_name_for_query'].to_string(index=False, header=False)
    filename = "model_names.txt"
    with open(filename, 'w') as file:
        file.write(model_names_text)

    df['Model Name'] = df.apply(extract_model_name, axis=1)
    desired_order = ['T', 'Model Name', 'Average ⬆️', 'HellaSwag', 'MMLU', 'TruthfulQA', 'Winogrande', 'GSM8K', 'Type', 'Architecture', 'Weight type', 'Precision', 'Merged', '#Params (B)', 'Hub ❤️', 'Available on the hub', 'Model sha', 'Flagged', 'MoE']
    df = df[desired_order]

    return df

@memory.cache
def generate_final_df(num_models):
    final_df = pd.DataFrame()
    with open("model_names.txt", 'r') as file:
        lines = file.readlines()
    progress_bar = st.progress(0)  # Create an empty placeholder for the progress bar
    for i, line in enumerate(lines[:num_models]):
        model_name = line.strip()
        details = f"details_{model_name.replace('/', '__')}"
        try:
            df = process_model_data(details)
        except Exception as e:
            st.warning(f"An error occurred while processing model: {details}. Skipping this model.")
              # Update the progress bar in the Model Analysis view
        progress = (i + 1) / num_models
        progress_bar.progress(progress)

        final_df = pd.concat([final_df, df], ignore_index=True)

    return final_df



def visualize_models(final_df):
    st.subheader('Visualization')
        
    selected_domains = st.multiselect('Select domains', final_df.columns[1:], default=["arc:challenge", "hellaswag", "truthfulqa:mc", "winogrande", "gsm8k", "all"])

    fig = px.bar(final_df, x='model_name', y=selected_domains, barmode='group',
                labels={'model_name': 'Model', 'value': 'Accuracy'}, title='Model Performance Across Domains')
    fig.update_layout(xaxis_tickangle=-45)
    fig.update_layout(height=800, width=800)
    st.plotly_chart(fig)
    

        
     # Visualization code using final_df
    # This function contains the code for creating and displaying visualizations
def optimize_model_selection():
    with open("model_names.txt", 'r') as file:
        lines = file.readlines()
    num_models = st.slider("Number of Models to Consider", min_value=1, max_value=len(lines), value=10)
    final_df = generate_final_df(num_models)
    st.write(final_df)
    selected_domain = st.selectbox('Select a domain', final_df.columns[1:])
    weight_parameter = st.slider('Select a weight parameter', min_value=1, max_value=10, value=1)
    scores = final_df[selected_domain].values
    selected_index = knapsack(scores, weight_parameter)

    st.subheader('Optimal Model Selection')
    
    optimal_model_data = []
    for idx in selected_index:
        domain = selected_domain
        model_name = final_df['model_name'][idx]
        accuracy = final_df[selected_domain][idx]
        suggestion = f"The organization operating in the {domain} can consider adopting the optimal model '{model_name}' for better performance."
        optimal_model_data.append({'Domain': domain, 'Optimal Model': model_name, 'Accuracy': accuracy,'Suggestion': suggestion})

    st.table(pd.DataFrame(optimal_model_data))
    st.markdown("> **Common Suggestion:** Evaluate model scalability and compatibility with business goals.")
    
    
    # Knapsack optimization code using final_df
    # This function contains the code for implementing knapsack optimization
    
def main():
    st.title("Dynamic LLM Ensemble Selection with Knapsack Optimization")

    page = st.sidebar.selectbox("Select View", ["Top Performing Models", "Model Analysis","Knapsack Optimization","Visualization", "About"])

    if page == "Top Performing Models":
        df = get_api_data()
        st.dataframe(df)
        
        st.subheader("Pandas Profiling")
        
        with st.container(height=600): 
            pr = df.profile_report()
            st_profile_report(pr)

        evaluation_methodology = """
        ### Evaluation Methodology
        The evaluation process involves running your models against several benchmarks from the Eleuther AI Harness, a unified framework for measuring the effectiveness of generative language models. Below is a brief overview of each benchmark:

        - **AI2 Reasoning Challenge (ARC)**: Grade-School Science Questions (25-shot)
        - **HellaSwag**: Commonsense Inference (10-shot)
        - **MMLU**: Massive Multi-Task Language Understanding, knowledge on 57 domains (5-shot)
        - **TruthfulQA**: Propensity to Produce Falsehoods (0-shot)
        - **Winogrande**: Adversarial Winograd Schema Challenge (5-shot)
        - **GSM8k**: Grade School Math Word Problems Solving Complex Mathematical Reasoning (5-shot)
        
        --- 
        - 🟢 Pretrained Model: This icon represents new, base models that have been trained on a given corpus. These are foundational models created from the ground up.

        - 🔶 Fine-Tuned Model: This category includes pretrained models that have been further refined and improved upon by training on additional data.

        - ⭕ Instruction-Tuned Model: These are models specifically fine-tuned on datasets of task instructions. They are tailored to better understand and respond to task-specific directions.

        - 🟦 RL-Tuned Model: Indicates models that have undergone reinforcement fine-tuning. This process usually involves modifying the model's loss function with an added policy.

        Together, these benchmarks provide an assessment of a model's capabilities in terms of knowledge, reasoning, and some math, in various scenarios.
        """
        st.markdown(evaluation_methodology)

    elif page == "Model Analysis": 
        with open("model_names.txt", 'r') as file:
             lines = file.readlines()
        num_models = st.slider("Number of Models to Consider", min_value=1, max_value=len(lines), value=10)    
        final_df = generate_final_df(num_models)
        selected_columns = st.multiselect("Select columns to display", options=final_df.columns.tolist(), default=final_df.columns[0:10].tolist())
        st.write(final_df[selected_columns])

        
        # Single model insights
        selected_model = st.selectbox("Select a Model", options=lines)

        if st.button("View Selected Model Result"):
            model_name = selected_model.strip()
            details = f"details_{model_name.replace('/', '__')}"
            try:
                df = process_model_data(details)
                st.write(df)
            except Exception as e:
                st.warning(f"An error occurred while processing model: {details}. Skipping this model.")
                
        
    elif page == "Knapsack Optimization":# Accessing final_df again for optimization
        optimize_model_selection()
          
    elif page =="Visualization":
        filename = "model_names.txt"
        with open(filename, 'r') as file:
            lines = file.readlines()
        num_models = st.slider("Number of Models to Consider", min_value=1, max_value=len(lines), value=10)

        final_df = generate_final_df(num_models)  # Accessing final_df again for visualization
        visualize_models(final_df)
 
    elif page == "About":
        st.subheader("About This Application")
        about_text = """
        This application provides dynamic LLM ensemble selection using knapsack optimization. 
        It analyzes the performance of various language models across different benchmarks and domains.
        """
        st.markdown(about_text)

        st.subheader("Citation")
        reference_info = """
        If you use this application in your research or work, please consider citing it using the following information:
        - Title: Dynamic LLM Ensemble Selection with Knapsack Optimization
        - Authors: [Vidhya Varshany J S]
        - Year: [2024]
        """
        st.markdown(reference_info)

        st.subheader("Credits")
        credits_info = """
        Special thanks to the Hugging Face community for providing the open LLM leaderboard and the resources to build this application:
        - [Hugging Face Open LLM Leaderboard](https://huggingface.co/spaces/HuggingFaceH4/open_llm_leaderboard)
        - [FelixZ Open LLM Leaderboard](https://huggingface.co/spaces/felixz/open_llm_leaderboard)
        """
        st.markdown(credits_info)

        st.subheader("References")
        references_info = """
        For more information on the LLM leaderboards and evaluation methodologies, you can refer to the following article:
        - [Which LLM is Better? Open LLM Leaderboard](https://deepnatural.ai/blog/which-llm-is-better-open-llm-leaderboard-en)
        - [Managing LLM context is the knapsack problem](https://www.awelm.com/posts/knapsack/)
        
        """
        st.markdown(references_info)
        
        st.subheader("Contact")
        contact_info = """
        For inquiries or feedback, please reach out to us at vidhyavarshany@gmail.com
        """
        st.markdown(contact_info)


if __name__ == "__main__":
    main()
