import streamlit as st
import pandas as pd
import numpy as np
import re

from datasets import load_dataset
import plotly.express as px
from gradio_client import Client

from ydata_profiling import ProfileReport
from streamlit_pandas_profiling import st_profile_report


pd.set_option('display.max_colwidth', None)

# Function to read and apply CSS from the local file
def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

# Layout for tab navigation
def generate_tabs():
    tabs = ["Home", "Analysis", "Optimization", "Visualization", "About"]
    emojis = ["🏠", "🔍", "🔑", "📊", "ℹ️"]

    if 'current_tab' not in st.session_state:
        st.session_state.current_tab = tabs[0]

    col1, col2, col3, col4, col5 = st.columns(5)
    columns = [col1, col2, col3, col4, col5]

    for idx, (col, tab, emoji) in enumerate(zip(columns, tabs, emojis)):
        with col:
            if tab == st.session_state.current_tab:
                st.button(f"{emoji} {tab}", key=f"tab-{idx}", disabled=True)
            else:
                if st.button(f"{emoji} {tab}", key=f"tab-{idx}"):
                    st.session_state.current_tab = tab

@st.cache_data
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

@st.cache_data
def knapsack(weights, values, W):
    n = len(values)
    dp = [[0 for x in range(W + 1)] for i in range(n + 1)]

    # Build the DP table
    for i in range(1, n + 1):
        for w in range(1, W + 1):
            if weights[i - 1] <= w:
                dp[i][w] = max(dp[i - 1][w], values[i - 1] + dp[i - 1][w - weights[i - 1]])
            else:
                dp[i][w] = dp[i - 1][w]
    
    # Find out the models that are included in the final selection
    selected_index = []
    w = W
    for i in range(n, 0, -1):
        if dp[i][w] != dp[i-1][w]:  # This means the item was included
            selected_index.append(i - 1)
            w -= weights[i - 1]  # Reduce the weight from the total weight
        
    return selected_index[::-1]  # Return the list in the order of the items

@st.cache_data
def extract_model_name(row):
    model_url_pattern = r'href="([^"]+)"'
    
    model_url_match = re.search(model_url_pattern, row['Model'])

    if model_url_match:
        model_name = model_url_match.group(1).split("/")[-1]
        return model_name

    return row['model_name_for_query']

@st.cache_data
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

@st.cache_data
def generate_final_df(num_models):
    final_df = pd.DataFrame()
    with open("model_names.txt", 'r') as file:
        lines = file.readlines()
    progress_bar = st.progress(0) 
    with st.info('Fetching data, please wait...'):# Create an empty placeholder for the progress bar
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
    st.success('Data fetched successfully!')
    return final_df

@st.cache_data(experimental_allow_widgets=True)
def visualize_models(final_df):
    
        # Main title
    st.header("Interactive Model Domain Visualizations")
    # Set default domains
    default_domains = ["arc:challenge", "hellaswag", "truthfulqa:mc", "winogrande", "gsm8k", "all"]

    # Sidebar for domain and model_name selection
    with st.sidebar:
            # 'Select All' feature option
        all_domains = list(final_df.columns[1:])  # Exclude the 'model_name' column
        selected_domains = st.multiselect(
            'Select domains to visualize:',
            options=['Select All'] + all_domains,
            default=default_domains
        )
        # Check if 'Select All' was chosen or if the user's selection matches all domains
        if 'Select All' in selected_domains or set(selected_domains) == set(all_domains):
            selected_domains = all_domains  # If so, use all domains
        
        # Removing 'Select All' from the list if there are other domains selected
        if 'Select All' in selected_domains and len(selected_domains) > 1:
            selected_domains.remove('Select All')
            
      
    
    with st.expander("Bar Chart (Mean Accuracy per Model)", expanded=True):
        # Bar chart: Mean accuracy across selected domains for each model
        mean_accuracies = final_df[selected_domains].mean(axis=1)
        final_df['mean_accuracy'] = mean_accuracies
        fig_bar = px.bar(
            final_df.nlargest(10, 'mean_accuracy'), 
            x='mean_accuracy', 
            y='model_name', 
            orientation='h',
            color='mean_accuracy',
            title='Mean Accuracy Across Selected Domains'
        )
        fig_bar.update_layout(plot_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig_bar, use_container_width=True)

    with st.expander("Line Chart (Trend of Accuracy per Domain)", expanded=True):
        # Line chart: Trend of model accuracies across selected domains
        domain_to_plot = st.selectbox('Select a Domain for line chart:', final_df.columns[1:], key='domain_line_chart')
        fig_line = px.line(
            final_df, 
            x='model_name', 
            y=domain_to_plot, 
            title=f'Accuracy Trend for {domain_to_plot}'
        )
        fig_line.update_layout(plot_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig_line, use_container_width=True)

    
    with st.expander("Radial Bar Chart (Average Accuracy)", expanded=True):
        # Radial bar chart: Median accuracy as radial bars
        median_accuracies = final_df[selected_domains].median(axis=1)
        fig_radial_bar = px.bar_polar(
            final_df, 
            r=median_accuracies, 
            theta='model_name', 
            color=selected_domains[0],  # Color based on the first selected domain
            title='Radial Bar Chart: Median Accuracy per Model',
            template='plotly_dark'
        )
        st.plotly_chart(fig_radial_bar, use_container_width=True)

 # Scatter Plot
    with st.expander("Scatter Plot (Accuracy Across Models)", expanded=True):
        selected_domain_scatter = st.selectbox(
            'Select domain for scatter plot:',
            selected_domains,
            key='domain_scatter_plot'
        )
        fig_scatter = px.scatter(
            final_df, 
            x='model_name', 
            y=selected_domain_scatter, 
            color='model_name',
            title=f'Model Performances on {selected_domain_scatter}'
        )
        st.plotly_chart(fig_scatter, use_container_width=True)
    
    # Tree Map
    with st.expander("Tree Map (Hierarchical View of Domain Accuracies)", expanded=True):
        fig_tree = px.treemap(
            final_df.melt(id_vars='model_name', value_vars=selected_domains, var_name='Domain', value_name='Accuracy'),
            path=['model_name', 'Domain'],
            values='Accuracy',
            title='Tree Map of Domain Accuracies'
        )
        st.plotly_chart(fig_tree, use_container_width=True)
    
    # Area Plot
    with st.expander("Area Plot (Aggregate Domain Accuracy)", expanded=True):
        selected_domain_area = st.selectbox(
            'Select domain for area plot:',
            selected_domains,
            key='domain_area_plot'
        )
        fig_area = px.area(
            final_df,
            x='model_name',
            y=selected_domain_area,
            title=f'Area Plot of {selected_domain_area} Accuracy Over Models'
        )
        st.plotly_chart(fig_area, use_container_width=True)
    # Bubble Chart
    with st.expander("Bubble Chart (Domain Accuracies and Counts)", expanded=True):
        selected_domain_bubble = st.selectbox(
            'Select domain for bubble chart:',
            selected_domains,
            key='domain_bubble_chart'
        )
        final_df['count'] = 1  # Placeholder for count dimension
        fig_bubble = px.scatter(
            final_df,
            x='model_name', 
            y=selected_domain_bubble, 
            size='count',  # Size of the bubble could be another meaningful metric
            color='model_name',
            title=f'Bubble Chart of Accuracies for {selected_domain_bubble}'
        )
        st.plotly_chart(fig_bubble, use_container_width=True)

    # Box Plot
    with st.expander("Box Plot (Distribution of Accuracy)", expanded=True):
        fig_box = px.box(
            final_df, 
            y=selected_domains, 
            title='Box Plot of Model Accuracies'
        )
        st.plotly_chart(fig_box, use_container_width=True)
    
        # Heatmap
    with st.expander("Heatmap (Model Accuracy Across Domains)", expanded=True):
        fig_heatmap = px.imshow(
            final_df.set_index('model_name')[selected_domains], 
            title='Heatmap of Model Accuracy Across Domains'
        )
        st.plotly_chart(fig_heatmap, use_container_width=True)
   

@st.cache_data
def generate_profile(df):
     st.info('Generating profile report... this may take a while with large datasets.')
     pr = df.profile_report()
     return pr

# Visualization code using final_df
# This function contains the code for creating and displaying visualizations
@st.cache_data(experimental_allow_widgets=True)
def optimize_model_selection():
    with open("model_names.txt", 'r') as file:
        lines = file.readlines()
    num_models = st.slider("Number of Models to Consider", min_value=1, max_value=len(lines), value=10)
    final_df = generate_final_df(num_models)
    st.write(final_df)
    
    selected_domain = st.selectbox('Select a domain', final_df.columns[1:])
    
    # We'll have a uniform weight of 1 for each model, for simplicity.
    weights = [1 for _ in range(num_models)]
    values = final_df[selected_domain].values.tolist()
    
    # The weight_parameter now represents the maximum number of models to select
    weight_parameter = st.slider('Select the maximum number of models to select', min_value=1, max_value=num_models, value=5)
    
    selected_indices = knapsack(weights, values, weight_parameter)
    
    st.subheader('Optimal Model Selection')
    
    optimal_model_data = []
    for idx in selected_indices:
        domain = selected_domain
        model_name = final_df.iloc[idx]['model_name']
        accuracy = final_df.iloc[idx][selected_domain]
        
        suggestion = f"The organization can consider adopting the model '{model_name}' within the domain '{domain}' for better performance based on its high accuracy score."
        
        optimal_model_data.append({
            'Domain': domain,
            'Model': model_name,
            'Accuracy': accuracy,
            'Suggestion': suggestion
        })
        
    st.dataframe(pd.DataFrame(optimal_model_data))
    st.markdown("> **Note:** The selection optimizes for accuracy while respecting the constraint on the number of models.")
    


def create_footer():
    st.markdown(
        """
        <hr style="height:2px;border:none;color:#333;background-color:#333;" />  
        <div style="position: relative; height: 50px; bottom: 0; width:100%; text-align: center; padding: 10px;">
            <p style="color: #999; font-size: 0.9em;">&copy; 2024 Vidhya Varshany. All rights reserved.</p>
         
        </div>
        """,
        unsafe_allow_html=True
    )
#  /  <p style="color: #999; font-size: 0.9em;">For more information, visit <a href="https" target="_blank">www.example.com</a>.</p>
    
def main():
    st.set_page_config(page_title="Dynamic LLM Ensemble Selection", 
                       page_icon="assets/favicon.ico",  layout="wide")
    local_css("style.css")
        # Custom CSS to inject for reducing top margin of the title
    st.markdown("""
       <style>
       .css-158tj05 {
           margin-bottom: 0rem;
           margin-top: -1px; /* Adjust top margin as required */
       }
       .css-16huue1 { /* this is for the navigation buttons padding */
           padding-bottom: 1rem;
       }
        h1 {
        text-align: center;
        line-height: -10; /* Decreased line-height */
    }
       </style>
       """, unsafe_allow_html=True)
    
    # Inject a custom CSS to reduce the padding at the top of the page
    st.markdown("""
    <style>
    .main .block-container {
        padding-top: 2rem; /* reduces the top padding */
    }
    .main .block-container .css-18e3th9 {
        padding-top: 0rem; /* reduces the top padding for the first element inside block-container */
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Title of your app (adjusted with reduced space)
    st.markdown("<h1 style='text-align: center;'>Dynamic LLM Ensemble Selection Dashboard</h1>", unsafe_allow_html=True)
    
        # Initialize the current tab if it doesn't exist in the session state
    if 'current_tab' not in st.session_state:
        st.session_state.current_tab = "Home"

    # Run the tab generation function
    generate_tabs()

    # Main app - This switches between pages based on what the user has selected
    if st.session_state.current_tab == "Home":
     

        df = get_api_data()
        st.dataframe(df)
        
        st.subheader("Pandas Profiling")
        
                # Use the cached function in your app
        if st.button("Generate Profile"):
            with st.container(height=600): 
                profile = generate_profile(df)
                st_profile_report(profile)
        

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

    elif st.session_state.current_tab == "Analysis":
        st.title("Model Analysis")
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
                
        
    elif st.session_state.current_tab == "Optimization":
        st.title("Knapsack Optimization")# Accessing final_df again for optimization
        optimize_model_selection()
          
    elif st.session_state.current_tab == "Visualization":
        st.title("Data Visualization")
        filename = "model_names.txt"
        with open(filename, 'r') as file:
            lines = file.readlines()
        num_models = st.slider("Number of Models to Consider", min_value=1, max_value=len(lines), value=10)

        final_df = generate_final_df(num_models)  # Accessing final_df again for visualization
        visualize_models(final_df)
 
    elif st.session_state.current_tab == "About":
            # Application Overview
        st.markdown("""
            ### Application Overview
            *Dynamic LLM Ensemble Selection* is an interactive platform that employs knapsack optimization 
            to guide the selection of language model ensembles.

            ### Key Features
            - **Comparative Analysis**: Contrast and compare language models in detail.
            - **Resource Optimization**: Employ knapsack algorithm principles to identify the most cost-effective models.
            - **Visual Insights**: Investigate model performance through an array of dynamic visualizations.
            - **Usability Focus**: Navigate a user-centric interface that simplifies complex data interpretation.

            ### Evaluation Methodology
            
            The evaluation harnesses a comprehensive suite of benchmarks for a holistic assessment of language models, including:
            """)
        
        # Use columns to list the benchmarks in a structured manner
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("""
            - **ARC**: _AI2 Reasoning Challenge_
            Grading based on science questions
            - **HellaSwag**:
            Tests on common sense reasoning
            """)
            
        with col2:
            st.markdown("""
            - **MMLU**: _Massive Multi-Task Language Understanding_
            Evaluation over multiple domains
            - **TruthfulQA**: 
            Focused on veracity in responses
            """)
            
        with col3:
            st.markdown("""
            - **Winogrande**: 
            Solves complex sentence completions
            - **GSM8K**: 
            Examines mathematical problem-solving abilities
            """)

        st.markdown("---")

        with st.expander("Citation"):
            st.write("""
        If you find this tool helpful, please consider supporting it by citing it in your publications or sharing it within your network.
    """)
            st.code("""
                @misc{dynamic_llm_selection,
                    title={Dynamic LLM Ensemble Selection with Knapsack Optimization},
                    author={Vidhya Varshany J S},
                    year={2024}
                }
            """, language='bibtex')

            # Credits
        with st.expander("Credits"):   
            st.markdown("""
                Special thanks to the Hugging Face community for providing the open LLM leaderboard and the resources to build this application:
                - [Hugging Face Open LLM Leaderboard](https://huggingface.co/spaces/HuggingFaceH4/open_llm_leaderboard)
                - [FelixZ Open LLM Leaderboard](https://huggingface.co/spaces/felixz/open_llm_leaderboard)
                """)
        with st.expander("References"):
        # References
            st.markdown("""
                To delve deeper into our benchmarks and methodology, consider these resources:
                
                - [Open LLM Leaderboard Overview](https://deepnatural.ai/blog/which-llm-is-better-open-llm-leaderboard-en)
                - [The Knapsack Problem and Model Management](https://www.awelm.com/posts/knapsack/)
                """)

        # Contact Information
        st.markdown("""
            ### Contact Information
            Questions, feedback, or need assistance? 
            
            - Email at: [vidhyavarshany@gmail.com](mailto:vidhyavarshany@gmail.com)
            """)
        
        local_css('style.css')

        # ... (other parts of the about page) ...
        
        # GitHub link button
        st.markdown("""
            ### Follow the Development Journey
            Stay updated with the project's progress and all upcoming features.
        """)
        
        # The actual link wrapped around with the right styling
        github_url = "https://github.com/VidhyaVarshanyJS/EnsembleX"
        st.markdown(f'<a href="{github_url}" target="_blank" class="button-style">GitHub Project</a>', unsafe_allow_html=True)
       
        create_footer()



if __name__ == "__main__":
    main()
