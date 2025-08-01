import os
import tempfile
import streamlit as st
from streamlit_option_menu import option_menu
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_community.chat_models import ChatGroq
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langgraph.graph import END, MessageGraph
from fpdf import FPDF
from PIL import Image
import io
import base64
import pdfplumber
from typing import List, Dict, Any, Optional
from langchain_core.agents import AgentAction, AgentFinish
from langchain_core.messages import BaseMessage
from langchain.agents import tool
from crewai import Agent, Task, Crew, Process
from autogen import AssistantAgent, UserProxyAgent, config_list_from_json

# Initialize Groq LLM
GROQ_API_KEY = ""
MODEL_NAME = "mixtral-8x7b-32768"

# Set page config
st.set_page_config(
    page_title="AI Data Analyzer",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .stApp {
        background-color: #f5f5f5;
    }
    .sidebar .sidebar-content {
        background-color: #2c3e50;
        color: white;
    }
    .st-bb {
        background-color: transparent;
    }
    .st-at {
        background-color: #3498db;
    }
    .st-ax {
        color: white;
    }
    .css-1aumxhk {
        background-color: #2c3e50;
        color: white;
    }
    .css-1v0mbdj {
        border: 1px solid #3498db;
    }
    .report-title {
        font-size: 24px;
        font-weight: bold;
        color: #2c3e50;
        margin-bottom: 20px;
    }
    .section-title {
        font-size: 20px;
        font-weight: bold;
        color: #3498db;
        margin-top: 20px;
        margin-bottom: 10px;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'df' not in st.session_state:
    st.session_state.df = None
if 'processed_df' not in st.session_state:
    st.session_state.processed_df = None
if 'report_content' not in st.session_state:
    st.session_state.report_content = ""
if 'charts' not in st.session_state:
    st.session_state.charts = {}
if 'api_key' not in st.session_state:
    st.session_state.api_key = ""
if 'model_name' not in st.session_state:
    st.session_state.model_name = MODEL_NAME

# File processing functions
def process_file(uploaded_file):
    try:
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        elif uploaded_file.name.endswith('.xlsx') or uploaded_file.name.endswith('.xls'):
            df = pd.read_excel(uploaded_file)
        elif uploaded_file.name.endswith('.pdf'):
            with pdfplumber.open(uploaded_file) as pdf:
                text = "\n".join([page.extract_text() for page in pdf.pages])
                # Simple table extraction - for complex PDFs consider camelot or tabula
                tables = []
                for page in pdf.pages:
                    tables.extend(page.extract_tables())
                if tables:
                    df = pd.DataFrame(tables[0])
                    if len(tables) > 1:
                        st.warning("Multiple tables found in PDF. Using the first table.")
                else:
                    st.error("No tables found in PDF. Extracting as text.")
                    df = pd.DataFrame({"Text": [text]})
        else:
            st.error("Unsupported file format")
            return None
        
        st.session_state.df = df
        return df
    except Exception as e:
        st.error(f"Error processing file: {str(e)}")
        return None

def clean_data(df):
    try:
        # Make a copy of the original dataframe
        cleaned_df = df.copy()
        
        # Convert columns to appropriate data types
        for col in cleaned_df.columns:
            # Try to convert to numeric first
            try:
                cleaned_df[col] = pd.to_numeric(cleaned_df[col], errors='ignore')
            except:
                pass
            
            # If still object type, try to clean strings
            if cleaned_df[col].dtype == 'object':
                cleaned_df[col] = cleaned_df[col].str.strip()
                cleaned_df[col] = cleaned_df[col].replace(['', 'NA', 'N/A', 'NaN', 'null'], np.nan)
        
        # Fill missing values
        for col in cleaned_df.columns:
            if cleaned_df[col].dtype in ['int64', 'float64']:
                cleaned_df[col].fillna(cleaned_df[col].mean(), inplace=True)
            else:
                cleaned_df[col].fillna('Unknown', inplace=True)
        
        # Remove duplicates
        cleaned_df.drop_duplicates(inplace=True)
        
        st.session_state.processed_df = cleaned_df
        return cleaned_df
    except Exception as e:
        st.error(f"Error cleaning data: {str(e)}")
        return df

# Visualization functions
def create_visualizations(df):
    charts = {}
    
    # Numeric columns
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    
    # Histograms for numeric columns
    for col in numeric_cols:
        fig = px.histogram(df, x=col, title=f'Distribution of {col}', color_discrete_sequence=['#3498db'])
        charts[f'hist_{col}'] = fig
    
    # Bar charts for categorical columns
    for col in categorical_cols:
        if len(df[col].unique()) < 20:  # Avoid columns with too many unique values
            fig = px.bar(df[col].value_counts(), title=f'Count of {col}', color_discrete_sequence=['#2ecc71'])
            charts[f'bar_{col}'] = fig
    
    # Scatter plots for numeric pairs
    if len(numeric_cols) >= 2:
        fig = px.scatter(df, x=numeric_cols[0], y=numeric_cols[1], title=f'{numeric_cols[0]} vs {numeric_cols[1]}')
        charts['scatter'] = fig
    
    # Pie chart for first categorical column
    if categorical_cols:
        col = categorical_cols[0]
        if len(df[col].unique()) < 10:  # Only for small number of categories
            fig = px.pie(df, names=col, title=f'Proportion of {col}', hole=0.3)
            charts['pie'] = fig
            
            # Donut chart
            fig = px.pie(df, names=col, title=f'Donut Chart of {col}', hole=0.5)
            charts['donut'] = fig
    
    # Area chart for numeric columns over index
    if len(numeric_cols) >= 1:
        fig = px.area(df, y=numeric_cols, title='Area Chart')
        charts['area'] = fig
    
    # Correlation heatmap if enough numeric columns
    if len(numeric_cols) > 1:
        corr = df[numeric_cols].corr()
        fig = px.imshow(corr, text_auto=True, title='Correlation Heatmap')
        charts['heatmap'] = fig
    
    # Box plots for numeric columns
    for col in numeric_cols:
        fig = px.box(df, y=col, title=f'Box Plot of {col}')
        charts[f'box_{col}'] = fig
    
    st.session_state.charts = charts
    return charts

# Report generation
def generate_report(df, cleaned_df):
    report = "# Data Analysis Report\n\n"
    
    # Basic information
    report += "## Basic Information\n"
    report += f"- Original rows: {len(df)}\n"
    report += f"- Original columns: {len(df.columns)}\n"
    report += f"- Processed rows: {len(cleaned_df)}\n"
    report += f"- Processed columns: {len(cleaned_df.columns)}\n\n"
    
    # Data quality issues
    report += "## Data Quality Issues Resolved\n"
    for col in df.columns:
        original_missing = df[col].isna().sum()
        cleaned_missing = cleaned_df[col].isna().sum()
        if original_missing > 0:
            report += f"- Column '{col}': {original_missing} missing values "
            if cleaned_missing == 0:
                report += "(all imputed)\n"
            else:
                report += f"(reduced to {cleaned_missing})\n"
    
    original_duplicates = df.duplicated().sum()
    cleaned_duplicates = cleaned_df.duplicated().sum()
    if original_duplicates > 0:
        report += f"- {original_duplicates} duplicate rows removed\n"
    
    # Summary statistics
    report += "## Summary Statistics\n"
    report += "### Original Data\n"
    report += df.describe().to_markdown() + "\n\n"
    report += "### Cleaned Data\n"
    report += cleaned_df.describe().to_markdown() + "\n\n"
    
    # Column information
    report += "## Column Information\n"
    for col in cleaned_df.columns:
        report += f"### {col}\n"
        report += f"- Data type: {cleaned_df[col].dtype}\n"
        if cleaned_df[col].dtype in ['int64', 'float64']:
            report += f"- Mean: {cleaned_df[col].mean():.2f}\n"
            report += f"- Standard deviation: {cleaned_df[col].std():.2f}\n"
            report += f"- Range: {cleaned_df[col].min():.2f} to {cleaned_df[col].max():.2f}\n"
        elif cleaned_df[col].dtype == 'object':
            report += f"- Unique values: {len(cleaned_df[col].unique())}\n"
            report += f"- Most common: {cleaned_df[col].mode()[0]}\n"
        report += "\n"
    
    st.session_state.report_content = report
    return report

def create_pdf_report(report_text, charts):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    
    # Add title
    pdf.set_font("Arial", 'B', 16)
    pdf.cell(200, 10, txt="Data Analysis Report", ln=True, align='C')
    pdf.ln(10)
    
    # Add report text
    pdf.set_font("Arial", size=12)
    for line in report_text.split('\n'):
        if line.startswith('#'):
            # Handle headings
            level = line.count('#')
            if level == 1:
                pdf.set_font("Arial", 'B', 14)
                pdf.cell(200, 10, txt=line.replace('#', '').strip(), ln=True)
            elif level == 2:
                pdf.set_font("Arial", 'B', 12)
                pdf.cell(200, 10, txt=line.replace('#', '').strip(), ln=True)
            else:
                pdf.set_font("Arial", 'B', 10)
                pdf.cell(200, 10, txt=line.replace('#', '').strip(), ln=True)
            pdf.ln(5)
        else:
            pdf.multi_cell(0, 5, txt=line)
    
    # Add charts (simplified - in a real app you'd save images and add them)
    pdf.add_page()
    pdf.set_font("Arial", 'B', 14)
    pdf.cell(200, 10, txt="Data Visualizations", ln=True, align='C')
    pdf.ln(10)
    
    # For each chart, we would normally save as image and add to PDF
    # Here we just list the charts
    pdf.set_font("Arial", size=12)
    pdf.cell(200, 10, txt="Visualizations included in the report:", ln=True)
    for chart_name in charts.keys():
        pdf.cell(200, 10, txt=f"- {chart_name}", ln=True)
    
    # Save to bytes
    output = io.BytesIO()
    pdf_bytes = pdf.output(dest='S').encode('latin1')
    output.write(pdf_bytes)
    output.seek(0)
    
    return output

# LangGraph and LangChain agents
def setup_langgraph_agent():
    # Define tools
    @tool
    def analyze_data(input: str) -> str:
        """Analyze the current dataset and provide insights."""
        if st.session_state.processed_df is None:
            return "No data available for analysis. Please upload and process a file first."
        
        df = st.session_state.processed_df
        numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        
        analysis = f"Dataset has {len(df)} rows and {len(df.columns)} columns.\n"
        analysis += f"Numeric columns: {', '.join(numeric_cols)}\n"
        analysis += f"Categorical columns: {', '.join(categorical_cols)}\n"
        
        if numeric_cols:
            analysis += "\nSummary statistics:\n"
            analysis += df[numeric_cols].describe().to_markdown() + "\n"
        
        if categorical_cols:
            analysis += "\nCategorical value counts:\n"
            for col in categorical_cols:
                analysis += f"{col}:\n{df[col].value_counts().to_markdown()}\n\n"
        
        return analysis

    @tool
    def visualize_data(chart_type: str, x_axis: str, y_axis: Optional[str] = None) -> str:
        """Generate a visualization of the data based on specified parameters.
        
        Args:
            chart_type: Type of chart to generate (histogram, bar, scatter, pie, etc.)
            x_axis: Column to use for x-axis
            y_axis: Column to use for y-axis (for scatter plots, etc.)
        """
        if st.session_state.processed_df is None:
            return "No data available for visualization. Please upload and process a file first."
        
        df = st.session_state.processed_df
        fig = None
        
        try:
            if chart_type.lower() == 'histogram':
                fig = px.histogram(df, x=x_axis, title=f'Histogram of {x_axis}')
            elif chart_type.lower() == 'bar':
                fig = px.bar(df[x_axis].value_counts(), title=f'Bar Chart of {x_axis}')
            elif chart_type.lower() == 'scatter' and y_axis:
                fig = px.scatter(df, x=x_axis, y=y_axis, title=f'Scatter Plot: {x_axis} vs {y_axis}')
            elif chart_type.lower() == 'pie':
                fig = px.pie(df, names=x_axis, title=f'Pie Chart of {x_axis}')
            elif chart_type.lower() == 'line' and y_axis:
                fig = px.line(df, x=x_axis, y=y_axis, title=f'Line Chart: {x_axis} vs {y_axis}')
            else:
                return f"Unsupported chart type: {chart_type}"
            
            # Convert plot to HTML and store in session state
            chart_html = fig.to_html(full_html=False)
            st.session_state.last_chart = chart_html
            return f"Successfully generated {chart_type} chart. The chart has been added to the dashboard."
        except Exception as e:
            return f"Error generating chart: {str(e)}"

    # Set up LLM
    llm = ChatGroq(temperature=0, model_name=st.session_state.model_name, groq_api_key=st.session_state.api_key)
    
    # Define tools
    tools = [analyze_data, visualize_data, DuckDuckGoSearchRun()]
    
    # Define prompt
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a expert data analyst AI assistant. Your job is to help users analyze, clean, and visualize their data.
        
        Follow these steps:
        1. First understand what data is available by asking for a summary if needed
        2. Analyze the data to find patterns, anomalies, or insights
        3. Suggest appropriate visualizations based on the data
        4. Help clean the data if needed
        5. Generate reports summarizing your findings
        
        Always be precise and provide actionable insights."""),
        MessagesPlaceholder(variable_name="messages"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ])
    
    # Create agent
    agent = create_openai_tools_agent(llm, tools, prompt)
    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
    
    # Define LangGraph workflow
    workflow = MessageGraph()
    
    # Define nodes
    workflow.add_node("agent", lambda state: agent_executor.invoke(state))
    workflow.add_node("user", lambda state: {"messages": [HumanMessage(content=state["messages"][-1].content)]})
    
    # Define edges
    workflow.add_edge("agent", "user")
    workflow.add_edge("user", "agent")
    
    # Set entry point
    workflow.set_entry_point("agent")
    
    # Compile
    app = workflow.compile()
    
    return app

# CrewAI setup
def setup_crewai_agents():
    # Define agents
    data_cleaner = Agent(
        role='Senior Data Cleaner',
        goal='Clean and preprocess data to ensure high quality for analysis',
        backstory="""You are an expert in data cleaning with years of experience handling messy datasets.
        You know all the tricks to identify and fix missing values, outliers, and inconsistencies.""",
        verbose=True,
        allow_delegation=False
    )
    
    data_analyst = Agent(
        role='Senior Data Analyst',
        goal='Analyze data to extract meaningful insights and patterns',
        backstory="""You are a seasoned data analyst with a keen eye for patterns and trends.
        You specialize in transforming raw data into actionable business insights.""",
        verbose=True,
        allow_delegation=False
    )
    
    visualization_expert = Agent(
        role='Data Visualization Specialist',
        goal='Create compelling visualizations that communicate insights effectively',
        backstory="""You are a design-savvy visualization expert who knows how to make data tell a story.
        You choose the right chart types and styling to maximize understanding.""",
        verbose=True,
        allow_delegation=False
    )
    
    report_writer = Agent(
        role='Technical Report Writer',
        goal='Write comprehensive reports summarizing data analysis findings',
        backstory="""You are a skilled technical writer who can distill complex analysis into clear,
        concise reports that stakeholders can understand and act upon.""",
        verbose=True,
        allow_delegation=True
    )
    
    return data_cleaner, data_analyst, visualization_expert, report_writer

# Streamlit UI
def main():
    st.sidebar.title("AI Data Analyzer")
    
    # API key input
    st.session_state.api_key = st.sidebar.text_input("Enter Groq API Key", type="password")
    st.session_state.model_name = st.sidebar.selectbox(
        "Select Model",
        ["mixtral-8x7b-32768", "llama2-70b-4096"],
        index=0
    )
    
    # Menu options
    with st.sidebar:
        selected = option_menu(
            menu_title="Main Menu",
            options=["Upload Data", "Clean Data", "Visualize Data", "AI Analysis", "Generate Report", "Download Results"],
            icons=["upload", "brush", "bar-chart", "robot", "file-text", "download"],
            menu_icon="cast",
            default_index=0
        )
    
    # File upload
    if selected == "Upload Data":
        st.header("Upload Your Data File")
        uploaded_file = st.file_uploader("Choose a CSV, Excel, or PDF file", type=["csv", "xlsx", "xls", "pdf"])
        
        if uploaded_file is not None:
            with st.spinner("Processing file..."):
                df = process_file(uploaded_file)
                if df is not None:
                    st.success("File successfully uploaded!")
                    st.write("Preview of your data:")
                    st.dataframe(df.head())
                    
                    # Basic info
                    st.subheader("Basic Information")
                    col1, col2 = st.columns(2)
                    col1.metric("Number of Rows", len(df))
                    col2.metric("Number of Columns", len(df.columns))
                    
                    # Data types
                    st.subheader("Data Types")
                    dtype_df = pd.DataFrame(df.dtypes, columns=["Data Type"])
                    st.dataframe(dtype_df)
    
    # Data cleaning
    elif selected == "Clean Data" and st.session_state.df is not None:
        st.header("Data Cleaning")
        st.write("Original Data Preview:")
        st.dataframe(st.session_state.df.head())
        
        if st.button("Clean Data"):
            with st.spinner("Cleaning data..."):
                cleaned_df = clean_data(st.session_state.df)
                st.session_state.processed_df = cleaned_df
                st.success("Data cleaning complete!")
                
                st.write("Cleaned Data Preview:")
                st.dataframe(cleaned_df.head())
                
                # Show changes
                st.subheader("Data Cleaning Summary")
                
                # Missing values
                st.write("Missing Values Handling:")
                for col in st.session_state.df.columns:
                    original_missing = st.session_state.df[col].isna().sum()
                    cleaned_missing = cleaned_df[col].isna().sum()
                    if original_missing > 0:
                        st.write(f"- {col}: {original_missing} missing values → {cleaned_missing} after cleaning")
                
                # Duplicates
                original_duplicates = st.session_state.df.duplicated().sum()
                cleaned_duplicates = cleaned_df.duplicated().sum()
                if original_duplicates > 0:
                    st.write(f"- {original_duplicates} duplicate rows removed")
    
    # Data visualization
    elif selected == "Visualize Data" and st.session_state.processed_df is not None:
        st.header("Data Visualization")
        df = st.session_state.processed_df
        
        # Automatic visualizations
        if st.button("Generate Automatic Visualizations"):
            with st.spinner("Creating visualizations..."):
                charts = create_visualizations(df)
                st.success(f"Generated {len(charts)} visualizations!")
                
                # Display charts
                for chart_name, fig in charts.items():
                    st.plotly_chart(fig, use_container_width=True)
        
        # Custom visualization
        st.subheader("Create Custom Visualization")
        col1, col2 = st.columns(2)
        
        chart_type = col1.selectbox("Select Chart Type", [
            "Histogram", "Bar Chart", "Scatter Plot", 
            "Line Chart", "Pie Chart", "Donut Chart",
            "Area Chart", "Box Plot", "Violin Plot"
        ])
        
        numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        
        if chart_type in ["Histogram", "Bar Chart", "Pie Chart", "Donut Chart"]:
            x_axis = col1.selectbox("Select Column", categorical_cols + numeric_cols)
            if st.button("Generate Chart"):
                try:
                    if chart_type == "Histogram":
                        fig = px.histogram(df, x=x_axis, title=f'Histogram of {x_axis}')
                    elif chart_type == "Bar Chart":
                        fig = px.bar(df[x_axis].value_counts(), title=f'Bar Chart of {x_axis}')
                    elif chart_type in ["Pie Chart", "Donut Chart"]:
                        if chart_type == "Pie Chart":
                            fig = px.pie(df, names=x_axis, title=f'Pie Chart of {x_axis}')
                        else:
                            fig = px.pie(df, names=x_axis, title=f'Donut Chart of {x_axis}', hole=0.4)
                    st.plotly_chart(fig, use_container_width=True)
                except Exception as e:
                    st.error(f"Error generating chart: {str(e)}")
        
        elif chart_type in ["Scatter Plot", "Line Chart"]:
            x_axis = col1.selectbox("Select X-Axis", numeric_cols + categorical_cols)
            y_axis = col2.selectbox("Select Y-Axis", numeric_cols)
            if st.button("Generate Chart"):
                try:
                    if chart_type == "Scatter Plot":
                        fig = px.scatter(df, x=x_axis, y=y_axis, title=f'Scatter Plot: {x_axis} vs {y_axis}')
                    else:
                        fig = px.line(df, x=x_axis, y=y_axis, title=f'Line Chart: {x_axis} vs {y_axis}')
                    st.plotly_chart(fig, use_container_width=True)
                except Exception as e:
                    st.error(f"Error generating chart: {str(e)}")
        
        elif chart_type in ["Area Chart", "Box Plot", "Violin Plot"]:
            y_axis = col1.selectbox("Select Column", numeric_cols)
            if st.button("Generate Chart"):
                try:
                    if chart_type == "Area Chart":
                        fig = px.area(df, y=y_axis, title=f'Area Chart of {y_axis}')
                    elif chart_type == "Box Plot":
                        fig = px.box(df, y=y_axis, title=f'Box Plot of {y_axis}')
                    else:
                        fig = px.violin(df, y=y_axis, title=f'Violin Plot of {y_axis}')
                    st.plotly_chart(fig, use_container_width=True)
                except Exception as e:
                    st.error(f"Error generating chart: {str(e)}")
    
    # AI Analysis
    elif selected == "AI Analysis":
        st.header("AI-Powered Data Analysis")
        
        if st.session_state.api_key == "":
            st.warning("Please enter your Groq API key in the sidebar to enable AI analysis.")
        elif st.session_state.processed_df is None:
            st.warning("Please upload and clean your data first.")
        else:
            # Initialize LangGraph agent
            if 'langgraph_agent' not in st.session_state:
                with st.spinner("Initializing AI agent..."):
                    st.session_state.langgraph_agent = setup_langgraph_agent()
                    st.session_state.crewai_agents = setup_crewai_agents()
                    st.session_state.messages = []
            
            # Display chat messages
            for message in st.session_state.messages:
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])
            
            # Chat input
            if prompt := st.chat_input("Ask about your data..."):
                # Add user message to chat history
                st.session_state.messages.append({"role": "user", "content": prompt})
                with st.chat_message("user"):
                    st.markdown(prompt)
                
                # Get AI response
                with st.chat_message("assistant"):
                    with st.spinner("Analyzing..."):
                        try:
                            # Use LangGraph agent
                            response = st.session_state.langgraph_agent.invoke({
                                "messages": [HumanMessage(content=prompt)]
                            })
                            
                            # Extract the response content
                            if isinstance(response, dict) and 'messages' in response:
                                ai_response = response['messages'][-1].content
                            else:
                                ai_response = str(response)
                            
                            st.markdown(ai_response)
                            st.session_state.messages.append({"role": "assistant", "content": ai_response})
                        except Exception as e:
                            st.error(f"Error in AI analysis: {str(e)}")
    
    # Report generation
    elif selected == "Generate Report" and st.session_state.processed_df is not None:
        st.header("Generate Analysis Report")
        
        if st.button("Generate Full Report"):
            with st.spinner("Generating report..."):
                report = generate_report(st.session_state.df, st.session_state.processed_df)
                st.session_state.report_content = report
                
                # Display report
                st.markdown(report, unsafe_allow_html=True)
                
                # Generate charts if not already done
                if not st.session_state.charts:
                    st.session_state.charts = create_visualizations(st.session_state.processed_df)
    
    # Download results
    elif selected == "Download Results":
        st.header("Download Results")
        
        if st.session_state.processed_df is not None:
            # Download cleaned data
            st.subheader("Download Cleaned Data")
            cleaned_csv = st.session_state.processed_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="Download Cleaned CSV",
                data=cleaned_csv,
                file_name="cleaned_data.csv",
                mime="text/csv"
            )
            
            # Download report
            if st.session_state.report_content:
                st.subheader("Download Report")
                pdf_report = create_pdf_report(st.session_state.report_content, st.session_state.charts)
                st.download_button(
                    label="Download PDF Report",
                    data=pdf_report,
                    file_name="data_analysis_report.pdf",
                    mime="application/pdf"
                )
            else:
                st.warning("Generate a report first before downloading.")
        else:
            st.warning("No processed data available to download. Please upload and clean your data first.")
    
    # Show appropriate messages if no data
    if selected != "Upload Data":
        if st.session_state.df is None:
            st.warning("Please upload a data file first.")
        elif selected not in ["Upload Data", "AI Analysis"] and st.session_state.processed_df is None:
            st.warning("Please clean your data first.")

if __name__ == "__main__":
    main()
