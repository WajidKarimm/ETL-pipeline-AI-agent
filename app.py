"""
Streamlit Web Interface for ETL Pipeline
A user-friendly interface for configuring and running ETL jobs.
Enhanced with robust error handling and better user feedback.
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import tempfile
import os
import sys
import json
from datetime import datetime
import time
import traceback

# Add src to path
sys.path.append('src')

from src.extractors.csv_extractor import CSVExtractor
from src.transformers.clean_transformer import CleanTransformer
from src.logger import get_logger

# Page configuration
st.set_page_config(
    page_title="ETL Pipeline AI Agent",
    page_icon="🔄",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .success-box {
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        border-radius: 5px;
        padding: 1rem;
        margin: 1rem 0;
    }
    .error-box {
        background-color: #f8d7da;
        border: 1px solid #f5c6cb;
        border-radius: 5px;
        padding: 1rem;
        margin: 1rem 0;
    }
    .info-box {
        background-color: #d1ecf1;
        border: 1px solid #bee5eb;
        border-radius: 5px;
        padding: 1rem;
        margin: 1rem 0;
    }
    .warning-box {
        background-color: #fff3cd;
        border: 1px solid #ffeaa7;
        border-radius: 5px;
        padding: 1rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

def main():
    """Main Streamlit application."""
    
    # Header
    st.markdown('<h1 class="main-header">🔄 ETL Pipeline AI Agent</h1>', unsafe_allow_html=True)
    st.markdown("### Transform your data with AI-powered ETL pipeline")
    
    # Sidebar for configuration
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # Data Source Selection
        st.subheader("📥 Data Source")
        data_source = st.selectbox(
            "Select Data Source",
            ["CSV File", "API Endpoint", "Database"],
            help="Choose where to extract data from"
        )
        
        # Data Destination Selection
        st.subheader("📤 Data Destination")
        data_destination = st.selectbox(
            "Select Destination",
            ["Preview Only", "CSV Export", "Database", "Cloud Storage"],
            help="Choose where to load the transformed data"
        )
        
        # Transformation Options
        st.subheader("🔄 Transformations")
        remove_nulls = st.checkbox("Remove null values", value=True)
        rename_columns = st.checkbox("Rename columns", value=False)
        map_fields = st.checkbox("Map field values", value=False)
        handle_duplicates = st.checkbox("Remove duplicates", value=True)
        
        # Advanced Options
        with st.expander("🔧 Advanced Options"):
            batch_size = st.number_input("Batch Size", min_value=100, max_value=10000, value=1000, step=100)
            max_retries = st.number_input("Max Retries", min_value=1, max_value=10, value=3)
            log_level = st.selectbox("Log Level", ["INFO", "DEBUG", "WARNING", "ERROR"])
            
            # CSV specific options
            if data_source == "CSV File":
                st.subheader("📄 CSV Options")
                encoding = st.selectbox("Encoding", ["auto", "utf-8", "latin-1", "cp1252"], help="File encoding")
                separator = st.selectbox("Separator", ["auto", ",", ";", "\t", "|"], help="CSV separator")
    
    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.header("📊 Data Processing")
        
        # File upload section
        if data_source == "CSV File":
            uploaded_file = st.file_uploader(
                "Choose a CSV file",
                type=['csv', 'txt'],
                help="Upload your CSV file for processing"
            )
            
            if uploaded_file is not None:
                # Display file info
                file_details = {
                    "Filename": uploaded_file.name,
                    "File size": f"{uploaded_file.size / 1024:.2f} KB",
                    "File type": uploaded_file.type
                }
                st.json(file_details)
                
                # Show file preview
                with st.expander("👀 Preview Uploaded File"):
                    try:
                        # Try to read first few lines
                        content = uploaded_file.read().decode('utf-8', errors='ignore')
                        lines = content.split('\n')[:10]
                        st.code('\n'.join(lines), language='text')
                        uploaded_file.seek(0)  # Reset file pointer
                    except Exception as e:
                        st.warning(f"Could not preview file: {str(e)}")
                
                # Process the file
                if st.button("🚀 Process Data", type="primary"):
                    process_data(uploaded_file, data_destination, remove_nulls, rename_columns, map_fields, handle_duplicates)
        
        elif data_source == "API Endpoint":
            st.info("API integration coming soon! For now, please use CSV file upload.")
        
        elif data_source == "Database":
            st.info("Database integration coming soon! For now, please use CSV file upload.")
    
    with col2:
        st.header("📈 Statistics")
        
        # Placeholder for statistics
        if 'processed_data' not in st.session_state:
            st.info("Upload and process data to see statistics here.")
        else:
            display_statistics(st.session_state.processed_data)

def process_data(uploaded_file, destination, remove_nulls, rename_columns, map_fields, handle_duplicates):
    """Process the uploaded data through the ETL pipeline."""
    
    try:
        # Create progress bar and status
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Step 1: Extract
        status_text.text("📥 Extracting data...")
        progress_bar.progress(10)
        
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix='.csv') as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_file_path = tmp_file.name
        
        # Extract data with error handling
        try:
            extractor_config = {
                'csv_options': {
                    'encoding': 'utf-8',
                    'on_bad_lines': 'skip'
                }
            }
            extractor = CSVExtractor(extractor_config)
            raw_data = extractor.extract(tmp_file_path)
            
            if raw_data.empty:
                st.error("❌ No data could be extracted from the file. Please check the file format.")
                return
                
        except Exception as e:
            st.error(f"❌ Error extracting data: {str(e)}")
            st.info("💡 Try uploading a different CSV file or check the file format.")
            return
        finally:
            # Clean up temp file
            try:
                os.unlink(tmp_file_path)
            except:
                pass
        
        progress_bar.progress(30)
        
        # Step 2: Transform
        status_text.text("🔄 Transforming data...")
        progress_bar.progress(50)
        
        # Prepare transformation config
        transform_config = {
            'dropna_axis': 0,
            'dropna_how': 'any' if remove_nulls else 'none',
            'handle_duplicates': handle_duplicates
        }
        
        # Add column renaming if requested
        if rename_columns:
            # Create a simple renaming scheme
            rename_map = {}
            for col in raw_data.columns:
                new_name = col.lower().replace(' ', '_').replace('-', '_')
                if new_name != col:
                    rename_map[col] = new_name
            transform_config['rename_map'] = rename_map
        
        # Add field mapping if requested
        if map_fields:
            # Create a simple mapping for categorical columns
            field_map = {}
            for col in raw_data.select_dtypes(include=['object']).columns:
                unique_values = raw_data[col].dropna().unique()
                if len(unique_values) <= 10:  # Only map if reasonable number of values
                    mapping = {val: idx for idx, val in enumerate(unique_values)}
                    field_map[col] = mapping
            transform_config['field_map'] = field_map
        
        # Apply transformations with error handling
        try:
            transformer = CleanTransformer(transform_config)
            transformed_data = transformer.transform(raw_data)
            
            if transformed_data.empty:
                st.warning("⚠️ All data was removed during transformation. Check your settings.")
                return
                
        except Exception as e:
            st.error(f"❌ Error transforming data: {str(e)}")
            st.info("💡 Try adjusting the transformation settings.")
            return
        
        # Step 3: Load/Preview
        status_text.text("📤 Preparing results...")
        progress_bar.progress(80)
        
        # Store results in session state
        st.session_state.raw_data = raw_data
        st.session_state.processed_data = transformed_data
        st.session_state.transform_config = transform_config
        
        # Step 4: Complete
        status_text.text("✅ Processing complete!")
        progress_bar.progress(100)
        time.sleep(1)
        
        # Display results
        display_results(raw_data, transformed_data, destination)
        
    except Exception as e:
        st.error(f"❌ Unexpected error: {str(e)}")
        with st.expander("🔍 Error Details"):
            st.code(traceback.format_exc())

def display_results(raw_data, transformed_data, destination):
    """Display the processing results."""
    
    st.success("🎉 Data processing completed successfully!")
    
    # Data comparison
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📥 Original Data")
        st.write(f"**Rows:** {len(raw_data)} | **Columns:** {len(raw_data.columns)}")
        st.dataframe(raw_data.head(), use_container_width=True)
    
    with col2:
        st.subheader("📤 Transformed Data")
        st.write(f"**Rows:** {len(transformed_data)} | **Columns:** {len(transformed_data.columns)}")
        st.dataframe(transformed_data.head(), use_container_width=True)
    
    # Data quality metrics
    st.subheader("📊 Data Quality Metrics")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Original Rows", len(raw_data))
    
    with col2:
        st.metric("Transformed Rows", len(transformed_data))
    
    with col3:
        nulls_removed = len(raw_data) - len(transformed_data)
        st.metric("Rows Removed", nulls_removed)
    
    with col4:
        if len(raw_data) > 0:
            reduction_pct = (nulls_removed / len(raw_data)) * 100
            st.metric("Reduction %", f"{reduction_pct:.1f}%")
    
    # Data visualization
    if len(transformed_data) > 0:
        st.subheader("📈 Data Visualization")
        
        # Select column for visualization
        numeric_cols = transformed_data.select_dtypes(include=['number']).columns
        if len(numeric_cols) > 0:
            selected_col = st.selectbox("Select column for visualization", numeric_cols)
            
            if selected_col:
                # Create histogram
                fig = px.histogram(transformed_data, x=selected_col, title=f"Distribution of {selected_col}")
                st.plotly_chart(fig, use_container_width=True)
        
        # Correlation matrix for numeric columns
        if len(numeric_cols) > 1:
            st.subheader("🔗 Correlation Matrix")
            corr_matrix = transformed_data[numeric_cols].corr()
            fig = px.imshow(corr_matrix, title="Correlation Matrix")
            st.plotly_chart(fig, use_container_width=True)
    
    # Download options
    st.subheader("💾 Download Results")
    
    if destination == "CSV Export":
        csv = transformed_data.to_csv(index=False)
        st.download_button(
            label="📥 Download Transformed CSV",
            data=csv,
            file_name=f"transformed_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )
    
    # Show transformation details
    with st.expander("🔍 View Transformation Details"):
        st.json(st.session_state.transform_config)

def display_statistics(data):
    """Display statistics about the processed data."""
    
    if data is None:
        return
    
    st.metric("Total Rows", len(data))
    st.metric("Total Columns", len(data.columns))
    
    # Data types
    st.subheader("📋 Data Types")
    dtype_counts = data.dtypes.value_counts()
    fig = px.pie(values=dtype_counts.values, names=dtype_counts.index, title="Data Types Distribution")
    st.plotly_chart(fig, use_container_width=True)
    
    # Missing values
    st.subheader("❓ Missing Values")
    missing_data = data.isnull().sum()
    if missing_data.sum() > 0:
        fig = px.bar(x=missing_data.index, y=missing_data.values, title="Missing Values per Column")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.success("✅ No missing values found!")

if __name__ == "__main__":
    main() 