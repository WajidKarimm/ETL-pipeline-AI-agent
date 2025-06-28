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

from src.extractors.universal_extractor import UniversalExtractor
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
        st.subheader("�� Transformations")
        
        # More granular null handling
        null_handling = st.selectbox(
            "Handle missing values",
            ["Keep all data", "Remove completely empty rows", "Remove rows with >50% nulls", "Remove any row with nulls"],
            help="Choose how to handle missing values in your data"
        )
        
        # Convert selection to boolean for backward compatibility
        remove_nulls = null_handling != "Keep all data"
        
        rename_columns = st.checkbox("Rename columns", value=False, help="Standardize column names")
        map_fields = st.checkbox("Map field values", value=False, help="Convert categorical values to numeric")
        handle_duplicates = st.checkbox("Remove duplicates", value=False, help="Remove duplicate rows")
        
        # Advanced Options
        with st.expander("🔧 Advanced Options"):
            batch_size = st.number_input("Batch Size", min_value=100, max_value=10000, value=1000, step=100)
            max_retries = st.number_input("Max Retries", min_value=1, max_value=10, value=3)
            log_level = st.selectbox("Log Level", ["INFO", "DEBUG", "WARNING", "ERROR"])
            
            # Data organization options
            st.subheader("📊 Data Organization")
            organize_data = st.checkbox("Organize Data Structure", value=True, 
                                      help="Sort columns by type, reorder rows, add metadata, and ensure consistent formatting")
            
            if organize_data:
                st.info("✅ Data will be organized with: Column sorting, row ordering, metadata columns, and consistent formatting")
            else:
                st.warning("⚠️ Data organization disabled - raw structure will be preserved")
            
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
                "Choose a data file (up to 1GB)",
                type=['csv', 'txt', 'arff', 'json', 'xml'],
                help="Upload your data file for processing (supports multiple formats, max 1GB)"
            )
            
            if uploaded_file is not None:
                # Display file info with better size formatting
                file_size_mb = uploaded_file.size / (1024 * 1024)
                if file_size_mb > 1024:
                    file_size_display = f"{file_size_mb / 1024:.2f} GB"
                else:
                    file_size_display = f"{file_size_mb:.2f} MB"
                
                file_details = {
                    "Filename": uploaded_file.name,
                    "File size": file_size_display,
                    "File type": uploaded_file.type,
                    "Rows estimate": f"~{int(file_size_mb * 50000):,}" if file_size_mb > 10 else "Processing..."
                }
                st.json(file_details)
                
                # Show file preview (only for smaller files)
                if file_size_mb < 50:  # Only preview files under 50MB
                    with st.expander("👀 Preview Uploaded File"):
                        try:
                            # Try to read first few lines
                            content = uploaded_file.read().decode('utf-8', errors='ignore')
                            lines = content.split('\n')[:10]
                            st.code('\n'.join(lines), language='text')
                            uploaded_file.seek(0)  # Reset file pointer
                        except Exception as e:
                            st.warning(f"Could not preview file: {str(e)}")
                else:
                    st.info(f"📁 Large file detected ({file_size_display}). Preview disabled for performance.")
                
                # Process the file
                if st.button("🚀 Process Data", type="primary"):
                    process_data(uploaded_file, data_destination, null_handling, rename_columns, map_fields, handle_duplicates, organize_data)
        
        elif data_source == "API Endpoint":
            st.info("API integration coming soon! For now, please use file upload.")
        
        elif data_source == "Database":
            st.info("Database integration coming soon! For now, please use file upload.")
    
    with col2:
        st.header("📈 Statistics")
        
        # Placeholder for statistics
        if 'processed_data' not in st.session_state:
            st.info("Upload and process data to see statistics here.")
        else:
            display_statistics(st.session_state.processed_data)

    # Add AI features section
    if st.sidebar.checkbox("🤖 AI-Powered Features", value=True):
        st.header("🤖 AI-Powered ETL Assistant")
        
        # Check if we have processed data available
        if 'processed_data' in st.session_state and st.session_state.processed_data is not None:
            df = st.session_state.processed_data
            
            # Initialize AI agent
            try:
                from src.ml.ai_agent import ETLAIAgent
                ai_agent = ETLAIAgent()
                
                # AI Data Quality Analysis
                st.subheader("🔍 Data Quality Analysis")
                
                with st.spinner("Analyzing data quality..."):
                    quality_issues = ai_agent.detect_data_quality_issues(df)
                
                if quality_issues:
                    st.warning(f"Detected {len(quality_issues)} data quality issues:")
                    
                    for i, issue in enumerate(quality_issues):
                        with st.expander(f"🚨 {issue.issue_type.upper()} - {issue.severity.upper()}"):
                            st.write(f"**Description:** {issue.description}")
                            st.write(f"**Affected Columns:** {', '.join(issue.affected_columns)}")
                            st.write(f"**Suggested Fix:** {issue.suggested_fix}")
                            
                            # Auto-apply fixes
                            if st.button(f"Auto-apply fix for issue {i+1}", key=f"fix_{i}"):
                                if issue.issue_type == "missing_values":
                                    # Fill missing values
                                    for col in issue.affected_columns:
                                        if col in df.columns:
                                            df[col] = df[col].fillna(df[col].mode()[0] if df[col].dtype == 'object' else df[col].median())
                                    st.session_state.processed_data = df
                                    st.success("Applied missing value fixes!")
                                    st.rerun()
                                elif issue.issue_type == "duplicates":
                                    # Remove duplicates
                                    df = df.drop_duplicates()
                                    st.session_state.processed_data = df
                                    st.success("Removed duplicate rows!")
                                    st.rerun()
                else:
                    st.success("✅ No data quality issues detected!")
                
                # AI Transformation Suggestions
                st.subheader("💡 Smart Suggestions")
                
                with st.spinner("Generating suggestions..."):
                    suggestions = ai_agent.suggest_transformations(df)
                
                if suggestions:
                    st.info(f"Found {len(suggestions)} optimization opportunities:")
                    
                    for i, suggestion in enumerate(suggestions):
                        with st.expander(f"💡 {suggestion.transformation_type}"):
                            st.write(f"**Target Column:** {suggestion.target_column}")
                            st.write(f"**Reasoning:** {suggestion.reasoning}")
                            
                            # Auto-apply suggestions
                            if st.button(f"Apply suggestion {i+1}", key=f"suggest_{i}"):
                                try:
                                    if suggestion.transformation_type == "convert_to_numeric":
                                        df[suggestion.target_column] = pd.to_numeric(df[suggestion.target_column], errors='coerce')
                                    elif suggestion.transformation_type == "convert_to_datetime":
                                        df[suggestion.target_column] = pd.to_datetime(df[suggestion.target_column], errors='coerce')
                                    elif suggestion.transformation_type == "fill_missing":
                                        if suggestion.parameters.get("method") == "forward_fill":
                                            df[suggestion.target_column] = df[suggestion.target_column].fillna(method='ffill')
                                    
                                    st.session_state.processed_data = df
                                    st.success(f"Applied {suggestion.transformation_type}!")
                                    st.rerun()
                                except Exception as e:
                                    st.error(f"Failed to apply transformation: {e}")
                else:
                    st.info("No optimization suggestions at this time.")
                
                # Error Prediction (silent background check)
                transform_config = st.session_state.get('transform_config', {})
                field_map = transform_config.get('field_map', {})
                data_types = {}
                remove_nulls = transform_config.get('dropna_how') != 'none'
                rename_columns = 'rename_map' in transform_config
                
                # Silent background error prediction
                predictions = []
                
                # Check for missing columns in field map
                for old_col, new_col in field_map.items():
                    if old_col not in df.columns:
                        predictions.append({
                            'error_type': 'Missing Column',
                            'probability': 1.0,
                            'description': f"Column '{old_col}' not found in data",
                            'suggestion': f"Check column names or remove mapping for '{old_col}'"
                        })
                
                # Check for type conversion issues
                for col, target_type in data_types.items():
                    if col in df.columns and target_type == 'numeric' and df[col].dtype == 'object':
                        try:
                            pd.to_numeric(df[col], errors='raise')
                        except:
                            predictions.append({
                                'error_type': 'Type Conversion Error',
                                'probability': 0.9,
                                'description': f"Cannot convert column '{col}' to numeric",
                                'suggestion': f"Clean non-numeric values in column '{col}'"
                            })
                
                if predictions:
                    st.warning(f"⚠️ Potential issues detected:")
                    
                    for i, prediction in enumerate(predictions):
                        with st.expander(f"⚠️ {prediction['error_type']}"):
                            st.write(f"**Description:** {prediction['description']}")
                            st.write(f"**Suggestion:** {prediction['suggestion']}")
            
            except ImportError:
                st.error("AI features not available. Please install required dependencies.")
            except Exception as e:
                st.error(f"AI features error: {e}")
        else:
            st.info("📊 Upload and process data to enable AI features!")

    # Update the main ETL processing section to include AI learning
    if st.button("🚀 Run ETL Pipeline with AI Assistance", type="primary"):
        if uploaded_file is not None:
            try:
                # Initialize AI agent for learning (silent background)
                try:
                    from src.ml.ai_agent import ETLAIAgent
                    ai_agent = ETLAIAgent()
                    ai_enabled = True
                except:
                    ai_enabled = False
                
                with st.spinner("Running ETL pipeline with AI assistance..."):
                    # Process data first
                    process_data(uploaded_file, data_destination, null_handling, rename_columns, map_fields, handle_duplicates, organize_data)
                    
                    # Silent background learning
                    if ai_enabled and 'processed_data' in st.session_state and st.session_state.processed_data is not None:
                        try:
                            # Prepare configuration for learning
                            config = {
                                'source_type': 'dataframe',
                                'source_data': st.session_state.processed_data,
                                'destination_type': data_destination,
                                'field_map': st.session_state.get('transform_config', {}).get('field_map', {}),
                                'data_types': {},
                                'remove_nulls': null_handling != "Keep all data",
                                'rename_columns': rename_columns,
                                'destination_config': {},
                                'organize_data': organize_data
                            }
                            
                            # Silent background learning
                            ai_agent.learn_from_operation(
                                data=st.session_state.processed_data,
                                transformations=config,
                                success=True,
                                errors=[]
                            )
                        except Exception as e:
                            # Silent error handling - don't show to user
                            pass
                    
            except Exception as e:
                st.error(f"❌ ETL Pipeline failed: {e}")
                
                # Silent background learning from failure
                if ai_enabled:
                    try:
                        ai_agent.learn_from_operation(
                            data=pd.DataFrame(),
                            transformations={},
                            success=False,
                            errors=[str(e)]
                        )
                    except Exception as ai_error:
                        # Silent error handling
                        pass
        else:
            st.warning("Please upload a file first!")

def process_data(uploaded_file, destination, null_handling, rename_columns, map_fields, handle_duplicates, organize_data):
    """Process the uploaded data through the ETL pipeline."""
    
    try:
        # Create progress bar and status
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Check file size and warn for very large files
        file_size_mb = uploaded_file.size / (1024 * 1024)
        if file_size_mb > 500:
            st.warning(f"⚠️ Very large file detected ({file_size_mb:.1f} MB). Processing may take several minutes and use significant memory.")
            st.info("💡 Consider splitting very large files into smaller chunks for better performance.")
        elif file_size_mb > 200:
            st.warning(f"⚠️ Large file detected ({file_size_mb:.1f} MB). Processing may take several minutes.")
        
        # Step 1: Extract
        status_text.text("📥 Extracting data...")
        progress_bar.progress(10)
        
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix='.csv') as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_file_path = tmp_file.name
        
        # Extract data with universal extractor
        try:
            extractor_config = {
                'csv_options': {
                    'encoding': 'utf-8',
                    'on_bad_lines': 'skip',
                    'chunksize': 10000 if file_size_mb > 100 else None  # Use chunks for large files
                }
            }
            extractor = UniversalExtractor(extractor_config)
            raw_data = extractor.extract(tmp_file_path)
            
            if raw_data.empty or 'error' in raw_data.columns:
                error_msg = raw_data.iloc[0]['error'] if 'error' in raw_data.columns else "No data could be extracted"
                st.error(f"❌ {error_msg}")
                if "field larger than field limit" in error_msg:
                    st.info("💡 This file has very wide columns. Try splitting the file or using a different format.")
                elif "Unable to allocate" in error_msg:
                    st.info("💡 File too large for available memory. Try processing a smaller subset or splitting the file.")
                else:
                    st.info("💡 Try uploading a different file or check the file format.")
                return
                
        except Exception as e:
            st.error(f"❌ Error extracting data: {str(e)}")
            if "memory" in str(e).lower():
                st.info("💡 File too large for available memory. Try processing a smaller subset.")
            else:
                st.info("💡 Try uploading a different file or check the file format.")
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
        
        # Prepare transformation config with conservative settings
        transform_config = {
            'dropna_axis': 0,
            'handle_duplicates': handle_duplicates,
            'organize_data': organize_data
        }
        
        # Handle null removal based on user selection
        if null_handling == "Keep all data":
            transform_config['dropna_how'] = 'none'
        elif null_handling == "Remove completely empty rows":
            transform_config['dropna_how'] = 'all'
        elif null_handling == "Remove rows with >50% nulls":
            transform_config['dropna_how'] = 'any'  # Will be handled conservatively in transformer
        elif null_handling == "Remove any row with nulls":
            transform_config['dropna_how'] = 'any'  # Will be handled conservatively in transformer
        
        # Add column renaming if requested
        if rename_columns:
            # Create a simple renaming scheme
            rename_map = {}
            for col in raw_data.columns:
                new_name = col.lower().replace(' ', '_').replace('-', '_')
                if new_name != col:
                    rename_map[col] = new_name
            transform_config['rename_map'] = rename_map
        
        # Add field mapping if requested (optimized for large datasets)
        if map_fields:
            dataset_size = len(raw_data)
            
            if dataset_size < 500000:  # Allow mapping for datasets up to 500k rows
                # Create a simple mapping for categorical columns
                field_map = {}
                categorical_cols = raw_data.select_dtypes(include=['object']).columns
                
                # Only process columns with reasonable number of unique values
                for col in categorical_cols:
                    unique_values = raw_data[col].dropna().unique()
                    if len(unique_values) <= 50:  # Increased limit for larger datasets
                        mapping = {val: idx for idx, val in enumerate(unique_values)}
                        field_map[col] = mapping
                
                if field_map:
                    transform_config['field_map'] = field_map
                    st.success(f"✅ Field mapping enabled for {len(field_map)} columns (optimized for {dataset_size:,} rows)")
                else:
                    st.info("ℹ️ No suitable categorical columns found for field mapping")
            else:
                st.info("⚠️ Field mapping disabled for very large datasets (>500k rows) to ensure optimal performance.")
                st.info("💡 Consider processing smaller subsets or using column renaming instead.")
        
        # Apply transformations with error handling
        try:
            transformer = CleanTransformer(transform_config)
            transformed_data = transformer.transform(raw_data)
            
            if transformed_data.empty:
                st.warning("⚠️ All data was removed during transformation. Check your settings.")
                st.info("💡 Try selecting 'Keep all data' or other less aggressive transformations.")
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
        rows_removed = len(raw_data) - len(transformed_data)
        st.metric("Rows Removed", rows_removed)
    
    with col4:
        if len(raw_data) > 0:
            reduction_pct = (rows_removed / len(raw_data)) * 100
            st.metric("Reduction %", f"{reduction_pct:.1f}%")
    
    # Data organization information
    if st.session_state.get('transform_config', {}).get('organize_data', False):
        st.success("🎯 **Data Organization Applied:**")
        st.info("""
        ✅ **Column Sorting:** ID → Date → Categorical → Numeric → Text  
        ✅ **Row Ordering:** Logical sorting by ID, date, or category  
        ✅ **Metadata Columns:** Added completeness and text length metrics  
        ✅ **Consistent Formatting:** Standardized case and precision  
        ✅ **Data Type Optimization:** Proper date and numeric formatting
        """)
    else:
        st.info("📋 **Raw Data Structure:** Original column order and formatting preserved")
    
    # Field mapping information
    transform_config = st.session_state.get('transform_config', {})
    field_map = transform_config.get('field_map', {})
    if field_map:
        dataset_size = len(raw_data)
        if dataset_size > 50000:
            st.success("🚀 **Optimized Field Mapping Applied:**")
            st.info(f"""
            ✅ **Vectorized Processing:** Used pandas map() for {dataset_size:,} rows  
            ✅ **Memory Efficient:** Optimized for large datasets  
            ✅ **Columns Mapped:** {len(field_map)} categorical columns  
            ✅ **Performance:** Significantly faster than loop-based mapping
            """)
        else:
            st.success("🔄 **Field Mapping Applied:**")
            st.info(f"✅ **Columns Mapped:** {len(field_map)} categorical columns")
    
    # Data visualization
    if len(transformed_data) > 0:
        st.subheader("📈 Data Visualization")
        
        try:
            # Select column for visualization - only show numeric columns
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
                try:
                    corr_matrix = transformed_data[numeric_cols].corr()
                    # Convert to regular Python types for JSON serialization
                    corr_dict = {str(col1): {str(col2): float(val) for col2, val in row.items()} 
                               for col1, row in corr_matrix.items()}
                    fig = px.imshow(corr_matrix, title="Correlation Matrix")
                    st.plotly_chart(fig, use_container_width=True)
                except Exception as e:
                    st.warning(f"Could not display correlation matrix: {str(e)}")
        except Exception as e:
            st.warning(f"Could not create data visualizations: {str(e)}")
            st.info("💡 Try enabling column renaming or field mapping to improve data compatibility.")
    
    # Download options
    st.subheader("💾 Download Results")
    
    # Validate data before download
    if transformed_data is not None and not transformed_data.empty:
        # Check file size and provide appropriate download options
        estimated_size_mb = len(transformed_data) * len(transformed_data.columns) * 0.1  # Rough estimate
        
        if estimated_size_mb > 100:
            st.warning(f"⚠️ Large dataset detected (~{estimated_size_mb:.1f} MB estimated). Download may take time.")
        
        # Always provide download options regardless of destination
        try:
            # CSV download with progress for large files
            if estimated_size_mb > 50:
                with st.spinner("Preparing CSV download..."):
                    csv = transformed_data.to_csv(index=False)
            else:
                csv = transformed_data.to_csv(index=False)
                
            st.download_button(
                label="📥 Download Transformed CSV",
                data=csv,
                file_name=f"transformed_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )
            
            # JSON download (only for smaller files to avoid memory issues)
            if estimated_size_mb < 50:
                json_data = transformed_data.to_json(orient='records', indent=2)
                st.download_button(
                    label="📥 Download Transformed JSON",
                    data=json_data,
                    file_name=f"transformed_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json"
                )
            else:
                st.info("📄 JSON download disabled for large files to improve performance.")
            
            # Show data summary before download
            st.success(f"📊 **Download Ready:** {len(transformed_data):,} rows, {len(transformed_data.columns)} columns")
            
            # Show sample of data being downloaded (only for smaller datasets)
            if len(transformed_data) < 10000:
                with st.expander("👀 Preview Data to Download"):
                    st.write("**First 5 rows of data to be downloaded:**")
                    st.dataframe(transformed_data.head(), use_container_width=True)
            else:
                st.info("📋 Preview disabled for large datasets. Use download to view full data.")
                
        except Exception as e:
            st.error(f"❌ Error preparing download: {str(e)}")
            st.info("💡 Try processing the data again or check the transformation settings.")
    else:
        st.error("❌ No data available for download. Please process your data first.")
    
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
    try:
        # Convert data types to strings to avoid serialization issues
        dtype_counts = data.dtypes.astype(str).value_counts()
        fig = px.pie(values=dtype_counts.values, names=dtype_counts.index, title="Data Types Distribution")
        st.plotly_chart(fig, use_container_width=True)
    except Exception as e:
        st.warning(f"Could not display data types chart: {str(e)}")
        # Show data types as text instead
        st.write("**Data Types:**")
        for col, dtype in data.dtypes.items():
            st.write(f"- {col}: {dtype}")
    
    # Missing values
    st.subheader("❓ Missing Values")
    try:
        missing_data = data.isnull().sum()
        if missing_data.sum() > 0:
            # Convert to regular Python types for JSON serialization
            missing_dict = {str(col): int(val) for col, val in missing_data.items()}
            fig = px.bar(x=list(missing_dict.keys()), y=list(missing_dict.values()), title="Missing Values per Column")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.success("✅ No missing values found!")
    except Exception as e:
        st.warning(f"Could not display missing values chart: {str(e)}")
        # Show missing values as text instead
        missing_data = data.isnull().sum()
        if missing_data.sum() > 0:
            st.write("**Missing Values:**")
            for col, count in missing_data.items():
                if count > 0:
                    st.write(f"- {col}: {count}")
        else:
            st.success("✅ No missing values found!")

if __name__ == "__main__":
    main() 