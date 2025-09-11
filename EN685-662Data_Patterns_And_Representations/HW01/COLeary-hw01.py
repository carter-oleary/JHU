import streamlit as st
import pandas as pd
import numpy as np
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import io

# Page configuration
st.set_page_config(
    page_title="AutoML Explorer",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better visual design
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #1f4e79;
        text-align: center;
        margin-bottom: 2rem;
    }
    .step-header {
        background: linear-gradient(90deg, #4CAF50, #45a049);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
        font-size: 1.5rem;
        font-weight: bold;
        margin: 1rem 0;
    }
    .metric-card {
        background: #f8f9fa;
        border: 2px solid #e9ecef;
        border-radius: 10px;
        padding: 1rem;
        margin: 0.5rem 0;
    }
    .best-model {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
        font-weight: bold;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

def initialize_session_state():
    """Initialize session state variables"""
    if 'step' not in st.session_state:
        st.session_state.step = 1
    if 'data' not in st.session_state:
        st.session_state.data = None
    if 'models_trained' not in st.session_state:
        st.session_state.models_trained = False
    if 'results' not in st.session_state:
        st.session_state.results = {}

def step_1_welcome():
    """Step 1: Welcome screen"""
    st.markdown('<div class="main-header">🤖 AutoML Explorer</div>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown("""
        <div style="text-align: center; background: #f0f2f6; padding: 2rem; margin-bottom: 1rem; border-radius: 15px;">
            <h2 style="color: black">Gain valuable insights from your data</h2>
            <p style="font-size: 1.1rem; color: #666;">
                Upload your CSV dataset and let our automated machine learning pipeline 
                analyze it using multiple models to find the best predictions.
            </p>
            <div style="margin: 2rem 0;">
                <div style="background: navy; padding: 1rem; border-radius: 10px; margin: 1rem 0;">
                    <strong>📁 Accepted Format:</strong> .csv files
                </div>
                <div style="background: green; padding: 1rem; border-radius: 10px; margin: 1rem 0;">
                    <strong>🤖 Models:</strong> Linear Regression, Random Forest, Neural Network
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        if st.button("Continue", key="continue_btn", use_container_width=True):
            st.session_state.step = 2
            st.rerun()

def step_2_upload():
    """Step 2: Dataset upload"""
    st.markdown('<div class="step-header">📊 Step 2: Dataset Upload</div>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown("""
        <div style="text-align: center; padding: 2rem;">
            <h3>Upload your dataset to get started</h3>
            <p>Drag and drop your CSV file or browse to select it</p>
        </div>
        """, unsafe_allow_html=True)
        
        uploaded_file = st.file_uploader(
            "Choose a CSV file",
            type=['csv'],
            help="Upload a CSV file with your dataset. The file should have column headers."
        )
        
        if uploaded_file is not None:
            try:
                # Read the CSV file
                df = pd.read_csv(uploaded_file)
                st.session_state.data = df
                
                # Display success message and data preview
                st.success("✅ File uploaded successfully!")
                
                st.markdown("### Data Preview")
                st.dataframe(df.head(), use_container_width=True)
                
                st.markdown("### Dataset Information")
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Rows", df.shape[0])
                with col2:
                    st.metric("Columns", df.shape[1])
                with col3:
                    numeric_cols = len(df.select_dtypes(include=[np.number]).columns)
                    st.metric("Numerical Columns", numeric_cols)
                with col4:
                    st.metric("Missing Values", df.isnull().sum().sum())
                
                # Check if we have enough numerical columns
                if numeric_cols < 2:
                    st.error("⚠️ Need at least 2 numerical columns for regression analysis!")
                    st.info("Please upload a dataset with more numerical data.")
                else:
                    st.success(f"✅ Ready for analysis with {numeric_cols} numerical columns")
                    
                    if st.button("Next: Configure Models", use_container_width=True):
                        st.session_state.step = 3
                        st.rerun()
                    
            except Exception as e:
                st.error(f"❌ Error reading file: {str(e)}")
                st.info("Please make sure your file is a valid CSV format.")

def step_3_model_setup():
    """Step 3: Model configuration"""
    st.markdown('<div class="step-header">⚙️ Step 3: Model Setup</div>', unsafe_allow_html=True)
    
    if st.session_state.data is None:
        st.error("Please upload a dataset first.")
        return
    
    df = st.session_state.data
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("### Select Target Variable")
        numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
        
        if not numeric_columns:
            st.error("No numeric columns found for regression analysis.")
            return
        
        target_column = st.selectbox(
            "Choose the target variable to predict:",
            numeric_columns,
            help="Select the column you want to predict"
        )
    
    with col2:
        st.markdown("### Model Selection")
        st.info("All three models will be trained and compared:")
        models_info = {
            "🔵 Linear Regression": "Fast, interpretable, works well for linear relationships",
            "🌳 Random Forest": "Robust, handles non-linear patterns, good for most datasets", 
            "🧠 Support Vector Machine": "Better with high dimensional data, handles non-linear patterns"
        }
        
        for model, description in models_info.items():
            st.markdown(f"**{model}**: {description}")
    
    st.markdown("### Feature Information")
    feature_columns = [col for col in df.columns if col != target_column]
    
    col1, col2 = st.columns(2)
    with col1:
        st.write("**Features (X):**")
        st.write(feature_columns)
    with col2:
        st.write("**Target (y):**")
        st.write(target_column)
    
    if st.button("🚀 Run Models", use_container_width=True):
        st.session_state.target_column = target_column
        st.session_state.step = 4
        st.rerun()

def step_4_execution():
    """Step 4: Model execution with progress tracking"""
    st.markdown('<div class="step-header">🔄 Step 4: Training Models</div>', unsafe_allow_html=True)
    
    if st.session_state.data is None:
        st.error("Please upload a dataset first.")
        return
    
    df = st.session_state.data
    target_column = st.session_state.target_column
    
    # Prepare data
    with st.spinner("Preparing data..."):
        # Handle categorical variables
        df_processed = df.copy()
        
        # Encode categorical variables
        le_dict = {}
        for col in df_processed.columns:
            if df_processed[col].dtype == 'object':
                le = LabelEncoder()
                df_processed[col] = le.fit_transform(df_processed[col].astype(str))
                le_dict[col] = le
        
        # Remove rows with missing target values
        df_processed = df_processed.dropna()
        
        X = df_processed.drop(columns=[target_column])
        y = df_processed[target_column]
        
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
    
    # Progress bar and model training
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    models = {
        "Linear Regression": LogisticRegression(),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
        "Support Vector Machine": svm.SVC(kernel='rbf', C=1.0)
    }
    
    results = {}
    
    for i, (name, model) in enumerate(models.items()):
        status_text.text(f"Training {name}...")
        progress_bar.progress((i + 1) / len(models))
        
        # Train model
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
            
        # Calculate metrics
        acc = accuracy_score(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, y_pred)
        
        results[name] = {
            'model': model,
            'predictions': y_pred,
            'accuracy': acc,
            'mse': mse,
            'rmse': rmse,
            'r2': r2
        }
    
    status_text.text("✅ All models trained successfully!")
    st.session_state.results = results
    st.session_state.y_test = y_test
    st.session_state.models_trained = True
    
    if st.button("View Results", use_container_width=True):
        st.session_state.step = 5
        st.rerun()

def step_5_results():
    """Step 5: Display results and comparisons"""
    st.markdown('<div class="step-header">📊 Step 5: Model Results</div>', unsafe_allow_html=True)
    
    if not st.session_state.models_trained:
        st.error("Please train the models first.")
        return
    
    results = st.session_state.results
    
    # Find best model based on R² score
    best_model_name = max(results.keys(), key=lambda x: results[x]['accuracy'])
    
    st.markdown(f'<div class="best-model">🏆 Best Performing Model: {best_model_name}</div>', 
                unsafe_allow_html=True)
    
    # Metrics comparison
    st.markdown("### Model Performance Comparison")
    
    # Create comparison chart
    models = list(results.keys())
    acc_scores = [results[model]['accuracy'] for model in models]  
    r2_scores = [results[model]['r2'] for model in models]
    rmse_scores = [results[model]['rmse'] for model in models]
    
    fig = make_subplots(
        rows=1, cols=3,
        subplot_titles=('Accuracy (Higher is better)', 'R² Score (Higher is Better)', 'RMSE (Lower is Better)'),
        specs=[[{"secondary_y": False}, {"secondary_y": False}, {"secondary_y": False}]]
    )
    
    fig.add_trace(
        go.Bar(name='Accuracy', x=models, y=acc_scores, marker_color='lightgreen'),
        row=1, col=1
    )
    
    # R² scores
    fig.add_trace(
        go.Bar(name='R² Score', x=models, y=r2_scores, marker_color='lightblue'),
        row=1, col=2
    )
    
    # RMSE scores
    fig.add_trace(
        go.Bar(name='RMSE', x=models, y=rmse_scores, marker_color='lightcoral'),
        row=1, col=3
    )
    
    fig.update_layout(height=400, showlegend=False)
    st.plotly_chart(fig, use_container_width=True)
    
    # Detailed metrics table
    st.markdown("### Detailed Performance Metrics")
    
    metrics_df = pd.DataFrame({
        'Model': models,
        'Accuracy (%)': [f"{results[model]['accuracy']*100:.2f}%" for model in models],
        'R² Score': [f"{results[model]['r2']:.4f}" for model in models],
        'RMSE': [f"{results[model]['rmse']:.4f}" for model in models],
        'MSE': [f"{results[model]['mse']:.4f}" for model in models]
    })
    
    st.dataframe(metrics_df, use_container_width=True)
    
    # Prediction vs Actual scatter plot for best model
    st.markdown(f"### Prediction Analysis - {best_model_name}")
    
    y_test = st.session_state.y_test
    best_predictions = results[best_model_name]['predictions']
    
    fig = px.scatter(
        x=y_test, y=best_predictions,
        labels={'x': 'Actual Values', 'y': 'Predicted Values'},
        title=f'{best_model_name} - Predicted vs Actual'
    )
    
    # Add perfect prediction line
    min_val = min(min(y_test), min(best_predictions))
    max_val = max(max(y_test), max(best_predictions))
    fig.add_trace(go.Scatter(
        x=[min_val, max_val], y=[min_val, max_val],
        mode='lines', name='Perfect Prediction',
        line=dict(dash='dash', color='red')
    ))
    
    st.plotly_chart(fig, use_container_width=True)
    
    if st.button("Export Results", use_container_width=True):
        st.session_state.step = 6
        st.rerun()

def step_6_export():
    """Step 6: Export and restart options"""
    st.markdown('<div class="step-header">📥 Step 6: Export Results</div>', unsafe_allow_html=True)
    
    if not st.session_state.models_trained:
        st.error("No results to export.")
        return
    
    results = st.session_state.results
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Download Results")
        
        # Prepare results for download
        export_data = []
        for model_name, model_results in results.items():
            export_data.append({
                'Model': model_name,
                'Accuracy (%)': model_results['accuracy'],
                'R² Score': model_results['r2'],
                'MSE': model_results['mse']
            })
        
        results_df = pd.DataFrame(export_data)
        
        # Convert to CSV
        csv_buffer = io.StringIO()
        results_df.to_csv(csv_buffer, index=False)
        csv_data = csv_buffer.getvalue()
        
        st.download_button(
            label="📥 Download Results (CSV)",
            data=csv_data,
            file_name="automl_results.csv",
            mime="text/csv",
            use_container_width=True
        )
    
    with col2:
        st.markdown("### Start New Analysis")
        st.info("Ready to analyze another dataset?")
        
        if st.button("🔄 Try Another Dataset", use_container_width=True):
            # Reset session state
            for key in ['data', 'models_trained', 'results', 'target_column', 'y_test']:
                if key in st.session_state:
                    del st.session_state[key]
            st.session_state.step = 1
            st.rerun()

def main():
    """Main application flow"""
    initialize_session_state()
    
    # Sidebar navigation
    with st.sidebar:
        st.markdown("## Navigation")
        steps = {
            1: "🏠 Welcome",
            2: "📊 Upload Data", 
            3: "⚙️ Setup Models",
            4: "🔄 Training",
            5: "📈 Results",
            6: "📥 Export"
        }
        
        current_step = st.session_state.step
        
        for step_num, step_name in steps.items():
            if step_num <= current_step:
                if step_num == current_step:
                    st.markdown(f"**➤ {step_name}**")
                else:
                    if st.button(step_name, key=f"nav_{step_num}"):
                        if step_num <= current_step:  # Only allow going to completed steps
                            st.session_state.step = step_num
                            st.rerun()
            else:
                st.markdown(f"⏳ {step_name}")
        
        # Progress indicator
        st.markdown("---")
        st.markdown("### Progress")
        progress = (current_step - 1) / 5
        st.progress(progress)
        st.markdown(f"Step {current_step} of 6")
    
    # Main content based on current step
    if st.session_state.step == 1:
        step_1_welcome()
    elif st.session_state.step == 2:
        step_2_upload()
    elif st.session_state.step == 3:
        step_3_model_setup()
    elif st.session_state.step == 4:
        step_4_execution()
    elif st.session_state.step == 5:
        step_5_results()
    elif st.session_state.step == 6:
        step_6_export()

if __name__ == "__main__":
    main()