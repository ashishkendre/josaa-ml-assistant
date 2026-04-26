"""
JoSAA ML-Powered Choice Filling Assistant
Integrated Streamlit Dashboard with ML Models
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

# Page config
st.set_page_config(
    page_title="JoSAA ML-Powered Assistant",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .stApp {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    .main .block-container {
        background: white;
        padding: 2rem;
        border-radius: 20px;
        box-shadow: 0 20px 60px rgba(0,0,0,0.3);
    }
    h1 { color: #667eea; }
    h2, h3 { color: #555; }
    .stButton>button {
        background: linear-gradient(135deg, #667eea, #764ba2);
        color: white;
        font-weight: 600;
        border-radius: 10px;
        padding: 0.5rem 2rem;
        border: none;
        width: 100%;
    }
    .stButton>button:hover {
        transform: translateY(-2px);
    }
    .metric-card {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        padding: 1.5rem;
        border-radius: 12px;
        text-align: center;
        margin: 0.5rem 0;
    }
    .recommendation-card {
        background: white;
        border: 2px solid #e0e0e0;
        border-radius: 12px;
        padding: 1.5rem;
        margin-bottom: 1rem;
        transition: all 0.3s;
    }
    .recommendation-card:hover {
        border-color: #667eea;
        box-shadow: 0 5px 15px rgba(102, 126, 234, 0.2);
    }
    .badge-high { background: #d4edda; color: #155724; padding: 0.25rem 0.75rem; border-radius: 20px; font-size: 0.85rem; font-weight: 600; }
    .badge-moderate { background: #fff3cd; color: #856404; padding: 0.25rem 0.75rem; border-radius: 20px; font-size: 0.85rem; font-weight: 600; }
    .badge-low { background: #f8d7da; color: #721c24; padding: 0.25rem 0.75rem; border-radius: 20px; font-size: 0.85rem; font-weight: 600; }
    .nirf-badge { background: #667eea; color: white; padding: 0.25rem 0.75rem; border-radius: 20px; font-size: 0.85rem; font-weight: 600; }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_models():
    """Load trained ML models"""
    try:
        cutoff_model = joblib.load('models/cutoff_prediction_model.pkl')
        admission_model = joblib.load('models/admission_probability_model.pkl')
        return cutoff_model, admission_model
    except Exception as e:
        st.error(f"Error loading models: {e}")
        return None, None


@st.cache_data
def load_data():
    """Load processed data"""
    try:
        df = pd.read_csv('processed_data/josaa_ml_ready.csv')
        return df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None


@st.cache_data
def load_predictions_2025():
    """Load 2025 predictions"""
    try:
        df = pd.read_csv('results/predictions_2025.csv')
        return df
    except:
        return None


def get_admission_probability(student_rank, college_data, admission_model_data):
    """Calculate admission probability using ML model"""
    model = admission_model_data['model']
    feature_cols = admission_model_data['feature_columns']
    
    # Prepare features
    features = {
        'student_rank': student_rank,
        'prev_year_closing_rank': college_data.get('Closing Rank', 0),
        'prev_year_opening_rank': college_data.get('Opening Rank', 0),
        'rank_vs_prev_closing': student_rank - college_data.get('Closing Rank', 0),
        'rank_ratio_prev': student_rank / college_data.get('Closing Rank', 1) if college_data.get('Closing Rank', 0) > 0 else 1.0,
        'prev_closing_opening_ratio': college_data.get('Closing Rank', 1) / college_data.get('Opening Rank', 1) if college_data.get('Opening Rank', 0) > 0 else 1.0,
        'year': 2025,
        'round': college_data.get('Round', 1),
        'institute_code': college_data.get('Institute_Code', 0),
        'institute_type_code': college_data.get('Institute_Type_Code', 0),
        'branch_code': college_data.get('Academic Program Name_Code', 0),
        'branch_category_code': college_data.get('Branch_Category_Code', 0),
        'quota_code': college_data.get('Quota_Code', 0),
        'seat_type_code': college_data.get('Seat Type_Code', 0),
        'gender_code': college_data.get('Gender_Code', 0),
        'nirf_rank': college_data.get('NIRF_Rank', 999),
        'round_progression': college_data.get('Round_Progression', 0.5)
    }
    
    X = pd.DataFrame([features])[feature_cols]
    probability = model.predict_proba(X)[0][1]
    
    return probability


def get_recommendations_ml(df, predictions_df, user_prefs, admission_model_data):
    """Get ML-powered recommendations"""
    
    # Filter by exam type
    exam_type = user_prefs['exam_type']
    
    if exam_type == 'advanced':
        filtered = df[df['Institute'].str.contains('Indian Institute of Technology', case=False, na=False)].copy()
    else:
        filtered = df[~df['Institute'].str.contains('Indian Institute of Technology', case=False, na=False)].copy()
    
    # Filter to most recent year (2024)
    filtered = filtered[filtered['Year'] == 2024]
    
    # Filter by category
    category = user_prefs['category']
    filtered = filtered[filtered['Seat Type'].str.contains(category, case=False, na=False)]
    
    # Filter by gender
    gender = user_prefs['gender']
    filtered = filtered[
        (filtered['Gender'].str.contains(gender, case=False, na=False)) |
        (filtered['Gender'].str.contains('Neutral', case=False, na=False))
    ]
    
    if len(filtered) == 0:
        return None
    
    user_rank = user_prefs['rank']
    
    # Calculate admission probability for each option using ML
    probabilities = []
    for idx, row in filtered.iterrows():
        prob = get_admission_probability(user_rank, row.to_dict(), admission_model_data)
        probabilities.append(prob)
    
    filtered['Admission_Probability'] = probabilities
    
    # Add 2025 predictions if available
    if predictions_df is not None:
        merge_cols = ['Institute', 'Academic Program Name', 'Quota', 'Seat Type', 'Gender', 'Round']
        pred_subset = predictions_df[merge_cols + ['Predicted_Closing_Rank_2025', 'Change_Percent']]
        filtered = filtered.merge(pred_subset, on=merge_cols, how='left')
    
    # Filter: only show colleges with meaningful probability
    filtered = filtered[filtered['Admission_Probability'] > 0.05]
    
    # Calculate composite score
    filtered['Quality_Score'] = filtered['NIRF_Rank'].apply(lambda x: max(0, 40 - x * 0.5) if x != 999 else 5)
    filtered['Branch_Quality'] = filtered['Branch_Category'].map({
        'CS/IT': 30, 'ECE/EE': 25, 'Mechanical': 20,
        'Aerospace': 22, 'Chemical': 18, 'Civil': 15,
        'Material': 17, 'Bio': 12, 'Other': 10
    }).fillna(10)
    
    filtered['Composite_Score'] = (
        filtered['Quality_Score'] +
        filtered['Branch_Quality'] +
        filtered['Admission_Probability'] * 30
    )
    
    # Sort by composite score
    filtered = filtered.sort_values('Composite_Score', ascending=False)
    
    # Take top N
    max_choices = user_prefs.get('max_choices', 50)
    return filtered.head(max_choices)


def categorize_probability(prob):
    """Categorize probability into High/Moderate/Low"""
    if prob >= 0.7:
        return "High", "badge-high"
    elif prob >= 0.4:
        return "Moderate", "badge-moderate"
    else:
        return "Low", "badge-low"


# Main App
st.title("🎓 JoSAA ML-Powered Choice Filling Assistant")
st.markdown("### Smart recommendations powered by Machine Learning")

# Load models and data
with st.spinner("Loading ML models..."):
    cutoff_model_data, admission_model_data = load_models()
    df = load_data()
    predictions_df = load_predictions_2025()

if cutoff_model_data is None or admission_model_data is None:
    st.error("⚠️ Could not load ML models. Make sure to run training scripts first!")
    st.stop()

if df is None:
    st.error("⚠️ Could not load data. Make sure preprocessing is complete!")
    st.stop()

# Show model info
with st.expander("🤖 ML Models Information"):
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"""
        **Cutoff Prediction Model**
        - Algorithm: {cutoff_model_data['model_name']}
        - Features: {len(cutoff_model_data['feature_columns'])}
        - Trained on: 173,699 records
        """)
    with col2:
        st.markdown(f"""
        **Admission Probability Model**
        - Algorithm: {admission_model_data['model_name']}
        - Features: {len(admission_model_data['feature_columns'])}
        - Trained on: 1.26M records
        """)

# Sidebar
with st.sidebar:
    st.header("📊 Your Details")
    
    exam_type_display = st.selectbox(
        "Exam Type *",
        ["", "JEE Advanced (IITs only)", "JEE Main (NITs/IIITs/GFTIs)"]
    )
    
    rank = st.number_input(
        "Your Rank *",
        min_value=1, max_value=1000000, value=5000
    )
    
    category = st.selectbox(
        "Category *",
        ["", "OPEN", "EWS", "OBC-NCL", "SC", "ST"]
    )
    
    gender = st.selectbox(
        "Gender *",
        ["", "Gender-Neutral", "Female"]
    )
    
    max_choices = st.slider("Number of Choices", 10, 100, 30)
    
    st.markdown("---")
    
    st.markdown("""
    ### 💡 ML Features:
    - **Predicted 2025 cutoffs**
    - **Exact admission probability**
    - **Smart recommendations**
    - **Historical trends**
    """)
    
    submit_button = st.button("🔍 Get Smart Recommendations")

# Tabs
tab1, tab2, tab3 = st.tabs(["🎯 Recommendations", "📊 2025 Predictions", "📈 Historical Trends"])

with tab1:
    if submit_button:
        if not exam_type_display or not category or not gender:
            st.error("⚠️ Please fill all required fields")
        else:
            exam_type = 'advanced' if 'Advanced' in exam_type_display else 'mains'
            
            user_prefs = {
                'exam_type': exam_type,
                'rank': rank,
                'category': category,
                'gender': gender,
                'max_choices': max_choices
            }
            
            with st.spinner("🤖 ML models analyzing your options..."):
                recommendations = get_recommendations_ml(df, predictions_df, user_prefs, admission_model_data)
            
            if recommendations is None or len(recommendations) == 0:
                st.warning("⚠️ No matching colleges found. Try different criteria.")
            else:
                # Summary metrics
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Total Options", len(recommendations))
                with col2:
                    high_prob = (recommendations['Admission_Probability'] >= 0.7).sum()
                    st.metric("High Probability", high_prob)
                with col3:
                    moderate_prob = ((recommendations['Admission_Probability'] >= 0.4) & 
                                     (recommendations['Admission_Probability'] < 0.7)).sum()
                    st.metric("Moderate Probability", moderate_prob)
                with col4:
                    avg_nirf = recommendations[recommendations['NIRF_Rank'] != 999]['NIRF_Rank'].mean()
                    st.metric("Avg NIRF Rank", f"{avg_nirf:.0f}" if pd.notna(avg_nirf) else "N/A")
                
                st.markdown("---")
                
                # Display recommendations
                st.subheader(f"🎯 Top {len(recommendations)} ML-Powered Recommendations")
                
                for idx, (_, rec) in enumerate(recommendations.iterrows(), 1):
                    prob = rec['Admission_Probability']
                    prob_label, prob_class = categorize_probability(prob)
                    
                    nirf = int(rec['NIRF_Rank']) if rec['NIRF_Rank'] != 999 else "Unranked"
                    nirf_display = f"NIRF #{nirf}" if nirf != "Unranked" else "Unranked"
                    
                    # Get 2025 prediction if available
                    pred_2025 = ""
                    if 'Predicted_Closing_Rank_2025' in rec and pd.notna(rec.get('Predicted_Closing_Rank_2025')):
                        change = rec.get('Change_Percent', 0)
                        arrow = "📈" if change > 0 else "📉" if change < 0 else "➡️"
                        pred_2025 = f"<br><strong>2025 Predicted:</strong> {int(rec['Predicted_Closing_Rank_2025'])} {arrow} ({change:+.1f}%)"
                    
                    st.markdown(f"""
                    <div class="recommendation-card">
                        <h3>#{idx} - {rec['Institute']}</h3>
                        <h4 style="color: #666;">{rec['Academic Program Name']}</h4>
                        <p>
                            <strong>2024 Closing Rank:</strong> {int(rec['Closing Rank'])} |
                            <strong>Round:</strong> {int(rec['Round'])} |
                            <strong>Quota:</strong> {rec['Quota']}
                        </p>
                        <p>
                            <span class="nirf-badge">{nirf_display}</span>
                            <span class="{prob_class}">{prob_label} ({prob*100:.1f}%)</span>
                            <strong>Composite Score:</strong> {rec['Composite_Score']:.1f}/100
                        </p>
                        {pred_2025}
                    </div>
                    """, unsafe_allow_html=True)
    else:
        st.info("👈 Fill the form on the left and click 'Get Smart Recommendations'")

with tab2:
    st.subheader("🔮 Predicted 2025 Cutoffs")
    st.markdown("Based on our ML model trained on 5 years of historical data")
    
    if predictions_df is not None:
        # Filter options
        col1, col2 = st.columns(2)
        with col1:
            filter_institute = st.selectbox(
                "Select Institute",
                ["All"] + sorted(predictions_df['Institute'].unique().tolist())
            )
        with col2:
            filter_seat = st.selectbox(
                "Select Seat Type",
                ["All"] + sorted(predictions_df['Seat Type'].unique().tolist())
            )
        
        filtered_pred = predictions_df.copy()
        if filter_institute != "All":
            filtered_pred = filtered_pred[filtered_pred['Institute'] == filter_institute]
        if filter_seat != "All":
            filtered_pred = filtered_pred[filtered_pred['Seat Type'] == filter_seat]
        
        # Display
        if len(filtered_pred) > 0:
            # Summary
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Predictions", len(filtered_pred))
            with col2:
                avg_change = filtered_pred['Change_Percent'].mean()
                st.metric("Avg Change %", f"{avg_change:+.1f}%")
            with col3:
                increasing = (filtered_pred['Change_Percent'] > 0).sum()
                st.metric("Cutoffs Increasing", f"{increasing}/{len(filtered_pred)}")
            
            # Show predictions table
            display_df = filtered_pred[['Institute', 'Academic Program Name', 'Round', 
                                       'Closing_Rank_2024', 'Predicted_Closing_Rank_2025', 
                                       'Change_Percent']].copy()
            display_df.columns = ['Institute', 'Program', 'Round', '2024 Closing', '2025 Predicted', 'Change %']
            
            st.dataframe(display_df.head(50), use_container_width=True)
            
            # Visualization
            st.subheader("📊 Top 20 Predicted Changes")
            top_changes = filtered_pred.nlargest(20, 'Change_Percent')
            fig = px.bar(
                top_changes,
                x='Change_Percent',
                y='Institute',
                color='Change_Percent',
                color_continuous_scale='RdYlGn_r',
                title='Predicted % Change in Closing Ranks (2024 → 2025)'
            )
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("No predictions available")

with tab3:
    st.subheader("📈 Historical Trends Analysis")
    
    if df is not None:
        # Select institute
        institutes = sorted(df['Institute'].unique().tolist())
        selected_institute = st.selectbox("Select Institute", institutes)
        
        institute_data = df[df['Institute'] == selected_institute]
        
        if len(institute_data) > 0:
            # Select branch
            branches = sorted(institute_data['Academic Program Name'].unique().tolist())
            selected_branch = st.selectbox("Select Branch", branches)
            
            branch_data = institute_data[institute_data['Academic Program Name'] == selected_branch]
            
            if len(branch_data) > 0:
                # Group by year
                yearly_data = branch_data.groupby('Year').agg({
                    'Opening Rank': 'min',
                    'Closing Rank': 'max'
                }).reset_index()
                
                # Plot
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=yearly_data['Year'],
                    y=yearly_data['Opening Rank'],
                    name='Opening Rank',
                    mode='lines+markers',
                    line=dict(color='green', width=3)
                ))
                fig.add_trace(go.Scatter(
                    x=yearly_data['Year'],
                    y=yearly_data['Closing Rank'],
                    name='Closing Rank',
                    mode='lines+markers',
                    line=dict(color='red', width=3)
                ))
                
                fig.update_layout(
                    title=f'Cutoff Trends: {selected_institute}<br>{selected_branch}',
                    xaxis_title='Year',
                    yaxis_title='Rank',
                    hovermode='x unified',
                    height=500
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Show data table
                st.subheader("📋 Year-wise Data")
                st.dataframe(yearly_data, use_container_width=True)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666;'>
    <p>🤖 Powered by Machine Learning | Built for JEE Aspirants</p>
    <p>Models: XGBoost (Cutoff Prediction & Admission Probability)</p>
</div>
""", unsafe_allow_html=True)
