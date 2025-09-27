# app.py - Streamlit Frontend for Size Recommendation System

import streamlit as st
import requests
import json
from PIL import Image
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import io

# Configuration
API_BASE_URL = "http://localhost:8000/api/v1"

# Page config
st.set_page_config(
    page_title="AI Size Recommendation",
    page_icon="👔",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main {
        padding: 2rem;
    }
    .stButton > button {
        width: 100%;
        background-color: #4CAF50;
        color: white;
        border-radius: 5px;
        padding: 0.5rem;
        font-weight: bold;
    }
    .stButton > button:hover {
        background-color: #45a049;
    }
    .recommendation-card {
        background-color: #f0f2f6;
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
        border-left: 5px solid #4CAF50;
    }
    .fit-score-high {
        color: #28a745;
        font-size: 2rem;
        font-weight: bold;
    }
    .fit-score-medium {
        color: #ffc107;
        font-size: 2rem;
        font-weight: bold;
    }
    .fit-score-low {
        color: #dc3545;
        font-size: 2rem;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'measurements' not in st.session_state:
    st.session_state.measurements = None
if 'recommendations' not in st.session_state:
    st.session_state.recommendations = None
if 'tryon_image' not in st.session_state:
    st.session_state.tryon_image = None

# Header
st.title("👔 AI-Powered Size Recommendation System")
st.markdown("### Find your perfect fit using advanced body measurement technology")

# Sidebar
with st.sidebar:
    st.header("⚙️ Settings")
    
    input_method = st.selectbox(
        "Input Method",
        ["Quick Input", "Photo Upload", "Video Upload"]
    )
    
    st.divider()
    
    st.header("📊 User Profile")
    if st.session_state.measurements:
        st.success("✅ Measurements extracted")
        with st.expander("View Measurements"):
            for key, value in st.session_state.measurements.items():
                st.text(f"{key}: {value:.1f} cm")
    else:
        st.info("No measurements yet")
    
    st.divider()
    
    # Brand selection
    st.header("🏷️ Brand Selection")
    selected_brands = st.multiselect(
        "Select Brands",
        ["Zara", "H&M", "Uniqlo", "Nike", "Adidas", "Ralph Lauren", "Tommy Hilfiger"],
        default=["Zara", "H&M"]
    )
    
    category = st.selectbox(
        "Category",
        ["T-Shirt", "Shirt", "Jeans", "Jacket", "Suit"]
    )

# Main content area
tab1, tab2, tab3, tab4 = st.tabs(["📏 Input Measurements", "🎯 Get Recommendations", "👕 Virtual Try-On", "📈 Analytics"])

# Tab 1: Input Measurements
with tab1:
    col1, col2 = st.columns(2)
    
    with col1:
        st.header("Enter Your Measurements")
        
        if input_method == "Quick Input":
            with st.form("quick_input_form"):
                height = st.number_input("Height (cm)", min_value=140, max_value=220, value=175)
                weight = st.number_input("Weight (kg)", min_value=40, max_value=150, value=75)
                age = st.number_input("Age", min_value=15, max_value=100, value=30)
                gender = st.selectbox("Gender", ["Male", "Female", "Other"])
                usual_size = st.selectbox("Usual Size", ["XS", "S", "M", "L", "XL", "XXL"])
                
                if st.form_submit_button("Generate Measurements"):
                    # Estimate measurements based on height/weight
                    estimated_measurements = {
                        "height": height,
                        "weight": weight,
                        "chest": 38 + (weight - 60) * 0.5,
                        "waist": 30 + (weight - 60) * 0.4,
                        "hips": 36 + (weight - 60) * 0.3,
                        "inseam": height * 0.45,
                        "shoulders": 40 + (height - 160) * 0.2,
                        "arm_length": height * 0.36,
                        "torso_length": height * 0.29,
                        "neck": 35 + (weight - 60) * 0.1
                    }
                    st.session_state.measurements = estimated_measurements
                    st.success("✅ Measurements estimated successfully!")
                    st.experimental_rerun()
        
        elif input_method == "Photo Upload":
            st.subheader("Upload Your Photos")
            
            col1_1, col1_2 = st.columns(2)
            with col1_1:
                front_image = st.file_uploader("Front View", type=['jpg', 'jpeg', 'png'])
                if front_image:
                    st.image(front_image, caption="Front View", use_column_width=True)
            
            with col1_2:
                side_image = st.file_uploader("Side View", type=['jpg', 'jpeg', 'png'])
                if side_image:
                    st.image(side_image, caption="Side View", use_column_width=True)
            
            height_photo = st.number_input("Your Height (cm)", min_value=140, max_value=220, value=175)
            
            if st.button("Extract Measurements from Photos"):
                if front_image:
                    with st.spinner("🔄 Analyzing your photos..."):
                        # Simulate API call
                        files = {'file': front_image}
                        data = {'height_cm': height_photo, 'image_type': 'front'}
                        
                        # Mock response for demo
                        mock_response = {
                            "measurements": {
                                "height": height_photo,
                                "chest": 102.5,
                                "waist": 84.2,
                                "hips": 96.8,
                                "inseam": 76.4,
                                "shoulders": 44.6,
                                "arm_length": 62.3,
                                "torso_length": 51.8,
                                "neck": 38.2,
                                "thigh": 58.4,
                                "calf": 37.6,
                                "bicep": 32.1,
                                "forearm": 27.8
                            },
                            "confidence": 0.87,
                            "body_type": "athletic"
                        }
                        
                        st.session_state.measurements = mock_response["measurements"]
                        st.success(f"✅ Measurements extracted with {mock_response['confidence']*100:.0f}% confidence!")
                        st.info(f"Body Type: {mock_response['body_type'].title()}")
                        st.experimental_rerun()
        
        elif input_method == "Video Upload":
            st.subheader("Upload 360° Video")
            video_file = st.file_uploader("Upload your 360° spin video", type=['mp4', 'avi', 'mov'])
            
            if video_file:
                st.video(video_file)
                
                if st.button("Extract Measurements from Video"):
                    with st.spinner("🔄 Processing video..."):
                        # Simulate processing
                        progress_bar = st.progress(0)
                        for i in range(100):
                            progress_bar.progress(i + 1)
                        
                        # Mock measurements
                        st.session_state.measurements = {
                            "height": 175,
                            "chest": 100,
                            "waist": 82,
                            "hips": 95,
                            "inseam": 75,
                            "shoulders": 43,
                            "arm_length": 61,
                            "torso_length": 50,
                            "neck": 37
                        }
                        st.success("✅ Video processed successfully!")
                        st.experimental_rerun()
    
    with col2:
        st.header("Your 3D Body Model")
        
        if st.session_state.measurements:
            # Create 3D visualization using plotly
            fig = go.Figure(data=[go.Scatter3d(
                x=[0, 1, 1, 0, 0, 0.5, 0.5, 0, 0, 1, 1, 0.5, 0.5, 1],
                y=[0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0],
                z=[0, 0, 0, 0, 0, 0.5, 0.5, 0.5, 1, 1, 1, 1, 1.5, 1.5],
                mode='markers+lines',
                marker=dict(size=10, color='lightblue'),
                line=dict(color='blue', width=3),
                text=['Hip', 'Hip', 'Hip', 'Hip', 'Waist', 'Waist', 'Waist', 'Waist', 
                      'Chest', 'Chest', 'Chest', 'Chest', 'Shoulder', 'Shoulder'],
                hovertemplate='%{text}<extra></extra>'
            )])
            
            fig.update_layout(
                title="3D Body Representation",
                scene=dict(
                    xaxis=dict(showgrid=False, visible=False),
                    yaxis=dict(showgrid=False, visible=False),
                    zaxis=dict(showgrid=False, visible=False),
                    camera=dict(eye=dict(x=1.5, y=1.5, z=1.5))
                ),
                height=400,
                showlegend=False
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Measurement summary
            st.subheader("📊 Measurement Summary")
            
            measurements_df = pd.DataFrame([
                {"Measurement": "Chest", "Value (cm)": st.session_state.measurements.get("chest", 0)},
                {"Measurement": "Waist", "Value (cm)": st.session_state.measurements.get("waist", 0)},
                {"Measurement": "Hips", "Value (cm)": st.session_state.measurements.get("hips", 0)},
                {"Measurement": "Shoulders", "Value (cm)": st.session_state.measurements.get("shoulders", 0)},
                {"Measurement": "Inseam", "Value (cm)": st.session_state.measurements.get("inseam", 0)}
            ])
            
            fig2 = px.bar(measurements_df, x="Measurement", y="Value (cm)", 
                         color="Value (cm)", color_continuous_scale="Viridis")
            fig2.update_layout(height=300)
            st.plotly_chart(fig2, use_container_width=True)
        else:
            st.info("📷 Please input your measurements to see your 3D body model")

# Tab 2: Get Recommendations
with tab2:
    st.header("🎯 Size Recommendations")
    
    if st.session_state.measurements:
        col1, col2, col3 = st.columns([2, 2, 1])
        
        with col1:
            st.subheader("Select Product")
            
            # Product selector
            product_brand = st.selectbox("Brand", selected_brands, key="rec_brand")
            product_name = st.text_input("Product Name", "Classic Oxford Shirt")
            product_id = st.text_input("Product ID (optional)", "SKU-12345")
            fit_type = st.selectbox("Fit Type", ["Regular", "Slim", "Relaxed", "Oversized"])
            
            if st.button("Get Size Recommendation", type="primary"):
                with st.spinner("🤔 Analyzing fit..."):
                    # Mock API response
                    recommendation = {
                        "recommended_size": "L",
                        "fit_score": 0.88,
                        "fit_notes": [
                            "Perfect fit across chest",
                            "Slightly snug at waist - consider if you prefer looser fit",
                            "Good length for your torso"
                        ],
                        "alternatives": [
                            {"size": "M", "score": 0.76},
                            {"size": "XL", "score": 0.65}
                        ]
                    }
                    
                    st.session_state.recommendations = recommendation
        
        with col2:
            if st.session_state.recommendations:
                st.subheader("📋 Recommendation Results")
                
                rec = st.session_state.recommendations
                
                # Display main recommendation
                st.markdown(f"""
                <div class="recommendation-card">
                    <h3>Recommended Size: {rec['recommended_size']}</h3>
                    <div class="fit-score-high">Fit Score: {rec['fit_score']*100:.0f}%</div>
                </div>
                """, unsafe_allow_html=True)
                
                # Fit notes
                st.subheader("📝 Fit Details")
                for note in rec['fit_notes']:
                    st.write(f"• {note}")
                
                # Alternative sizes
                st.subheader("🔄 Alternative Sizes")
                for alt in rec['alternatives']:
                    st.write(f"**{alt['size']}**: {alt['score']*100:.0f}% fit score")
        
        with col3:
            if st.session_state.recommendations:
                st.subheader("📊 Fit Visualization")
                
                # Create gauge chart
                fig = go.Figure(go.Indicator(
                    mode = "gauge+number",
                    value = st.session_state.recommendations['fit_score'] * 100,
                    domain = {'x': [0, 1], 'y': [0, 1]},
                    title = {'text': "Fit Score"},
                    gauge = {
                        'axis': {'range': [None, 100]},
                        'bar': {'color': "darkgreen"},
                        'steps': [
                            {'range': [0, 50], 'color': "lightgray"},
                            {'range': [50, 80], 'color': "yellow"},
                            {'range': [80, 100], 'color': "lightgreen"}
                        ],
                        'threshold': {
                            'line': {'color': "red", 'width': 4},
                            'thickness': 0.75,
                            'value': 85
                        }
                    }
                ))
                fig.update_layout(height=250)
                st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("⚠️ Please input your measurements first in the 'Input Measurements' tab")

# Tab 3: Virtual Try-On
with tab3:
    st.header("👕 Virtual Try-On")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Upload Images")
        
        your_photo = st.file_uploader("Your Photo", type=['jpg', 'jpeg', 'png'], key="tryon_person")
        garment_photo = st.file_uploader("Garment Photo", type=['jpg', 'jpeg', 'png'], key="tryon_garment")
        
        if your_photo and garment_photo:
            col1_1, col1_2 = st.columns(2)
            with col1_1:
                st.image(your_photo, caption="Your Photo", use_column_width=True)
            with col1_2:
                st.image(garment_photo, caption="Garment", use_column_width=True)
            
            if st.button("Generate Virtual Try-On", type="primary"):
                with st.spinner("🎨 Generating try-on image... This may take a moment"):
                    progress = st.progress(0)
                    for i in range(100):
                        progress.progress(i + 1)
                    
                    # For demo, just show the garment image as result
                    st.session_state.tryon_image = garment_photo
                    st.success("✅ Try-on generated successfully!")
    
    with col2:
        st.subheader("Try-On Result")
        
        if st.session_state.tryon_image:
            st.image(st.session_state.tryon_image, caption="Virtual Try-On Result", use_column_width=True)
            
            # Download button
            st.download_button(
                label="📥 Download Try-On Image",
                data=st.session_state.tryon_image.getvalue() if hasattr(st.session_state.tryon_image, 'getvalue') else st.session_state.tryon_image,
                file_name="virtual_tryon.jpg",
                mime="image/jpeg"
            )
            
            # Feedback
            st.subheader("Rate This Try-On")
            rating = st.slider("How realistic does this look?", 1, 5, 3)
            if st.button("Submit Feedback"):
                st.success("Thank you for your feedback!")
        else:
            st.info("🖼️ Generate a virtual try-on to see the result here")

# Tab 4: Analytics
with tab4:
    st.header("📈 Your Fit Analytics")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Size Distribution Across Brands")
        
        # Mock data
        size_data = pd.DataFrame({
            'Brand': ['Zara', 'H&M', 'Uniqlo', 'Nike', 'Adidas'],
            'Your Size': ['L', 'XL', 'L', 'M', 'L'],
            'Fit Score': [88, 84, 91, 79, 85]
        })
        
        fig = px.bar(size_data, x='Brand', y='Fit Score', color='Your Size',
                     title="Your Fit Scores Across Brands")
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("Fit Preferences Over Time")
        
        # Mock trend data
        dates = pd.date_range(start='2024-01-01', periods=10, freq='M')
        trend_data = pd.DataFrame({
            'Date': dates,
            'Preferred Fit': np.random.choice(['Regular', 'Slim', 'Relaxed'], 10),
            'Satisfaction': np.random.uniform(3.5, 5, 10)
        })
        
        fig = px.line(trend_data, x='Date', y='Satisfaction', 
                     title="Your Satisfaction with Recommendations")
        st.plotly_chart(fig, use_container_width=True)
    
    # Statistics
    st.subheader("📊 Your Statistics")
    
    metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
    
    with metric_col1:
        st.metric("Total Recommendations", "47", "+5 this month")
    
    with metric_col2:
        st.metric("Average Fit Score", "86%", "+2%")
    
    with metric_col3:
        st.metric("Returns Avoided", "12", "Saved $480")
    
    with metric_col4:
        st.metric("Favorite Brand", "Uniqlo", "91% fit score")
    
    # Recent activity
    st.subheader("📅 Recent Activity")
    
    activity_data = pd.DataFrame({
        'Date': ['2024-10-15', '2024-10-12', '2024-10-08', '2024-10-01'],
        'Brand': ['Zara', 'H&M', 'Nike', 'Uniqlo'],
        'Product': ['Oxford Shirt', 'Slim Jeans', 'Running Tee', 'Casual Blazer'],
        'Recommended Size': ['L', 'XL', 'M', 'L'],
        'Fit Score': ['88%', '84%', '79%', '91%']
    })
    
    st.dataframe(activity_data, use_container_width=True)

# Footer
st.divider()
st.markdown("""
<div style="text-align: center; color: gray; padding: 20px;">
    <p>Powered by Advanced AI • MediaPipe • SMPL-X • Stable Diffusion</p>
    <p>© 2024 AI Size Recommendation System</p>
</div>
""", unsafe_allow_html=True)

# Run instructions
if __name__ == "__main__":
    st.sidebar.markdown("""
    ---
    ### 🚀 Quick Start
    1. Input your measurements
    2. Select a brand & product
    3. Get instant size recommendations
    4. Try virtual try-on
    
    ### 📱 API Status
    """)
    
    # Check API status
    try:
        response = requests.get(f"{API_BASE_URL}/health", timeout=2)
        if response.status_code == 200:
            st.sidebar.success("API: Online ✅")
        else:
            st.sidebar.error("API: Offline ❌")
    except:
        st.sidebar.warning("API: Not Connected ⚠️")
    
    st.sidebar.markdown("""
    ### 📚 Resources
    - [API Documentation](http://localhost:8000/docs)
    - [GitHub Repository](#)
    - [Report Issues](#)
    """)