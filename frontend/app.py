import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import os
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(page_title="Customer Segmentation", layout="wide")

# Title
st.title("Customer Segmentation Dashboard")

# Initialize session state
if 'df' not in st.session_state:
    st.session_state.df = None
if 'df_processed' not in st.session_state:
    st.session_state.df_processed = None
if 'kmeans_model' not in st.session_state:
    st.session_state.kmeans_model = None
if 'scaler' not in st.session_state:
    st.session_state.scaler = None
if 'features' not in st.session_state:
    st.session_state.features = None
if 'optimal_k' not in st.session_state:
    st.session_state.optimal_k = 5
if 'model_trained' not in st.session_state:
    st.session_state.model_trained = False

# Sidebar
with st.sidebar:
    st.header("Data & Model Configuration")
    
    # Data options
    st.subheader("1. Data Source")
    data_option = st.radio("Select data source:", 
                          ["Generate Sample Data", "Upload CSV File"])
    
    if data_option == "Generate Sample Data":
        if st.button("Generate Sample Data"):
            np.random.seed(42)
            n_customers = 200
            
            # Create sample data
            data = {
                'CustomerID': list(range(1, n_customers + 1)),
                'Gender': np.random.choice(['Male', 'Female'], n_customers),
                'Age': np.random.randint(18, 70, n_customers),
                'AnnualIncome_k': np.random.randint(15, 80, n_customers),
                'SpendingScore': np.random.randint(1, 100, n_customers)
            }
            
            df = pd.DataFrame(data)
            
            # Create natural clusters
            mask1 = (df['Age'] < 30) & (df['SpendingScore'] > 70)
            df.loc[mask1, 'AnnualIncome_k'] = np.random.randint(40, 80, mask1.sum())
            
            mask2 = (df['Age'] >= 30) & (df['Age'] < 50) & (df['AnnualIncome_k'] > 50)
            df.loc[mask2, 'SpendingScore'] = np.random.randint(40, 60, mask2.sum())
            
            mask3 = (df['Age'] >= 50)
            df.loc[mask3, 'SpendingScore'] = np.random.randint(10, 40, mask3.sum())
            
            st.session_state.df = df
            st.session_state.model_trained = False
            st.success(f"✅ Generated {n_customers} sample customers")
    
    else:
        uploaded_file = st.file_uploader("Upload CSV file", type=['csv'])
        if uploaded_file is not None:
            try:
                df = pd.read_csv(uploaded_file)
                st.session_state.df = df
                st.session_state.model_trained = False
                st.success(f"✅ Loaded {len(df)} records")
            except Exception as e:
                st.error(f"Error: {e}")
    
    st.divider()
    
    # Model configuration
    st.subheader("2. Model Configuration")
    
    # Number of clusters
    st.session_state.optimal_k = st.slider(
        "Number of clusters (k):",
        min_value=2,
        max_value=10,
        value=5,
        help="Select number of clusters for K-Means"
    )
    
    st.divider()
    
    # Model actions
    st.subheader("3. Model Actions")
    
    col1, col2 = st.columns(2)
    
    with col1:
        train_btn = st.button("🚀 Train Model", type="primary", use_container_width=True)
    
    with col2:
        save_btn = st.button("💾 Save Model", use_container_width=True)
    
    # Load model button
    load_btn = st.button("📂 Load Model", use_container_width=True)

# Main content
if st.session_state.df is not None:
    # Display dataset info
    st.header("Dataset Overview")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Customers", len(st.session_state.df))
    with col2:
        if 'Gender' in st.session_state.df.columns:
            male_count = len(st.session_state.df[st.session_state.df['Gender'] == 'Male'])
            st.metric("Male Customers", male_count)
        else:
            st.metric("Gender Column", "Not Found")
    with col3:
        if 'Gender' in st.session_state.df.columns:
            female_count = len(st.session_state.df[st.session_state.df['Gender'] == 'Female'])
            st.metric("Female Customers", female_count)
        else:
            st.metric("N/A", "N/A")
    with col4:
        if 'Age' in st.session_state.df.columns:
            avg_age = st.session_state.df['Age'].mean()
            st.metric("Average Age", f"{avg_age:.1f}")
        else:
            st.metric("Age Column", "Not Found")
    
    # Data preview tabs
    tab1, tab2, tab3 = st.tabs(["📊 Data Preview", "📈 Statistics", "🔍 Distributions"])
    
    with tab1:
        st.dataframe(st.session_state.df.head(20), use_container_width=True)
    
    with tab2:
        st.dataframe(st.session_state.df.describe(), use_container_width=True)
    
    with tab3:
        if st.session_state.df is not None:
            fig, axes = plt.subplots(2, 2, figsize=(12, 8))
            
            # Age distribution
            if 'Age' in st.session_state.df.columns:
                axes[0, 0].hist(st.session_state.df['Age'], bins=20, alpha=0.7, color='blue')
                axes[0, 0].set_title('Age Distribution')
                axes[0, 0].set_xlabel('Age')
                axes[0, 0].set_ylabel('Frequency')
            else:
                axes[0, 0].text(0.5, 0.5, 'Age column not found', ha='center', va='center')
                axes[0, 0].set_title('Age Distribution')
            
            # Income distribution
            if 'AnnualIncome_k' in st.session_state.df.columns:
                axes[0, 1].hist(st.session_state.df['AnnualIncome_k'], bins=20, alpha=0.7, color='green')
                axes[0, 1].set_title('Income Distribution')
                axes[0, 1].set_xlabel('Annual Income (k$)')
                axes[0, 1].set_ylabel('Frequency')
            else:
                axes[0, 1].text(0.5, 0.5, 'AnnualIncome_k column not found', ha='center', va='center')
                axes[0, 1].set_title('Income Distribution')
            
            # Spending score distribution
            if 'SpendingScore' in st.session_state.df.columns:
                axes[1, 0].hist(st.session_state.df['SpendingScore'], bins=20, alpha=0.7, color='orange')
                axes[1, 0].set_title('Spending Score Distribution')
                axes[1, 0].set_xlabel('Spending Score')
                axes[1, 0].set_ylabel('Frequency')
            else:
                axes[1, 0].text(0.5, 0.5, 'SpendingScore column not found', ha='center', va='center')
                axes[1, 0].set_title('Spending Score Distribution')
            
            # Gender distribution
            if 'Gender' in st.session_state.df.columns:
                gender_counts = st.session_state.df['Gender'].value_counts()
                axes[1, 1].pie(gender_counts.values, labels=gender_counts.index, autopct='%1.1f%%')
                axes[1, 1].set_title('Gender Distribution')
            else:
                axes[1, 1].text(0.5, 0.5, 'Gender column not found', ha='center', va='center')
                axes[1, 1].set_title('Gender Distribution')
            
            plt.tight_layout()
            st.pyplot(fig)

# Train model
if train_btn and st.session_state.df is not None:
    with st.spinner("Training K-Means model..."):
        try:
            # Preprocessing
            df_processed = st.session_state.df.copy()
            
            # Encode gender if exists
            if 'Gender' in df_processed.columns:
                df_processed['Gender_Encoded'] = df_processed['Gender'].map({'Female': 0, 'Male': 1})
            
            # Select features - use Age, AnnualIncome_k, SpendingScore if available
            features = []
            for col in ['Age', 'AnnualIncome_k', 'SpendingScore']:
                if col in df_processed.columns:
                    features.append(col)
            
            if len(features) < 2:
                st.error("Need at least 2 features for clustering! Make sure your data has Age, AnnualIncome_k, or SpendingScore columns.")
            else:
                X = df_processed[features]
                
                # Scale features
                scaler = StandardScaler()
                X_scaled = scaler.fit_transform(X)
                
                # Elbow method to determine optimal k
                wcss = []
                silhouette_scores = []
                k_range = range(2, 11)
                
                for k in k_range:
                    kmeans_temp = KMeans(n_clusters=k, init='k-means++', random_state=42, n_init=10)
                    kmeans_temp.fit(X_scaled)
                    wcss.append(kmeans_temp.inertia_)
                    
                    if k > 1:
                        silhouette_scores.append(silhouette_score(X_scaled, kmeans_temp.labels_))
                
                # Train with selected k
                k = st.session_state.optimal_k
                kmeans = KMeans(n_clusters=k, init='k-means++', random_state=42, n_init=10)
                cluster_labels = kmeans.fit_predict(X_scaled)
                
                # Add cluster labels
                df_processed['Cluster'] = cluster_labels
                
                # Calculate metrics
                silhouette_avg = silhouette_score(X_scaled, cluster_labels)
                wcss_final = kmeans.inertia_
                
                # Update session state
                st.session_state.kmeans_model = kmeans
                st.session_state.scaler = scaler
                st.session_state.features = features
                st.session_state.df_processed = df_processed
                st.session_state.X_scaled = X_scaled
                st.session_state.model_trained = True
                
                st.success("✅ Model trained successfully!")
                
                # Display metrics
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Clusters", k)
                with col2:
                    st.metric("Silhouette Score", f"{silhouette_avg:.3f}")
                with col3:
                    st.metric("WCSS", f"{wcss_final:.2f}")
                
                # Show elbow plot
                st.subheader("Elbow Method Analysis")
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
                
                ax1.plot(k_range, wcss, 'bo-')
                ax1.set_xlabel('Number of Clusters (k)')
                ax1.set_ylabel('WCSS')
                ax1.set_title('Elbow Method')
                ax1.axvline(x=k, color='r', linestyle='--', alpha=0.5)
                
                ax2.plot(range(2, 11), silhouette_scores, 'go-')
                ax2.set_xlabel('Number of Clusters (k)')
                ax2.set_ylabel('Silhouette Score')
                ax2.set_title('Silhouette Score')
                ax2.axvline(x=k, color='r', linestyle='--', alpha=0.5)
                
                st.pyplot(fig)
                
        except Exception as e:
            st.error(f"Error training model: {str(e)}")

# Save model
if save_btn and st.session_state.model_trained:
    try:
        model_data = {
            'kmeans': st.session_state.kmeans_model,
            'scaler': st.session_state.scaler,
            'features': st.session_state.features,
            'optimal_k': st.session_state.optimal_k
        }
        
        with open('trained_model.pkl', 'wb') as f:
            pickle.dump(model_data, f)
        
        # Also save the processed data
        if st.session_state.df_processed is not None:
            st.session_state.df_processed.to_csv('customer_segmentation_results.csv', index=False)
        
        st.success("✅ Model saved as 'trained_model.pkl'")
        st.success("✅ Results saved as 'customer_segmentation_results.csv'")
        
    except Exception as e:
        st.error(f"Error saving model: {str(e)}")

# Load model
if load_btn:
    try:
        with open('trained_model.pkl', 'rb') as f:
            model_data = pickle.load(f)
        
        st.session_state.kmeans_model = model_data['kmeans']
        st.session_state.scaler = model_data['scaler']
        st.session_state.features = model_data['features']
        st.session_state.optimal_k = model_data['optimal_k']
        st.session_state.model_trained = True
        
        # Try to load results
        try:
            if os.path.exists('customer_segmentation_results.csv'):
                df_results = pd.read_csv('customer_segmentation_results.csv')
                st.session_state.df_processed = df_results
        except Exception as e:
            st.warning(f"Could not load results: {str(e)}")
        
        st.success("✅ Model loaded from 'trained_model.pkl'")
        
    except FileNotFoundError:
        st.error("❌ 'trained_model.pkl' not found. Please train and save a model first.")
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")

# Display clustering results
if st.session_state.model_trained and st.session_state.df_processed is not None:
    st.header("Clustering Results")
    
    df_processed = st.session_state.df_processed
    k = st.session_state.optimal_k
    
    # Cluster distribution
    st.subheader("Cluster Distribution")
    cluster_counts = df_processed['Cluster'].value_counts().sort_index()
    
    fig1, ax1 = plt.subplots(figsize=(10, 5))
    bars = ax1.bar([str(i) for i in cluster_counts.index], cluster_counts.values)
    ax1.set_xlabel("Cluster")
    ax1.set_ylabel("Number of Customers")
    ax1.set_title("Customers per Cluster")
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                f'{int(height)}', ha='center', va='bottom')
    
    st.pyplot(fig1)
    
    # Cluster visualization
    st.subheader("Cluster Visualization")
    
    if len(st.session_state.features) >= 2:
        fig2, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Feature 1 vs Feature 2
        scatter1 = axes[0].scatter(df_processed[st.session_state.features[0]], 
                                  df_processed[st.session_state.features[1]],
                                  c=df_processed['Cluster'], cmap='viridis', alpha=0.7)
        axes[0].set_xlabel(st.session_state.features[0])
        axes[0].set_ylabel(st.session_state.features[1])
        axes[0].set_title(f"{st.session_state.features[0]} vs {st.session_state.features[1]}")
        plt.colorbar(scatter1, ax=axes[0])
        
        if len(st.session_state.features) >= 3:
            # Feature 1 vs Feature 3
            scatter2 = axes[1].scatter(df_processed[st.session_state.features[0]], 
                                      df_processed[st.session_state.features[2]],
                                      c=df_processed['Cluster'], cmap='plasma', alpha=0.7)
            axes[1].set_xlabel(st.session_state.features[0])
            axes[1].set_ylabel(st.session_state.features[2])
            axes[1].set_title(f"{st.session_state.features[0]} vs {st.session_state.features[2]}")
            plt.colorbar(scatter2, ax=axes[1])
        
        st.pyplot(fig2)
    
    # Cluster profiles
    st.subheader("Cluster Profiles")
    
    cluster_profiles = []
    for cluster_num in range(k):
        cluster_data = df_processed[df_processed['Cluster'] == cluster_num]
        
        profile = {
            'Cluster': cluster_num,
            'Size': len(cluster_data),
            'Percentage': (len(cluster_data) / len(df_processed)) * 100,
        }
        
        # Add feature statistics
        for feature in st.session_state.features:
            profile[f'{feature}_mean'] = cluster_data[feature].mean()
            profile[f'{feature}_std'] = cluster_data[feature].std()
        
        # Add gender statistics if available
        if 'Gender' in df_processed.columns:
            gender_counts = cluster_data['Gender'].value_counts()
            if 'Male' in gender_counts:
                profile['Male_%'] = (gender_counts['Male'] / len(cluster_data)) * 100
            if 'Female' in gender_counts:
                profile['Female_%'] = (gender_counts.get('Female', 0) / len(cluster_data)) * 100
        
        cluster_profiles.append(profile)
    
    profile_df = pd.DataFrame(cluster_profiles)
    st.dataframe(profile_df.round(2), use_container_width=True)
    
    # Marketing recommendations
    st.subheader("Marketing Recommendations")
    
    for _, profile in profile_df.iterrows():
        cluster_num = int(profile['Cluster'])
        
        # Get relevant metrics
        avg_age = profile.get('Age_mean', 0)
        avg_income = profile.get('AnnualIncome_k_mean', 0)
        avg_spending = profile.get('SpendingScore_mean', 0)
        
        # Determine recommendation
        if avg_income > 60 and avg_spending > 60:
            recommendation = "Premium Segment - Target with luxury products and exclusive offers"
            color = "🟣"
        elif avg_income > 50 and avg_spending > 50:
            recommendation = "High-value Segment - Offer loyalty programs and bundled products"
            color = "🔵"
        elif avg_income < 40 and avg_spending > 60:
            recommendation = "Spendthrift Segment - Target with discounts and limited-time offers"
            color = "🟢"
        elif avg_income > 50 and avg_spending < 40:
            recommendation = "Saver Segment - Focus on value-for-money products"
            color = "🟡"
        elif avg_age < 30:
            recommendation = "Young Segment - Use social media marketing and trendy products"
            color = "🔴"
        elif avg_age > 50:
            recommendation = "Senior Segment - Focus on comfort and reliability"
            color = "🟠"
        else:
            recommendation = "Standard Segment - General marketing with balanced offers"
            color = "⚫"
        
        with st.expander(f"{color} Cluster {cluster_num} ({int(profile['Size'])} customers)"):
            st.write(f"**Recommendation:** {recommendation}")
            if avg_age > 0:
                st.write(f"**Average Age:** {avg_age:.1f}")
            if avg_income > 0:
                st.write(f"**Average Income:** ${avg_income:.1f}k")
            if avg_spending > 0:
                st.write(f"**Average Spending Score:** {avg_spending:.1f}")
    
    # Prediction interface
    st.subheader("Predict Cluster for New Customer")
    
    if st.session_state.features:
        cols = st.columns(len(st.session_state.features))
        input_data = {}
        
        for i, feature in enumerate(st.session_state.features):
            with cols[i]:
                if feature == 'Age':
                    input_data[feature] = st.number_input("Age", min_value=18, max_value=100, value=30, key=f"age_input")
                elif feature == 'AnnualIncome_k':
                    input_data[feature] = st.number_input("Annual Income (k$)", min_value=10, max_value=200, value=50, key=f"income_input")
                elif feature == 'SpendingScore':
                    input_data[feature] = st.number_input("Spending Score", min_value=1, max_value=100, value=50, key=f"spending_input")
                else:
                    input_data[feature] = st.number_input(feature, value=0.0, key=f"{feature}_input")
        
        if st.button("Predict Cluster"):
            # Prepare input
            input_array = []
            for feature in st.session_state.features:
                input_array.append(input_data.get(feature, 0))
            
            input_array = np.array(input_array).reshape(1, -1)
            
            # Scale and predict
            input_scaled = st.session_state.scaler.transform(input_array)
            cluster = st.session_state.kmeans_model.predict(input_scaled)[0]
            
            st.success(f"**Predicted Cluster:** {cluster}")
            
            # Show cluster details
            cluster_data = df_processed[df_processed['Cluster'] == cluster]
            st.write(f"**Cluster {cluster} has {len(cluster_data)} similar customers**")
            
            if 'Age' in cluster_data.columns:
                st.write(f"• Average Age in Cluster: {cluster_data['Age'].mean():.1f}")
            if 'AnnualIncome_k' in cluster_data.columns:
                st.write(f"• Average Income: ${cluster_data['AnnualIncome_k'].mean():.1f}k")
            if 'SpendingScore' in cluster_data.columns:
                st.write(f"• Average Spending Score: {cluster_data['SpendingScore'].mean():.1f}")

# Export section
if st.session_state.model_trained:
    st.divider()
    st.header("Export Results")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.session_state.df_processed is not None:
            csv = st.session_state.df_processed.to_csv(index=False)
            st.download_button(
                label="📥 Download Clustered Data",
                data=csv,
                file_name="customer_segmentation_results.csv",
                mime="text/csv"
            )
    
    with col2:
        if os.path.exists('trained_model.pkl'):
            with open('trained_model.pkl', 'rb') as f:
                model_bytes = f.read()
            
            st.download_button(
                label="📥 Download Model",
                data=model_bytes,
                file_name="trained_model.pkl",
                mime="application/octet-stream"
            )
        else:
            st.info("Save the model first to download it")

# Instructions - FIXED LINE
if st.session_state.df is None:  # Changed from: if not st.session_state.df:
    st.info("👈 Please generate sample data or upload a CSV file from the sidebar to get started.")

# Footer
st.divider()
st.markdown("**Files:** `trained_model.pkl` | `customer_segmentation_results.csv`")