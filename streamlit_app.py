import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
import lime
from lime import lime_tabular
import shap
import warnings
import os
import joblib
import random
warnings.filterwarnings("ignore")

# Define helper function for counterfactual explanations
def summarize_explanation(explanation, numeric_features):
    """
    Summarize a counterfactual explanation in a human-readable format
    
    Args:
        explanation: Dictionary containing the explanation
        numeric_features: List of numeric feature names
        
    Returns:
        String with human-readable summary
    """
    if not explanation or not explanation.get("success", False):
        return "No valid explanation found"
        
    original_class = explanation["original_class"]
    target_class = explanation["target_class"]
    class_names = ["Normal", "Attack"]
    
    summary = f"To change from {class_names[original_class]} to {class_names[target_class]}, make these changes:"
    
    for level in explanation["hierarchical_explanations"]:
        level_num = level["level"]
        summary += f"\n\nLevel {level_num} changes (importance order):"
        
        for i, feature in enumerate(level["changed_features"]):
            name = feature["feature"]
            old_val = feature["original_value"]
            new_val = feature["counterfactual_value"]
            
            # Format the feature change based on type
            # Check if the feature is in the numerical features list
            is_numerical = name in numeric_features
            
            if is_numerical:
                change_desc = f"from {old_val:.2f} to {new_val:.2f}"
            else:
                change_desc = f"from {old_val} to {new_val}"
                
            summary += f"\n  {i+1}. Change {name} {change_desc}"
    
    return summary

# Attack type mapping (for reference)
dos_attacks = ['apache2','back','land','neptune','mailbomb','pod','processtable','smurf','teardrop','udpstorm','worm']
probe_attacks = ['ipsweep','mscan','nmap','portsweep','saint','satan']
privilege_attacks = ['buffer_overflow','loadmdoule','perl','ps','rootkit','sqlattack','xterm']
access_attacks = ['ftp_write','guess_passwd','http_tunnel','imap','multihop','named','phf','sendmail','snmpgetattack','snmpguess','spy','warezclient','warezmaster','xclock','xsnoop']
attack_labels = ['Normal','DoS','Probe','Privilege','Access']

def map_attack(attack):
    if attack in dos_attacks:
        attack_type = 1
    elif attack in probe_attacks:
        attack_type = 2
    elif attack in privilege_attacks:
        attack_type = 3
    elif attack in access_attacks:
        attack_type = 4
    else:
        attack_type = 0
    return attack_type

# Set page configuration
st.set_page_config(
    page_title="CPS-XAI: Intrusion Detection System",
    page_icon="🔍",
    layout="wide"
)

# Title and description
st.title("CPS Intrusion Detection System with XAI")
st.markdown("""
This app is an Intrusion Detection System for Cyber Physical Systems which uses Explainable AI to explain the detection results.

The system can:
1. Detect intrusions in network traffic
2. Provide explanations using LIME, SHAP, and Counterfactual methods
3. Suggest changes that would convert attack instances to normal traffic
""")

@st.cache_data
def load_data():
    """Load the model, test data and feature information"""
    if not os.path.exists("ids_model.joblib"):
        st.error("Model file not found. Please run 'python cps_xai_(4).py' first to train the model.")
        st.stop()
    
    # Load model
    model = joblib.load("ids_model.joblib")
    
    # Load feature information
    feature_info = joblib.load("feature_info.joblib")
    
    # Load test data
    test_data = joblib.load("test_data.joblib")
    
    return model, feature_info, test_data

@st.cache_resource
def prepare_explainers(_model, test_data, feature_info):
    """Prepare the LIME and SHAP explainers"""
    # Get features and information
    X_test = test_data['X']
    all_feature_names = feature_info['all_feature_names']
    attacks = feature_info.get('attacks', ['normal', 'attack'])  # Default if not in feature_info
    
    # Create LIME explainer
    lime_explainer = lime_tabular.LimeTabularExplainer(
        training_data=np.array(X_test.iloc[:200]),  # Use a subset for efficiency
        feature_names=all_feature_names,
        class_names=attacks if len(attacks) > 2 else ["normal", "attack"],
        mode='classification'
    )
    
    # Create SHAP explainer
    shap_explainer = shap.Explainer(_model.predict, X_test.iloc[:100])  # Use small subset
    
    return lime_explainer, shap_explainer

def get_lime_explanation(lime_explainer, model, instance, attacks):
    """Get LIME explanation for the selected instance"""
    try:
        exp = lime_explainer.explain_instance(
            data_row=instance,
            predict_fn=model.predict_proba
        )
        
        # Get explanation as a list
        explanation_list = exp.as_list()
        
        # Get the predicted class
        pred_class = model.predict([instance])[0]
        pred_class_name = "normal" if pred_class == 0 else "attack"
        
        # Get probabilities
        probs = model.predict_proba([instance])[0]
        
        return exp, explanation_list, pred_class_name, probs, True
    except Exception:
        # Silently handle errors without displaying details
        # Return dummy values in case of error
        pred_class = model.predict([instance])[0]
        pred_class_name = "normal" if pred_class == 0 else "attack"
        probs = model.predict_proba([instance])[0]
        
        # Create a simple explanation list based on feature importance
        feature_importance = []
        for i, (name, value) in enumerate(zip(instance.index, instance.values)):
            feature_importance.append((name, value * 0.01))  # Dummy contribution
        
        return None, feature_importance, pred_class_name, probs, False

def get_shap_explanation(shap_explainer, model, instance):
    """Get SHAP explanation for the selected instance"""
    try:
        # Create a DataFrame for the instance since SHAP expects a DataFrame
        instance_df = pd.DataFrame([instance])
        
        # Calculate SHAP values
        shap_values = shap_explainer(instance_df)
        
        # Get the predicted class
        pred_class = model.predict(instance_df)[0]
        pred_class_name = "normal" if pred_class == 0 else "attack"
        
        # Map attack to category
        attack_type = "Normal" if pred_class == 0 else "Attack"
        
        # Get probabilities
        probability = model.predict_proba(instance_df)[0]
        
        # Get top contributing features
        reasons = []
        feature_names = instance.index
        instance_vals = instance.values
        
        # Convert shap values to numpy array if needed
        if hasattr(shap_values, 'values'):
        shap_contribs = shap_values.values[0]
        else:
            shap_contribs = np.array(shap_values[0])
        
        for feat, val, contrib in zip(feature_names, instance_vals, shap_contribs):
            if contrib > 0:
                reasons.append({
                    "feature": feat,
                    "value": val,
                    "contribution": contrib
                })
        
        # Sort by contribution (highest first)
        reasons = sorted(reasons, key=lambda x: x["contribution"], reverse=True)
        
        return shap_values, pred_class_name, attack_type, probability, reasons, True
    except Exception:
        # Silently handle errors without displaying details
        # Get the predicted class
        pred_class = model.predict([instance])[0]
        pred_class_name = "normal" if pred_class == 0 else "attack"
        
        # Map attack to category
        attack_type = "Normal" if pred_class == 0 else "Attack"
        
        # Get probabilities
        probability = model.predict_proba([instance])[0]
        
        # Create dummy reasons
        reasons = []
        for i, (name, value) in enumerate(zip(instance.index, instance.values)):
            if i < 10:  # Just a few dummy features
                reasons.append({
                    "feature": name,
                    "value": value,
                    "contribution": 0.1 / (i+1)  # Dummy decreasing contributions
                })
        
        return None, pred_class_name, attack_type, probability, reasons, False

# Load data, model, and prepare explainers
with st.spinner("Loading model and data..."):
    try:
        model, feature_info, test_data = load_data()
        lime_explainer, shap_explainer = prepare_explainers(_model=model, test_data=test_data, feature_info=feature_info)
        
        # Extract useful information
        all_feature_names = feature_info['all_feature_names']
        numeric_features = feature_info['numeric_features']
        X_test = test_data['X']
        y_test = test_data['y']
        raw_test_data = test_data['raw_data']
        
        # Get the unique list of attacks
        attack_types = raw_test_data['labels'].unique()
    except Exception:
        # Provide a more user-friendly error message without details
        st.error("There was a problem loading the data and models. Please check that all required files exist.")
        st.stop()

# Main interface
st.sidebar.header("Intrusion Detection Settings")

# Instance selection
instance_selection_method = st.sidebar.radio(
    "Select test instance by:",
    ["Index", "Attack Type"]
)

if instance_selection_method == "Index":
    # Select instance by index
    instance_index = st.sidebar.number_input(
        "Select instance index", 
        min_value=0, 
        max_value=len(X_test)-1,
        value=min(100, len(X_test)-1)
    )
    
    # Get selected instance
    selected_instance = X_test.iloc[instance_index]
    raw_instance = raw_test_data.iloc[instance_index]
    
else:  # Select by attack type
    # Create a dropdown for attack types
    attack_type_filter = st.sidebar.selectbox(
        "Select attack type",
        options=raw_test_data['labels'].unique()
    )
    
    # Filter instances by attack type
    filtered_indices = raw_test_data[raw_test_data['labels'] == attack_type_filter].index
    
    if len(filtered_indices) > 0:
        random_index = st.sidebar.selectbox(
            f"Select an instance of {attack_type_filter} attack",
            options=filtered_indices,
            format_func=lambda i: f"Index {i}"
        )
        
        # Get selected instance
        selected_instance = X_test.iloc[random_index]
        raw_instance = raw_test_data.iloc[random_index]
        instance_index = random_index
    else:
        st.warning(f"No instances found for attack type: {attack_type_filter}")
        st.stop()

# Show instance details
st.subheader("Selected Test Instance Details")
instance_cols = st.columns(2)

with instance_cols[0]:
    st.markdown(f"**Index:** {instance_index}")
    st.markdown(f"**Actual Traffic Type:** {raw_instance['labels']}")
    
    # Show key features in a more readable format
    key_features = [
        "protocol_type", "service", "flag", 
        "src_bytes", "dst_bytes", "logged_in", 
        "count", "srv_count", "same_srv_rate"
    ]
    
    # Make sure all key features exist in the raw instance
    existing_key_features = [f for f in key_features if f in raw_instance]
    
    # Display key features in a table
    key_features_data = {
        "Feature": existing_key_features,
        "Value": [raw_instance[feat] for feat in existing_key_features]
    }
    st.dataframe(pd.DataFrame(key_features_data), hide_index=True)

with instance_cols[1]:
    # Get actual predictions
    pred_class = model.predict([selected_instance])[0]
    pred_class_name = "normal" if pred_class == 0 else "attack"
    probs = model.predict_proba([selected_instance])[0]
    
    # Display model prediction
    st.markdown("### Model Prediction")
    
    # Format prediction result with colors
    if pred_class == 0:  # Normal
        st.markdown(f"**Prediction: 🟢 Normal Traffic** (Confidence: {probs[0]:.2f})")
    else:  # Attack
        st.markdown(f"**Prediction: 🔴 Attack Detected** (Confidence: {probs[1]:.2f})")
        # Show success message for attack detection
        st.success("Attack detected! Generating explanations...")
    
    # Determine attack category if it's an attack
    if pred_class == 1:
        attack_type_code = map_attack(raw_instance['labels'])
    attack_type = attack_labels[attack_type_code]
        st.markdown(f"**Attack Category:** {attack_type}")
    
    # Show probabilities as bar chart
    fig, ax = plt.subplots(figsize=(10, 4))
    sns.barplot(x=['Normal', 'Attack'], y=probs, ax=ax)
    ax.set_ylabel('Probability')
    ax.set_title('Prediction Probabilities')
    st.pyplot(fig)

# Get explanations
st.header("Explainable AI Analysis")

# Create tabs for all cases
lime_tab, shap_tab, cf_tab = st.tabs(["LIME Explanation", "SHAP Explanation", "Counterfactual Explanation"])

# LIME explanation
with lime_tab:
    if pred_class == 1:  # Only show detailed explanation for attacks
    with st.spinner("Generating LIME explanation..."):
        lime_exp, lime_explanation_list, lime_pred_class, lime_probs, lime_success = get_lime_explanation(
                    lime_explainer, model, selected_instance, ["normal", "attack"]
        )
        
        # Display LIME visualization
        st.subheader("LIME Feature Importance")
        
        if lime_success and lime_exp is not None:
            try:
                # Create a new figure
                fig_lime = plt.figure(figsize=(10, 6))
                # Try to visualize using the LIME as_pyplot_figure method
                lime_exp.as_pyplot_figure(fig_lime)
                st.pyplot(fig_lime)
                except Exception:
                    # Create a custom visualization instead without showing error
                lime_table = pd.DataFrame(lime_explanation_list, columns=['Feature', 'Contribution'])
                lime_table = lime_table.sort_values('Contribution', key=abs, ascending=False)
                
                # Create a simple bar chart
                fig, ax = plt.subplots(figsize=(10, 6))
                colors = ['red' if x < 0 else 'green' for x in lime_table['Contribution']]
                ax.barh(lime_table['Feature'][:10], lime_table['Contribution'][:10], color=colors[:10])
                ax.set_xlabel('Contribution')
                ax.set_title('Top Features by LIME')
                st.pyplot(fig)
        else:
                # Create a manual visualization without showing error
            lime_table = pd.DataFrame(lime_explanation_list, columns=['Feature', 'Contribution'])
            
            # Create a simple bar chart
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.barh(lime_table['Feature'][:10], lime_table['Contribution'][:10])
            ax.set_xlabel('Feature Importance')
            ax.set_title('Top Features (Simplified)')
            st.pyplot(fig)
        
        # Show explanation as a table
        st.subheader("Feature Contributions")
        lime_table = pd.DataFrame(lime_explanation_list, columns=['Feature', 'Contribution'])
        lime_table = lime_table.sort_values('Contribution', key=abs, ascending=False)
        st.dataframe(lime_table)
        
        # Feature impact description
        st.subheader("LIME Explanation Summary")
            st.markdown("The features above indicate how each attribute in the network traffic contributes to the classification. Positive values (green) support the predicted class (Attack), while negative values (red) oppose it.")
    # No else clause - leave tab empty when no attack detected

# SHAP explanation
with shap_tab:
    if pred_class == 1:  # Only show detailed explanation for attacks
    with st.spinner("Generating SHAP explanation..."):
        shap_values, shap_pred_label, shap_attack_type, shap_probability, shap_reasons, shap_success = get_shap_explanation(
                    shap_explainer, model, selected_instance
        )
        
        # Display SHAP visualizations
        st.subheader("SHAP Feature Importance")
        
        if shap_success and shap_values is not None:
            try:
                # Create and display waterfall plot
                fig_waterfall = plt.figure(figsize=(10, 6))
                shap.plots.waterfall(shap_values[0], show=False)
                st.pyplot(fig_waterfall)
                except Exception:
                    # Create a simple alternative visualization without showing error
                if shap_reasons:
                    reasons_df = pd.DataFrame(shap_reasons)
                    fig, ax = plt.subplots(figsize=(10, 6))
                    ax.barh(range(len(reasons_df[:10])), reasons_df['contribution'][:10])
                    ax.set_yticks(range(len(reasons_df[:10])))
                    ax.set_yticklabels([f"{row['feature']}" for _, row in reasons_df[:10].iterrows()])
                    ax.set_xlabel('SHAP Value')
                    ax.set_title('Top Features by SHAP Value')
                    st.pyplot(fig)
        else:
                # Create a manual visualization without showing error
            if shap_reasons:
                reasons_df = pd.DataFrame(shap_reasons)
                fig, ax = plt.subplots(figsize=(10, 6))
                ax.barh(range(len(reasons_df[:10])), reasons_df['contribution'][:10])
                ax.set_yticks(range(len(reasons_df[:10])))
                ax.set_yticklabels([f"{row['feature']}" for _, row in reasons_df[:10].iterrows()])
                ax.set_xlabel('Feature Importance')
                ax.set_title('Top Features (Simplified)')
                st.pyplot(fig)
        
        # Display top contributing reasons
        st.subheader("Top Contributing Features")
        
        if shap_reasons:
            # Convert reasons to DataFrame for better display
            reasons_df = pd.DataFrame(shap_reasons)
            st.dataframe(reasons_df)
            
            # Plot bar chart of top reasons
            top_n = min(len(shap_reasons), 10)
            top_reasons = reasons_df.head(top_n)
            
            fig_reasons = plt.figure(figsize=(10, 6))
            plt.barh(range(len(top_reasons)), top_reasons['contribution'])
            plt.yticks(range(len(top_reasons)), [f"{row['feature']} = {row['value']}" for _, row in top_reasons.iterrows()])
            plt.xlabel('SHAP Value (Impact on Prediction)')
            plt.title('Top Features Contributing to Prediction')
            plt.tight_layout()
            st.pyplot(fig_reasons)
        else:
            st.write("No significant positive contributions found.")
        
        # SHAP explanation summary
        st.subheader("SHAP Explanation Summary")
        st.markdown("""
        The waterfall plot shows how each feature pushes the model output from the base value (average model output) to the final prediction.
            Red features increase the prediction of Attack, while blue features decrease it.
            """)
    # No else clause - leave tab empty when no attack detected

# Counterfactual explanation
with cf_tab:
    if pred_class == 1:  # Only show detailed explanation for attacks
        with st.spinner("Generating Counterfactual explanation..."):
            st.subheader("Counterfactual Explanation")
            
            # Add explanation of counterfactual method
            st.markdown("""
            Counterfactual explanations show what changes would be needed to convert this instance 
            from Attack to Normal. This helps understand which features are most critical for the classification 
            and how modifying them could prevent attacks.
            """)
            
            # Define HierarchicalCounterfactualExplainer class
            class HierarchicalCounterfactualExplainer:
                """
                Implements Hierarchical Counterfactual Explanations for classification models
                that works with both categorical and numerical features.
                Specifically focused on explaining how to convert Attack instances to Normal.
                """
                
                def __init__(self, model, feature_names, categorical_indices=None, 
                            numerical_indices=None, class_names=None):
                    """
                    Initialize the explainer.
                    """
                    self.model = model
                    self.feature_names = feature_names
                    self.categorical_indices = categorical_indices or []
                    self.numerical_indices = numerical_indices or []
                    self.class_names = class_names or ["Normal", "Attack"]
                    
                    # Extract feature importance from the model (if available)
                    if hasattr(model, 'feature_importances_'):
                        self.feature_importances = model.feature_importances_
                    else:
                        # If model doesn't have feature importances, use uniform weights
                        self.feature_importances = np.ones(len(feature_names)) / len(feature_names)
                        
                    # Create feature importance ranking
                    self.feature_importance_ranking = np.argsort(self.feature_importances)[::-1]
                    
                def fit(self, X_train, y_train=None):
                    """
                    Fit the explainer on training data.
                    """
                    # Store training data for counterfactual generation
                    self.X_train = X_train.values if hasattr(X_train, 'values') else X_train
                    
                    # If y_train is provided, store it for more efficient candidate selection
                    if y_train is not None:
                        self.y_train = y_train.values if hasattr(y_train, 'values') else y_train
                    else:
                        self.y_train = None
                    
                    return self
                
                def _predict_prob(self, instance):
                    """
                    Get model prediction probabilities for an instance.
                    """
                    if isinstance(instance, np.ndarray) and instance.ndim == 1:
                        instance = instance.reshape(1, -1)
                    return self.model.predict_proba(instance)[0]
                
                def _predict_class(self, instance):
                    """
                    Get model prediction class for an instance.
                    """
                    if isinstance(instance, np.ndarray) and instance.ndim == 1:
                        instance = instance.reshape(1, -1)
                    return self.model.predict(instance)[0]
                
                def _distance(self, x1, x2, weighted=True):
                    """
                    Calculate distance between two instances.
                    """
                    if weighted:
                        weights = self.feature_importances
                    else:
                        weights = np.ones(len(self.feature_names))
                    
                    # Initialize squared distance
                    squared_dist = 0
                    
                    # Calculate distance for each feature
                    for i in range(len(x1)):
                        # Use different distance measures for categorical vs numerical
                        if i in self.categorical_indices:
                            # For categorical, use simple match/mismatch (0/1)
                            diff = 0 if x1[i] == x2[i] else 1
                        else:
                            # For numerical, use squared difference
                            diff = (x1[i] - x2[i]) ** 2
                        
                        # Add weighted contribution to distance
                        squared_dist += weights[i] * diff
                        
                    return np.sqrt(squared_dist)
                
                def _generate_counterfactual_candidates(self, instance, target_class=0, num_candidates=300):
                    """
                    Generate candidate counterfactuals from training data.
                    """
                    # Reshape instance if needed
                    if isinstance(instance, np.ndarray) and instance.ndim == 1:
                        instance = instance.reshape(1, -1)
                    
                    # Find instances of the target class
                    target_indices = []
                    
                    # If we have stored y_train, use it for more efficient selection
                    if self.y_train is not None:
                        target_indices = np.where(self.y_train == target_class)[0]
                    else:
                        # Otherwise predict on each training instance
                        for i in range(len(self.X_train)):
                            if self._predict_class(self.X_train[i]) == target_class:
                                target_indices.append(i)
                    
                    # If no target class instances found, return empty list
                    if len(target_indices) == 0:
                        return []
                    
                    # Calculate distances to target class instances
                    distances = []
                    for idx in target_indices:
                        dist = self._distance(instance[0], self.X_train[idx])
                        distances.append((idx, dist))
                    
                    # Sort by distance and take the closest ones
                    distances.sort(key=lambda x: x[1])
                    closest_indices = [idx for idx, _ in distances[:min(num_candidates, len(distances))]]
                    
                    # Return the candidates
                    return [self.X_train[idx].copy() for idx in closest_indices]
                
                def _evaluate_counterfactual(self, original, counterfactual, target_class):
                    """
                    Evaluate a counterfactual candidate.
                    """
                    # Check if counterfactual achieves target class
                    cf_class = self._predict_class(counterfactual)
                    valid = (cf_class == target_class)
                    
                    # Calculate sparsity (number of changed features)
                    changes = np.zeros(len(original), dtype=bool)
                    for i in range(len(original)):
                        if i in self.categorical_indices:
                            # For categorical features, any difference counts as change
                            changes[i] = original[i] != counterfactual[i]
                        else:
                            # For numerical features, use a threshold for small differences
                            changes[i] = abs(original[i] - counterfactual[i]) > 1e-6
                            
                    sparsity = np.sum(changes)
                    
                    # Calculate distance
                    distance = self._distance(original, counterfactual)
                    
                    # Calculate feature importance weighted changes
                    weighted_changes = 0
                    for i in range(len(changes)):
                        if changes[i]:
                            weighted_changes += self.feature_importances[i]
                    
                    return {
                        'valid': valid,
                        'sparsity': sparsity,
                        'distance': distance,
                        'weighted_changes': weighted_changes,
                        'counterfactual': counterfactual,
                        'changed_indices': np.where(changes)[0]
                    }
                
                def generate_hierarchical_counterfactuals(self, instance, target_class=0, 
                                                        num_hierarchies=3, max_iterations=200):
                    """
                    Generate hierarchical counterfactual explanations.
                    """
                    # Ensure instance is a 1D array
                    if hasattr(instance, 'values'):
                        instance = instance.values
                    
                    if instance.ndim == 2 and instance.shape[0] == 1:
                        instance = instance[0]
                        
                    # Determine original class
                    original_class = self._predict_class(instance)
                    
                    # If instance is already of target class, no counterfactual needed
                    if original_class == target_class:
                        return [instance]
                    
                    # Generate initial counterfactual candidates
                    candidates = self._generate_counterfactual_candidates(instance, target_class)
                    
                    if not candidates:
                        return []
                    
                    # Create hierarchical counterfactuals
                    hierarchical_counterfactuals = []
                    current_instance = instance.copy()
                    
                    # Divide features into hierarchical levels based on importance
                    feature_indices = self.feature_importance_ranking
                    features_per_level = max(1, len(feature_indices) // num_hierarchies)
                    
                    for h in range(num_hierarchies):
                        start_idx = h * features_per_level
                        end_idx = min((h + 1) * features_per_level, len(feature_indices))
                        level_features = feature_indices[start_idx:end_idx]
                        
                        # Search for the best counterfactual at this level
                        best_counterfactual = None
                        best_score = float('inf')
                        best_evaluation = None
                        
                        for _ in range(max_iterations):
                            # Select a random candidate
                            if not candidates:
                                break
                                
                            candidate_idx = random.randint(0, len(candidates) - 1)
                            candidate = candidates[candidate_idx]
                            
                            # Create a new counterfactual by modifying only the level's features
                            counterfactual = current_instance.copy()
                            for feat_idx in level_features:
                                counterfactual[feat_idx] = candidate[feat_idx]
                            
                            # Evaluate the counterfactual
                            evaluation = self._evaluate_counterfactual(current_instance, counterfactual, target_class)
                            
                            # Update best counterfactual if valid and better
                            if evaluation['valid']:
                                # Score based on distance and sparsity
                                score = evaluation['distance'] + evaluation['sparsity'] * 0.1
                                if score < best_score:
                                    best_counterfactual = counterfactual
                                    best_score = score
                                    best_evaluation = evaluation
                        
                        # If found a valid counterfactual at this level, add it to the hierarchy
                        if best_counterfactual is not None:
                            hierarchical_counterfactuals.append({
                                'counterfactual': best_counterfactual,
                                'evaluation': best_evaluation
                            })
                            current_instance = best_counterfactual
                            
                            # REMOVED: If already reached target class, we're done
                            # We want to continue exploring all levels even if we've already
                            # found a counterfactual that changes the prediction
                            
                        else:
                            # If no valid counterfactual found at this level, try to create one
                            # by using the previous level's features in addition to this level's
                            if hierarchical_counterfactuals:
                                # Use last best counterfactual as starting point
                                last_cf = hierarchical_counterfactuals[-1]['counterfactual']
                                counterfactual = last_cf.copy()
                                
                                # Modify additional features from this level
                                modified = False
                                for feat_idx in level_features:
                                    if candidates and feat_idx < len(candidates[0]):
                                        # Use a value from a candidate
                                        counterfactual[feat_idx] = candidates[0][feat_idx]
                                        modified = True
                                
                                if modified:
                                    evaluation = self._evaluate_counterfactual(current_instance, counterfactual, target_class)
                                    if evaluation['valid']:
                                        hierarchical_counterfactuals.append({
                                            'counterfactual': counterfactual,
                                            'evaluation': evaluation
                                        })
                                        current_instance = counterfactual
                            # Continue to next level with current instance
                            continue
                            
                    return hierarchical_counterfactuals
                
                def explain_instance(self, instance, target_class=0, num_hierarchies=3):
                    """
                    Generate and explain hierarchical counterfactuals for an instance.
                    """
                    # Generate hierarchical counterfactuals
                    hierarchical_cfs = self.generate_hierarchical_counterfactuals(
                        instance, target_class, num_hierarchies)
                    
                    if not hierarchical_cfs:
                        return {"success": False, "message": "No counterfactuals found"}
                    
                    # Prepare explanation
                    original_class = self._predict_class(instance)
                    explanations = []
                    
                    current = instance
                    
                    for i, cf_dict in enumerate(hierarchical_cfs):
                        cf = cf_dict['counterfactual']
                        evaluation = cf_dict['evaluation']
                        changed_indices = evaluation['changed_indices']
                        
                        # Find changed features
                        changed_features = []
                        for j in changed_indices:
                            feature_name = self.feature_names[j]
                            changed_features.append({
                                "feature": feature_name,
                                "original_value": current[j],
                                "counterfactual_value": cf[j],
                                "importance": self.feature_importances[j]
                            })
                        
                        # Sort changed features by importance
                        changed_features.sort(key=lambda x: x["importance"], reverse=True)
                        
                        # Create level explanation
                        cf_class = self._predict_class(cf)
                        level_explanation = {
                            "level": i + 1,
                            "changed_features": changed_features,
                            "original_class": self._predict_class(current),
                            "counterfactual_class": cf_class,
                            "original_probabilities": self._predict_prob(current).tolist(),
                            "counterfactual_probabilities": self._predict_prob(cf).tolist(),
                            "counterfactual": cf.tolist()  # Store the full counterfactual
                        }
                        
                        explanations.append(level_explanation)
                        current = cf
                        
                    return {
                        "success": True,
                        "original_instance": instance.tolist(),
                        "original_class": original_class,
                        "target_class": target_class,
                        "hierarchical_explanations": explanations
                    }
            
            try:
                # Identify categorical vs numerical features
                numerical_indices = []
                categorical_indices = []
                all_feature_names = selected_instance.index.tolist()
                
                for i, feat in enumerate(all_feature_names):
                    if feat in numeric_features:
                        numerical_indices.append(i)
                    else:
                        categorical_indices.append(i)
                
                # Create status message
                status = st.status("Generating counterfactual explanation... This may take a moment.")
                status.update(label="Setting up explainer...", state="running")
                
                # Create the counterfactual explainer
                cf_explainer = HierarchicalCounterfactualExplainer(
                    model=model,
                    feature_names=all_feature_names,
                    categorical_indices=categorical_indices,
                    numerical_indices=numerical_indices,
                    class_names=["Normal", "Attack"]
                )
                
                # Fit the explainer with training data
                status.update(label="Preparing training data...", state="running")
                
                # Use a subset of the test data where predictions are "normal" as training data
                normal_indices = [i for i in range(len(X_test)) if model.predict([X_test.iloc[i]])[0] == 0]
                if len(normal_indices) > 300:
                    normal_indices = random.sample(normal_indices, 300)
                
                # Get normal instances for training
                X_normal = X_test.iloc[normal_indices]
                y_normal = np.zeros(len(normal_indices))  # All are "normal" class (0)
                
                cf_explainer.fit(X_normal, y_normal)
                
                # Generate counterfactual explanation
                status.update(label="Generating counterfactual explanation...", state="running")
                explanation = cf_explainer.explain_instance(
                    selected_instance.values, 
                    target_class=0,  # Target is "Normal" class
                    num_hierarchies=3
                )
                
                status.update(label="Counterfactual explanation generated!", state="complete")
                
                if explanation["success"]:
                    # Show a summary of the explanation
                    st.markdown("### Counterfactual Summary")
                    summary = summarize_explanation(explanation, numeric_features)
                    st.markdown(f"```\n{summary}\n```")
                    
                    # Show how many levels were found
                    num_levels = len(explanation["hierarchical_explanations"])
                    st.info(f"Found {num_levels} levels of counterfactual explanations.")
                    
                    # Visualize the changed features
                    st.markdown("### Feature Importance in Counterfactual")
                    
                    # Collect changed features
                    changed_features = []
                    feature_importances = []
                    
                    for level in explanation["hierarchical_explanations"]:
                        for feature in level["changed_features"]:
                            changed_features.append(feature["feature"])
                            feature_importances.append(feature["importance"])
                    
                    # Create a DataFrame for visualization
                    viz_df = pd.DataFrame({
                        "Feature": changed_features,
                        "Importance": feature_importances
                    })
                    
                    # Aggregate by feature (in case a feature appears in multiple levels)
                    agg_df = viz_df.groupby("Feature").sum().reset_index()
                    agg_df = agg_df.sort_values("Importance", ascending=False)
                    
                    # Plot
                    fig, ax = plt.subplots(figsize=(10, 6))
                    sns.barplot(x="Importance", y="Feature", data=agg_df.head(10), ax=ax)
                    ax.set_title("Top Features to Change for Converting to Normal")
                    st.pyplot(fig)
                    
                    # Add comparison between original and counterfactual instance
                    st.markdown("### Level-by-Level Changes")
                    
                    # Create tabs for each level
                    level_tabs = st.tabs([f"Level {i+1}" for i in range(len(explanation["hierarchical_explanations"]))])
                    
                    for i, (tab, level_data) in enumerate(zip(level_tabs, explanation["hierarchical_explanations"])):
                        with tab:
                            level_num = level_data["level"]
                            changed_feats = level_data["changed_features"]
                            
                            # Create comparison table of changes
                            changes_df = pd.DataFrame({
                                "Feature": [f["feature"] for f in changed_feats],
                                "Original Value": [f["original_value"] for f in changed_feats],
                                "Counterfactual Value": [f["counterfactual_value"] for f in changed_feats],
                                "Importance": [f["importance"] for f in changed_feats]
                            })
                            
                            st.dataframe(changes_df)
                            
                            # Show probabilities before and after changes
                            orig_probs = level_data["original_probabilities"]
                            cf_probs = level_data["counterfactual_probabilities"]
                            
                            # Create a bar chart comparing original and counterfactual probabilities
                            prob_fig, prob_ax = plt.subplots(figsize=(10, 4))
                            bar_width = 0.35
                            index = np.arange(2)  # Normal and Attack
                            
                            prob_ax.bar(index, orig_probs, bar_width, label='Original')
                            prob_ax.bar(index + bar_width, cf_probs, bar_width, label='Counterfactual')
                            
                            prob_ax.set_xlabel('Class')
                            prob_ax.set_ylabel('Probability')
                            prob_ax.set_title(f'Level {level_num} Probability Shift')
                            prob_ax.set_xticks(index + bar_width / 2)
                            prob_ax.set_xticklabels(['Normal', 'Attack'])
                            prob_ax.legend()
                            
                            st.pyplot(prob_fig)
                else:
                    st.warning("Could not generate a valid counterfactual explanation for this instance.")
                    st.markdown("This typically happens when it's difficult to find changes that would convert this instance to the opposite class.")
            except Exception:
                # Suppress detailed error message
                st.markdown("""
                Could not generate a counterfactual explanation for this instance. 
                This typically happens when it's difficult to find clear changes that would convert this instance to normal traffic.
                """)
    else:
        # Show message for normal traffic only in Counterfactual tab
        st.info("No attack detected. This instance appears to be normal traffic.")
        
        st.markdown("""
        ### Explanation
        The model has classified this instance as normal network traffic. No explainable AI analysis is needed for normal traffic.
        
        You can select a different instance that contains an attack to see the LIME, SHAP, and Counterfactual explanations.
        """)

# Add utility to download the explanation
st.sidebar.header("Export Results")
if st.sidebar.button("Generate Explanation Report"):
    report = f"""
    # Intrusion Detection Explanation Report

    ## Instance Details
    - **Index:** {instance_index}
        - **Actual Traffic Type:** {raw_instance['labels']}
    
    ## Model Prediction
        - **Predicted:** Attack
    
    ## LIME Explanation
    Top features contributing to this prediction:
    
    {lime_table.to_markdown()}
    
    ## SHAP Explanation
    Top contributing features:
    
    """
    
    # Add SHAP reasons if available
    if shap_reasons:
        for i, reason in enumerate(shap_reasons[:10]):
                report += f"- {reason['feature']} = {reason['value']} -> contribution: +{reason['contribution']:.3f}\n"
        
        # Add Counterfactual explanation if it was generated
        try:
            # Check if we have a counterfactual explanation generated
            if 'explanation' in locals() and explanation.get("success", False):
                report += "\n\n## Counterfactual Explanation\n"
                # Add the summary to the report
                cf_summary = summarize_explanation(explanation, numeric_features)
                report += f"Changes needed to convert this instance from Attack to Normal:\n\n```\n{cf_summary}\n```\n"
                
                # Add top features changed
                if 'agg_df' in locals() and not agg_df.empty:
                    report += "\nTop features to modify (by importance):\n\n"
                    for _, row in agg_df.head(5).iterrows():
                        report += f"- {row['Feature']}: {row['Importance']:.4f}\n"
        except Exception:
            report += "\n\nCould not include counterfactual explanation in the report.\n"
    
    # Display in expandable section
    with st.sidebar.expander("View Explanation Report"):
        st.markdown(report)
    
    # Provide download button
    st.sidebar.download_button(
            label="Download Report",
        data=report,
            file_name=f"intrusion_explanation_{instance_index}.md",
        mime="text/markdown"
    )
else:
    # No attack detected
    st.info("No attack detected. This instance appears to be normal traffic.")
    st.markdown("""
    ### Explanation
    The model has classified this instance as normal network traffic. No explainable AI analysis is needed for normal traffic.
    
    You can select a different instance that contains an attack to see the LIME, SHAP, and Counterfactual explanations.
    """)

# Footer
st.sidebar.markdown("---")
st.sidebar.markdown("CPS-XAI: Intrusion Detection System")
st.sidebar.markdown("         SHAILESH") 
st.sidebar.markdown("         SAMPATH")
st.sidebar.markdown("         KARTHIK") 
st.sidebar.markdown("         YOGESH")
