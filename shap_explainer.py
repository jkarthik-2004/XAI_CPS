from common import *
import shap

def create_shap_explainer(model, x_test):
    """Create a SHAP explainer for the model"""
    explainer = shap.Explainer(model.predict, x_test)
    shap_values = explainer(x_test)
    return explainer, shap_values

def generate_shap_visualizations(shap_values, x_test):
    """Generate various SHAP visualizations"""
    # Get expected value correctly
    expected_value = shap_values.base_values.mean()  # Approximate expected value
    
    # Initialize JavaScript for visualization
    shap.initjs()
    
    # Force plot for a single instance
    instance_index = 0  # Change this for different instances
    shap.force_plot(expected_value, shap_values[instance_index].values, x_test.iloc[instance_index])
    
    # Bar Plot (global feature importance)
    shap.plots.bar(shap_values)
    
    # Summary Plot (Beeswarm)
    shap.plots.beeswarm(shap_values)
    
    # Summary Plot (Violin)
    shap.summary_plot(shap_values, plot_type='violin')
    
    # Local Bar Plot for first instance
    shap.plots.bar(shap_values[0])
    
    # Waterfall Plot for first instance
    shap.plots.waterfall(shap_values[0])

def detect_intrusion(model, explainer, new_data_row, attacks):
    """Detect intrusions with SHAP explanations"""
    # Predict
    prediction_index = model.predict(new_data_row)[0]
    prediction_label = attacks[prediction_index]  # Get actual attack name
    probability = model.predict_proba(new_data_row)[0]

    # Map attack to category
    attack_type_code = map_attack(prediction_label)
    attack_type = attack_labels[attack_type_code]

    # Explain
    shap_values = explainer(pd.DataFrame(new_data_row))
    shap.plots.waterfall(shap_values[0])

    # Get contributing reasons
    reasons = []
    feature_names = new_data_row.columns
    instance_vals = new_data_row.iloc[0]
    shap_contribs = shap_values.values[0]

    for feat, val, contrib in zip(feature_names, instance_vals, shap_contribs):
        if contrib > 0:
            reasons.append(f"{feat} = {val} → contributed to classification (+{contrib:.3f})")

    return prediction_label, attack_type, probability, reasons

def run_shap_analysis():
    """Run SHAP analysis on the intrusion detection model"""
    # Load data
    data = pd.read_csv("KDDTrain.csv")
    data.columns = columns
    data.drop(['level'], axis=1, inplace=True)
    
    # Get unique attacks
    attacks = data['attack'].unique()
    
    # Encode categorical features
    data = encode_all_categorical(data)
    
    # Split features and target
    x = data.iloc[:, :-1]
    y = data.iloc[:, -1]
    
    # Train model for XAI
    model, x_train, x_test, y_train, y_test = train_model_for_xai(x, y)
    
    # Create SHAP explainer
    explainer, shap_values = create_shap_explainer(model, x_test)
    
    # Generate SHAP visualizations
    generate_shap_visualizations(shap_values, x_test)
    
    # Test intrusion detection with explanations
    for idx in range(100, 104):
        new_data_row = x_test.iloc[idx:idx+1]
        
        pred_label, pred_type, probs, reasons = detect_intrusion(
            model, explainer, new_data_row, attacks)
        
        print(f"\n🔍 Instance {idx}")
        print(f"Predicted Attack: {pred_label}")
        print(f"Attack Type: {pred_type}")
        print("Probabilities:", probs)
        print("Reasons for classification:")
        for reason in reasons:
            print("-", reason) 