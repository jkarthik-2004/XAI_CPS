from common import *
import lime
from lime import lime_tabular
import warnings

def create_lime_explainer(x_train, attacks):
    """Create a LIME explainer for the model"""
    explainer = lime_tabular.LimeTabularExplainer(
        training_data=np.array(x_train),
        feature_names=x_train.columns,
        class_names=attacks,
        mode='classification'
    )
    return explainer

def explain_instances(explainer, model, x_train, x_test, num_instances=4):
    """Generate explanations for training and test instances"""
    warnings.filterwarnings("ignore", category=UserWarning)
    
    # Explain training instances
    print("Explaining training instances:")
    for i in range(1, num_instances+1):
        exp = explainer.explain_instance(
            data_row=x_train.iloc[i],
            predict_fn=model.predict_proba
        )
        exp.show_in_notebook(show_table=True)
    
    # Explain test instances
    print("\nExplaining test instances:")
    for i in range(1, num_instances+1):
        exp = explainer.explain_instance(
            data_row=x_test.iloc[i],
            predict_fn=model.predict_proba
        )
        print(f"Instance {i} Explanation:")
        print(exp.as_list())  # Get feature importance for this prediction
        exp.show_in_notebook(show_table=True)

def run_lime_analysis():
    """Run LIME analysis on the intrusion detection model"""
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
    
    # Create LIME explainer
    explainer = create_lime_explainer(x_train, attacks)
    
    # Explain instances
    explain_instances(explainer, model, x_train, x_test) 