from common import *
from preprocessing import preprocess_data, encode_categorical_features, prepare_train_test_data
from visualization import visualize_protocol_attacks, visualize_flag_distribution, plot_model_comparison
from model_training import train_evaluate_models, train_decision_tree
from lime_explainer import run_lime_analysis
from shap_explainer import run_shap_analysis

def main():
    """Main function to run the entire analysis pipeline"""
    # Load data
    print("Loading data...")
    df, test_df = load_data()
    
    # Preprocess data
    print("Preprocessing data...")
    df, test_df = preprocess_data(df, test_df)
    
    # Visualize data
    print("\nVisualizing attacks by protocol...")
    visualize_protocol_attacks(df)
    
    print("\nVisualizing flag distribution...")
    visualize_flag_distribution(df)
    
    # Encode features
    print("\nEncoding categorical features...")
    to_fit, test_set = encode_categorical_features(df, test_df)
    
    # Prepare training and validation data
    data_splits = prepare_train_test_data(df, to_fit)
    binary_train_X, binary_val_X, binary_train_y, binary_val_y = data_splits['binary']
    
    # Train and evaluate models
    print("\nTraining and evaluating models...")
    result_df = train_evaluate_models(binary_train_X, binary_train_y)
    
    # Use pre-defined recalls from the original code
    recalls = np.array([0.87238, 0.97812, 0.90601, 0.67129, 0.46921])
    
    # Plot model comparison
    print("\nPlotting model comparison...")
    plot_model_comparison(result_df, recalls)
    
    # Train and evaluate decision tree
    print("\nTraining and evaluating decision tree...")
    clf, binary_prediction_data = train_decision_tree(
        binary_train_X, binary_train_y, binary_val_X, binary_val_y)
    
    # Choose which XAI method to run
    run_xai = input("\nChoose XAI method to run (lime/shap/both/none): ").lower()
    
    if run_xai == 'lime' or run_xai == 'both':
        print("\nRunning LIME analysis...")
        run_lime_analysis()
    
    if run_xai == 'shap' or run_xai == 'both':
        print("\nRunning SHAP analysis...")
        run_shap_analysis()
    
    print("\nAnalysis complete!")

if __name__ == "__main__":
    main() 