from common import *
from visualization import add_predictions, plot_confusion_matrix

def train_evaluate_models(binary_train_X, binary_train_y):
    """Train and evaluate multiple classification models"""
    models = [
        LogisticRegression(max_iter=250),
        DecisionTreeClassifier(max_depth=10),
        RandomForestClassifier(),
        KNeighborsClassifier(),
        GaussianNB()
    ]
    
    model_comps = []
    for model in models:
        model_name = model.__class__.__name__
        accuracies = cross_val_score(model, binary_train_X, binary_train_y, scoring='accuracy')
        for count, accuracy in enumerate(accuracies):
            model_comps.append((model_name, count, accuracy))
    
    # Create DataFrame with results
    result_df = pd.DataFrame(model_comps, columns=['model_name', 'count', 'accuracy'])
    
    # Plot box plot of accuracies
    print("Box Plot of Accuracy Scores")
    result_df.pivot(index='count', columns='model_name', values='accuracy').boxplot(rot=45)
    plt.show()
    
    # Compute mean accuracies
    result_df = result_df.groupby('model_name').mean()
    
    return result_df

def train_decision_tree(binary_train_X, binary_train_y, binary_val_X, binary_val_y):
    """Train a Decision Tree classifier and evaluate its performance"""
    clf = DecisionTreeClassifier()  # max_depth=10
    clf = clf.fit(binary_train_X, binary_train_y)
    
    # Evaluate on training data
    train_acc = clf.score(binary_train_X, binary_train_y)
    print(f"Training accuracy is: {train_acc*100:.2f}%")
    
    # Evaluate on validation data
    clf_predictions = clf.predict(binary_val_X)
    base_clf_score = accuracy_score(clf_predictions, binary_val_y)
    print(f"Testing accuracy is: {base_clf_score*100:.2f}%")
    
    # Calculate performance metrics
    Tree_f1 = f1_score(binary_val_y, clf_predictions, average="macro")
    Tree_precision = precision_score(binary_val_y, clf_predictions, average="macro")
    Tree_recall = recall_score(binary_val_y, clf_predictions, average="macro")
    Tree_accuracy = accuracy_score(binary_val_y, clf_predictions)
    
    cm_dtree = confusion_matrix(binary_val_y, clf_predictions)
    print('Confusion Matrix: ')
    print(cm_dtree)
    print(f"F1 Score: {Tree_f1:.4f}")
    print(f"Precision Score: {Tree_precision:.4f}")
    print(f"Recall Score: {Tree_recall:.4f}")
    print(f"Accuracy Score: {Tree_accuracy:.4f}")
    
    # Plot confusion matrix
    plot_confusion_matrix(binary_val_y, clf_predictions)
    
    # Analyze predictions
    binary_prediction_data = add_predictions(
        binary_val_X.reset_index().drop('index', axis=1), 
        clf_predictions, 
        binary_val_y)
    
    return clf, binary_prediction_data

def train_model_for_xai(x, y):
    """Train a model for XAI (LIME/SHAP) visualization"""
    try:
        # Ensure data is clean before splitting
        x = x.fillna(0)  # Replace NaN values with 0
        
        # Convert any remaining non-numeric values (just in case)
        for col in x.select_dtypes(include=['object']).columns:
            x[col] = pd.to_numeric(x[col], errors='coerce').fillna(0)
        
        # Split data
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
        
        # Train the model
        model = DecisionTreeClassifier(random_state=18)
        model.fit(x_train, y_train)
        
        return model, x_train, x_test, y_train, y_test
    
    except Exception as e:
        print(f"Error in train_model_for_xai: {str(e)}")
        # In case of error, return a very simple model with placeholder data
        model = DecisionTreeClassifier(random_state=18)
        sample_size = min(1000, len(x))
        x_sample = x.iloc[:sample_size].fillna(0)
        y_sample = y.iloc[:sample_size]
        
        x_train, x_test, y_train, y_test = train_test_split(x_sample, y_sample, test_size=0.2)
        model.fit(x_train, y_train)
        
        return model, x_train, x_test, y_train, y_test 