from common import *

def preprocess_data(df, test_df):
    """Preprocess the data for model training and testing"""
    # Create attack flags (binary classification)
    is_attack = df.attack.map(lambda a: 0 if a == 'normal' else 1)
    test_attack = test_df.attack.map(lambda a: 0 if a == 'normal' else 1)
    
    df['attack_flag'] = is_attack
    test_df['attack_flag'] = test_attack
    
    # Map attacks to their categories
    attack_map = df.attack.apply(map_attack)
    df['attack_map'] = attack_map
    test_attack_map = test_df.attack.apply(map_attack)
    test_df['attack_map'] = test_attack_map
    
    return df, test_df

def encode_categorical_features(df, test_df):
    """Encode categorical features for model training"""
    features_to_encode = ['protocol_type', 'service', 'flag']
    encoded = pd.get_dummies(df[features_to_encode])
    
    # Handle potential differences in categorical values between train and test
    test_encoded_base = pd.get_dummies(test_df[features_to_encode])
    test_index = np.arange(len(test_df.index))
    column_diffs = list(set(encoded.columns.values)-set(test_encoded_base.columns.values))
    diff_df = pd.DataFrame(0, index=test_index, columns=column_diffs)
    column_order = encoded.columns.to_list()
    test_encoded_temp = test_encoded_base.join(diff_df)
    test_final = test_encoded_temp[column_order].fillna(0)
    
    # Add numeric features
    numeric_features = ['duration', 'src_bytes', 'dst_bytes']
    to_fit = encoded.join(df[numeric_features])
    test_set = test_final.join(test_df[numeric_features])
    
    return to_fit, test_set

def prepare_train_test_data(df, to_fit):
    """Split data into training and validation sets"""
    binary_y = df['attack_flag']
    multi_y = df['attack_map']
    
    binary_train_X, binary_val_X, binary_train_y, binary_val_y = train_test_split(
        to_fit, binary_y, test_size=0.6)
    
    multi_train_X, multi_val_X, multi_train_y, multi_val_y = train_test_split(
        to_fit, multi_y, test_size=0.6)
    
    return {
        'binary': (binary_train_X, binary_val_X, binary_train_y, binary_val_y),
        'multi': (multi_train_X, multi_val_X, multi_train_y, multi_val_y)
    }

def encode_all_categorical(df):
    """Encode all categorical columns for models like LIME and SHAP"""
    # Make a copy to avoid modifying the original dataframe
    df_encoded = df.copy()
    
    # Find categorical columns
    catCols = df_encoded.select_dtypes(include="object").columns
    
    # Encode each categorical column
    le = LabelEncoder()
    for feat in catCols:
        try:
            df_encoded[feat] = le.fit_transform(df_encoded[feat].astype(str))
        except Exception as e:
            print(f"Error encoding column {feat}: {str(e)}")
            # If there's an error, just convert to string and continue
            df_encoded[feat] = df_encoded[feat].astype(str)
    
    return df_encoded 