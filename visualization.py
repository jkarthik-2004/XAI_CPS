from common import *

def bake_pies(data_list, labels):
    """Create multiple pie charts for visualization"""
    list_length = len(data_list)

    color_list = sns.color_palette()
    color_cycle = itertools.cycle(color_list)
    cdict = {}

    fig, axs = plt.subplots(1, list_length, figsize=(18, 10), tight_layout=False)
    plt.subplots_adjust(wspace=1/list_length)

    for count, data_set in enumerate(data_list):
        for num, value in enumerate(np.unique(data_set.index)):
            if value not in cdict:
                cdict[value] = next(color_cycle)
        wedges, texts = axs[count].pie(data_set,
                           colors=[cdict[v] for v in data_set.index])
        axs[count].legend(wedges, data_set.index,
                           title="Flags",
                           loc="center left",
                           bbox_to_anchor=(1, 0, 0.5, 1))
        axs[count].set_title(labels[count])
    return axs

def visualize_protocol_attacks(df):
    """Visualize attacks by protocol type"""
    attack_vs_protocol = pd.crosstab(df.attack, df.protocol_type)
    
    icmp_attacks = attack_vs_protocol.icmp
    tcp_attacks = attack_vs_protocol.tcp
    udp_attacks = attack_vs_protocol.udp
    
    bake_pies([icmp_attacks, tcp_attacks, udp_attacks], ['icmp', 'tcp', 'udp'])
    plt.show()

def visualize_flag_distribution(df):
    """Visualize flag distribution between normal and attack traffic"""
    normal_flags = df.loc[df.attack_flag == 0].flag.value_counts()
    attack_flags = df.loc[df.attack_flag == 1].flag.value_counts()
    
    flag_axs = bake_pies([normal_flags, attack_flags], ['normal', 'attack'])
    plt.show()

def plot_model_comparison(result_df, recalls):
    """Plot model performance comparison"""
    models = np.array(['RandomForest', 'DTree', 'KNN', 'LogReg', 'GaussianNB'])
    accuracies = result_df['accuracy'].values
    accuracies = sorted(accuracies, reverse=True)
    
    X_axis = np.arange(len(models))

    plt.bar(X_axis - 0.2, accuracies, 0.4, label='Accuracy')
    plt.bar(X_axis + 0.2, recalls, 0.4, label='Recall')

    plt.xticks(X_axis, models)
    plt.xlabel("Machine Learning Models")
    plt.ylabel("Score")
    plt.title("Performance Evaluation Metrics")
    plt.legend()
    plt.show()

def plot_confusion_matrix(y_true, y_pred):
    """Create and plot confusion matrix for binary classification"""
    cm = confusion_matrix(y_true, y_pred)
    
    sns.heatmap(data=cm,
                xticklabels=['Predicted Normal', 'Predicted Attack'],
                yticklabels=['Actual Normal', 'Actual Attack'],
                cmap="YlGnBu",
                fmt='d',
                annot=True)
    plt.show()

def add_predictions(data_set, predictions, y):
    """Add predictions to dataset and analyze errors"""
    prediction_series = pd.Series(predictions, index=y.index)

    predicted_vs_actual = data_set.assign(predicted=prediction_series)
    original_data = predicted_vs_actual.assign(actual=y).dropna()
    conf_matrix = confusion_matrix(original_data['actual'],
                                  original_data['predicted'])

    # capture rows with failed predictions
    base_errors = original_data[original_data['actual'] != original_data['predicted']]

    # drop columns with no value
    non_zeros = base_errors.loc[:, (base_errors != 0).any(axis=0)]

    # identify the type of error
    false_positives = non_zeros.loc[non_zeros.actual==0]
    false_negatives = non_zeros.loc[non_zeros.actual==1]

    # put everything into an object
    prediction_data = {'data': original_data,
                      'confusion_matrix': conf_matrix,
                      'errors': base_errors,
                      'non_zeros': non_zeros,
                      'false_positives': false_positives,
                      'false_negatives': false_negatives}
    return prediction_data 