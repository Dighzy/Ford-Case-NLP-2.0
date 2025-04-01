
import sys, os, ast, json

import pandas as pd
import tensorflow as tf
import numpy as np

import keras_tuner as kt

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from data_processing.process_data import get_processed_data

from sklearn.model_selection import train_test_split

from sklearn.utils.class_weight import compute_sample_weight

from tensorflow.keras.layers import Input, Dense, Dropout, BatchNormalization, GaussianNoise
from tensorflow.keras.callbacks import EarlyStopping , ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras import backend as K
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import AUC

#from tensorflow_addons.metrics import MatthewsCorrelationCoefficient, F1Score

def f1_score_manual(y_true, y_pred):
    y_pred = K.round(y_pred)
    tp = K.sum(K.cast(y_true * y_pred, 'float'), axis=0)
    fp = K.sum(K.cast((1 - y_true) * y_pred, 'float'), axis=0)
    fn = K.sum(K.cast(y_true * (1 - y_pred), 'float'), axis=0)

    precision = tp / (tp + fp + K.epsilon())
    recall = tp / (tp + fn + K.epsilon())
    f1 = 2 * precision * recall / (precision + recall + K.epsilon())

    # Média dos F1-scores em cada rótulo (por rótulo, não por instância)
    return K.mean(f1)

def mcc_manual(y_true, y_pred):
    y_pred = K.round(y_pred)
    tp = K.sum(K.cast(y_true * y_pred, 'float'), axis=0)
    tn = K.sum(K.cast((1 - y_true) * (1 - y_pred), 'float'), axis=0)
    fp = K.sum(K.cast((1 - y_true) * y_pred, 'float'), axis=0)
    fn = K.sum(K.cast(y_true * (1 - y_pred), 'float'), axis=0)

    numerator = (tp * tn) - (fp * fn)
    denominator = K.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    mcc = numerator / (denominator + K.epsilon())
    return K.mean(mcc)



def get_model_tuner(X_emb_train: pd.DataFrame, n_components_classes: int, n_type_classes: int, n_cause_classes: int ) -> kt.BayesianOptimization:
    """
    Builds and returns a model tuner for hyperparameter optimization using Bayesian Optimization.
    The model use embeddings for input branches: embeddings from complaint texts and structured features. 
    Bayesian optimization was used to select the best hyperparameters, ensuring an efficient and effective search for optimal model configurations.

    ## Parameters
        X_emb_train (pd.DataFrame): The training dataset containing the text embeddings.
        n_components_classes (int) : The number os categories for the output layer.
        n_type_classes (int) : The number os categories for the output layer.
        n_cause_classes (int) : The number os categories for the output layer.

    """
    # Define the model-building function for the tuner
    def build_model(hp):


        # Add Gaussian noise as a regularization technique
        #x1 = GaussianNoise(hp.Float('noise', 0.005, 0.02))

        # Define input
        embedding_input = Input(shape=(X_emb_train.shape[1],), name="embedding_input")

        # Shared dense branchs
        x = Dense(hp.Int('units_x1', 256, 512, step=64), activation='relu', kernel_regularizer=l2(hp.Choice('l2_x1', [0.00001, 0.0001, 0.001, 0.01])))(embedding_input)
        x = BatchNormalization()(x)
        x = Dropout(hp.Float('dropout_x1', 0.1, 0.3))(x)

        x = Dense(hp.Int('units_x2', 64, 256, step=64), activation='relu', kernel_regularizer=l2(hp.Choice('l2_x2', [0.00001, 0.0001, 0.001])))(x)
        x = BatchNormalization()(x)
        x = Dropout(hp.Float('dropout_x2', 0.0, 0.2))(x)

        x = Dense(hp.Int('units_x3', 32, 128, step=64), activation='relu', kernel_regularizer=l2(hp.Choice('l2_x3', [0.00001, 0.0001, 0.001])))(x)
        x = BatchNormalization()(x)
        x = Dropout(hp.Float('dropout_x3', 0.0, 0.2))(x)

        # Pieces branch
        components_branch = Dense(hp.Int('units_components1', 64, 128, step=64), activation='relu')(x)
        components_branch = Dense(hp.Int('units_components2', 16, 64, step=16), activation='relu')(components_branch)
        components_output = Dense(n_components_classes, activation='sigmoid', name='components_output')(components_branch)

        # Type branch
        type_branch = Dense(hp.Int('units_types1', 64, 128, step=64), activation='relu')(x)
        type_branch = Dense(hp.Int('units_types2', 16, 64, step=16), activation='relu')(type_branch)
        type_output = Dense(n_type_classes, activation='sigmoid', name='type_output')(type_branch)

        # Cause branch
        cause_branch = Dense(hp.Int('units_cause1', 64, 128, step=64), activation='relu')(x)
        cause_branch = Dense(hp.Int('units_cause2', 16, 64, step=16), activation='relu')(cause_branch)
        cause_output = Dense(n_cause_classes, activation='sigmoid', name='cause_output')(cause_branch)

        # Criar modelo com três saídas
        model = Model(inputs=[embedding_input], outputs=[components_output, type_output, cause_output])

        # Compilação
        optimizer = Adam(learning_rate=hp.Choice('learning_rate', [1e-5, 3e-5, 1e-4, 3e-4, 1e-3]))
        model.compile(optimizer=optimizer,
            loss='BinaryCrossentropy',
            metrics= [mcc_manual, AUC(name='pr_auc', curve='PR'), f1_score_manual])
        
        return model

    # Set up the tuner for Bayesian optimization
    tuner = kt.BayesianOptimization(
        build_model,
        objective='val_loss',  
        max_trials=50,  # Number of trials (different configurations)
        executions_per_trial=2,
        directory="models/tuner_results", 
        project_name="bayesian_opt_nlp"  
    )

    return tuner

if __name__ == "__main__":

    np.random.seed(42)
    tf.random.set_seed(42)

    file_path = './data/processed/df_processed.csv'

    if os.path.exists(file_path):
         df_final = pd.read_csv(file_path)
         print('yes')
    else:
        df_final = pd.read_csv('data/raw/full_data_2020_2025_FORD.csv')
        #df_final = df_final.head(1000).copy()
        df_final = get_processed_data(df_final, is_training=True)

    embeddings = np.array(df_final["summary_embedding"].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x).to_list())
    components_binary = np.array(df_final["components_binary"].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x).to_list())
    problems_type_binary = np.array(df_final["problem_type_binary"].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x).to_list())
    problems_cause_binary = np.array(df_final["cause_binary"].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x).to_list())

    X_emb_train, X_emb_val, y_components_train, y_components_val, y_type_train, y_type_val, y_cause_train, y_cause_val = train_test_split(
        embeddings, components_binary, problems_type_binary, problems_cause_binary, test_size=0.3, random_state=42
    )

    tuner = get_model_tuner(X_emb_train, n_components_classes=y_components_train.shape[1], n_type_classes=y_type_train.shape[1], n_cause_classes=y_cause_train.shape[1])

    # Define callbacks
    early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

    best_model_path = "models/best_model.h5"
    model_checkpoint = ModelCheckpoint(filepath=best_model_path, monitor='val_loss', save_best_only=True, mode='min', verbose=1)

    #lr_scheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.25, patience=3, min_lr=1e-6)

    # Start searching for the best hyperparameters
    tuner.search(
        {"embedding_input": X_emb_train},  
        {"components_output": y_components_train, "type_output": y_type_train, "cause_output": y_cause_train},
        validation_data=(
            {"embedding_input": X_emb_val},
            {"components_output": y_components_val, "type_output": y_type_val, "cause_output": y_cause_val}
        ),
        epochs=25,
        batch_size=128,
        callbacks=[early_stopping]
    )

    # Get the best hyperparameters from the search
    best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
    model = tuner.hypermodel.build(best_hps)
    best_hps_dict = {
    "Best tuner hyperparameters": best_hps.values,
    }

    hps_file = "./models/params/best_hyperparameters.json"
    with open(hps_file, 'w') as f:
        json.dump(best_hps_dict, f, indent=4)

    print(f"Best hyperparameters saved to {hps_file}")

    early_stopping = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)

    history = model.fit(
        {"embedding_input": X_emb_train},
        {"components_output": y_components_train, "type_output": y_type_train, "cause_output": y_cause_train},
        validation_data=(
            {"embedding_input": X_emb_val},
            {"components_output": y_components_val, "type_output": y_type_val, "cause_output": y_cause_val}
        ),
        epochs=500,
        batch_size=128,
        callbacks=[early_stopping, model_checkpoint]
    )
   
    best_epoch = np.argmin(history.history['val_loss'])
    metrics_dict = {
    "Training and Validation Metrics": {metric: float(values[best_epoch]) for metric, values in history.history.items()},
    }

    metrics_file = "./models/params/model_metrics.json"
    with open(metrics_file, 'w') as f:
        json.dump(metrics_dict, f, indent=4)

    print(f"Metrics saved to {metrics_file}")


