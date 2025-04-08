import pandas as pd
import keras_tuner as kt
import tensorflow as tf

from tensorflow.keras.layers import Input, Dense, Dropout, BatchNormalization
from tensorflow.keras import backend as K
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import AUC

from hydra.utils import to_absolute_path

def f1_score_manual(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
    """
    Computes the F1-score manually for binary classification.
    
    Parameters:
    y_true (K.Tensor): Ground truth labels.
    y_pred (K.Tensor): Predicted labels.
    """
    y_pred = K.round(y_pred)
    tp = K.sum(K.cast(y_true * y_pred, 'float'), axis=0)
    fp = K.sum(K.cast((1 - y_true) * y_pred, 'float'), axis=0)
    fn = K.sum(K.cast(y_true * (1 - y_pred), 'float'), axis=0)

    precision = tp / (tp + fp + K.epsilon())
    recall = tp / (tp + fn + K.epsilon())
    f1 = 2 * precision * recall / (precision + recall + K.epsilon())

    return K.mean(f1)

def mcc_manual(y_true:tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
    """
    Computes the Matthews correlation coefficient (MCC) manually.
    
    Parameters:
    y_true (K.Tensor): Ground truth labels.
    y_pred (K.Tensor): Predicted labels.
    """
    y_pred = K.round(y_pred)
    tp = K.sum(K.cast(y_true * y_pred, 'float'), axis=0)
    tn = K.sum(K.cast((1 - y_true) * (1 - y_pred), 'float'), axis=0)
    fp = K.sum(K.cast((1 - y_true) * y_pred, 'float'), axis=0)
    fn = K.sum(K.cast(y_true * (1 - y_pred), 'float'), axis=0)

    numerator = (tp * tn) - (fp * fn)
    denominator = K.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    mcc = numerator / (denominator + K.epsilon())
    return K.mean(mcc)

def get_model_tuner(tuning_dict: dict, X_emb_train: pd.DataFrame, n_components_classes: int, n_type_classes: int, n_cause_classes: int ) -> kt.BayesianOptimization:
    """
    Builds and returns a model tuner for hyperparameter optimization using Bayesian Optimization.
    Loads hyperparameter search ranges from a YAML configuration file.
    """
    
    def build_model(hp):
        embedding_input = Input(shape=(X_emb_train.shape[1],), name="embedding_input")

        x = Dense(hp.Int('units_x1', **tuning_dict["units_x1"]), activation='relu',
                  kernel_regularizer=l2(hp.Choice('l2_x1', tuning_dict["l2_x1"]))) (embedding_input)
        x = BatchNormalization()(x)
        x = Dropout(hp.Float('dropout_x1', **tuning_dict["dropout_x1"])) (x)

        x = Dense(hp.Int('units_x2', **tuning_dict["units_x2"]), activation='relu',
                  kernel_regularizer=l2(hp.Choice('l2_x2', tuning_dict["l2_x2"]))) (x)
        x = BatchNormalization()(x)
        x = Dropout(hp.Float('dropout_x2', **tuning_dict["dropout_x2"])) (x)

        x = Dense(hp.Int('units_x3', **tuning_dict["units_x3"]), activation='relu',
                  kernel_regularizer=l2(hp.Choice('l2_x3', tuning_dict["l2_x3"]))) (x)
        x = BatchNormalization()(x)
        x = Dropout(hp.Float('dropout_x3', **tuning_dict["dropout_x3"])) (x)

        # Components branch
        components_branch = Dense(hp.Int('units_components1', **tuning_dict["units_components1"]), activation='relu')(x)
        components_branch = Dense(hp.Int('units_components2', **tuning_dict["units_components2"]), activation='relu')(components_branch)
        components_output = Dense(n_components_classes, activation='sigmoid', name='components_output')(components_branch)

        # Type branch
        type_branch = Dense(hp.Int('units_types1', **tuning_dict["units_types1"]), activation='relu')(x)
        type_branch = Dense(hp.Int('units_types2', **tuning_dict["units_types2"]), activation='relu')(type_branch)
        type_output = Dense(n_type_classes, activation='sigmoid', name='type_output')(type_branch)

        # Cause branch
        cause_branch = Dense(hp.Int('units_cause1', **tuning_dict["units_cause1"]), activation='relu')(x)
        cause_branch = Dense(hp.Int('units_cause2', **tuning_dict["units_cause2"]), activation='relu')(cause_branch)
        cause_output = Dense(n_cause_classes, activation='sigmoid', name='cause_output')(cause_branch)

        model = Model(inputs=[embedding_input], outputs=[components_output, type_output, cause_output])
        optimizer = Adam(learning_rate=hp.Choice('learning_rate', tuning_dict["learning_rate"]))
        model.compile(optimizer=optimizer, loss='BinaryCrossentropy', metrics=[mcc_manual, AUC(name='pr_auc', curve='PR'), f1_score_manual])
        return model

    tuner = kt.BayesianOptimization(
        build_model,
        objective='val_loss',
        max_trials=tuning_dict["max_trials"],
        executions_per_trial=tuning_dict["executions_per_trial"],
        directory=to_absolute_path(tuning_dict["directory"]),
        project_name=tuning_dict["project_name"]
    )

    return tuner
