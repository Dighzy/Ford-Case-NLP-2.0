
import pandas as pd
import tensorflow as tf
import numpy as np
import json

import keras_tuner as kt

from process_data import DataProcessor, FeatureExtractor, TopicsProcessor, TextProcessor

from sklearn.model_selection import train_test_split

from tensorflow.keras.layers import Input, Dense, Concatenate, Dropout, BatchNormalization, GaussianNoise
from tensorflow.keras.callbacks import EarlyStopping , ModelCheckpoint, ReduceLROnPlateau

from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import Adam

def get_model_tuner(X_emb_train: pd.DataFrame, X_add_train: pd.DataFrame, n_categories: int = 7) -> kt.BayesianOptimization:
    """
    Builds and returns a model tuner for hyperparameter optimization using Bayesian Optimization.

    The model uses two input branches: embeddings from complaint texts and structured features. 
    Bayesian optimization was used to select the best hyperparameters, ensuring an efficient and effective search for optimal model configurations.

    ## Parameters
        X_emb_train (pd.DataFrame): The training dataset containing the text embeddings.
        X_add_train (pd.DataFrame): The training dataset containing additional structured features.
        n_categories (int) : The number os categories for the output layer. 

    """
    # Define the model-building function for the tuner
    def build_model(hp):
        # Define inputs
        embedding_input = Input(shape=(X_emb_train.shape[1],), name="embedding_input")
        additional_input = Input(shape=(X_add_train.shape[1],), name="additional_input")

        # Add Gaussian noise as a regularization technique
        x1 = GaussianNoise(hp.Float('noise', 0.005, 0.02))(embedding_input)

        # Dense layer for embeddings (input from X_emb_train)
        x1 = Dense(hp.Int('units_x1', 64, 256, step=32), activation='relu', kernel_regularizer=l2(hp.Choice('l2_x1', [0.001, 0.01, 0.1])))(x1)
        x1 = BatchNormalization()(x1)
        x1 = Dropout(hp.Float('dropout_x1', 0.2, 0.5))(x1)

        # Dense layer for additional features (input from X_add_train)
        x2 = Dense(hp.Int('units_x2', 32, 128, step=16), activation='relu', kernel_regularizer=l2(hp.Choice('l2_x2', [0.001, 0.01, 0.1])))(additional_input)
        x2 = BatchNormalization()(x2)
        x2 = Dropout(hp.Float('dropout_x2', 0.2, 0.5))(x2)

        # Concatenate the two blocks of layers
        combined = Concatenate()([x1, x2])
        combined = Dense(hp.Int('units_combined', 16, 64, step=16), activation='relu', kernel_regularizer=l2(hp.Choice('l2_combined', [0.001, 0.01, 0.1])))(combined)
        combined = BatchNormalization()(combined)
        combined = Dropout(hp.Float('dropout_combined', 0.2, 0.5))(combined)

        # Output layer for multi-label classification (sigmoid activation for each label)
        output = Dense(n_categories, activation='sigmoid')(combined)

        # Create the model
        model = Model(inputs=[embedding_input, additional_input], outputs=output)

        # Compile the model with Adam optimizer and binary cross-entropy loss for multi-label classification
        optimizer = Adam(learning_rate=hp.Float('learning_rate', 1e-5, 1e-3, sampling='log'))
        model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy', 'AUC', 'Recall', 'Precision'])

        return model

    # Set up the tuner for Bayesian optimization
    tuner = kt.BayesianOptimization(
        build_model,
        objective='val_loss',  
        max_trials=20,  # Number of trials (different configurations)
        executions_per_trial=1,
        directory="models/tuner_results", 
        project_name="bayesian_opt_nlp"  
    )

    return tuner

def set_test_df(X_emb_test: np.ndarray, X_add_test: np.ndarray, y_test: np.ndarray) -> None:
    """
    Prepares the test dataset and saves it as a CSV file.
    
    ## Parameters
        X_emb_test (np.ndarray): The test set embeddings.
        X_add_test (np.ndarray): The test set additional features.
        y_test (np.ndarray): The test set labels (multi-label).

    """
    # Convert arrays to DataFrames
    df_emb_test = pd.DataFrame(X_emb_test)
    df_add_test = pd.DataFrame(X_add_test)

    # Convert multi-label `y_test` into lists (each row is a list of labels)
    y_test_lists = [list(row) for row in y_test]

    # Combine features and labels into a single DataFrame
    df_test_final = pd.DataFrame({
        'embeddings': df_emb_test.values.tolist(),  # Convert rows of feature data to  list
        'additional_features': df_add_test.values.tolist(),  # Convert additional features to list
        'category_binary': y_test_lists  # Store labels as lists
    })

    # Save to CSV
    df_test_final.to_csv("data/processed/test_data.csv", index=False)


if __name__ == "__main__":
    """ Seeding """
    np.random.seed(42)
    tf.random.set_seed(42)

    df_final = pd.read_csv('data/raw/full_data_2020_2025_FORD.csv')
    #df_final = df_final.head(100).copy()

    # Getting classes
    text_processor = TextProcessor()
    feature_extractor = FeatureExtractor()
    topic_processor = TopicsProcessor()
    data_processor = DataProcessor(is_training=True)


    # Getting model types
    models = df_final['Model'].unique().tolist()
    model_data = {
        "models":models
    }
    with open('models/params/models_params.json', 'w') as f:
        json.dump(model_data, f)
    
    # Text preprocessing function
    df_final['processed_summary'] = df_final['summary'].apply(text_processor.preprocess_text)

    df_final = feature_extractor.get_categories(df_final)
    df_final = feature_extractor.transform_categories(df_final, is_training=True)

    # Get embeddings
    df_final['summary_embedding'] = df_final['summary'].apply(feature_extractor.get_model_embedding)
    df_final['model_embedding'] = df_final['Model'].apply(feature_extractor.get_model_embedding)

    # Get sentiment and topics
    df_final = feature_extractor.get_sentiment_and_count(df_final)
    df_final = topic_processor.get_topics(df_final,is_training=True)

    # Processing and normalazing the embedding and additional_features
    embeddings = np.array(df_final["summary_embedding"].to_list())
    category_binary = np.array(df_final["category_binary"].to_list())

    # Saving my final data to use in the model
    df_salvar = df_final.copy()
    df_salvar['summary_embedding'] = df_salvar['summary_embedding'].apply(lambda x: json.dumps(x.tolist()) if isinstance(x, np.ndarray) else x)
    df_salvar['model_embedding'] = df_salvar['model_embedding'].apply(lambda x: json.dumps(x.tolist()) if isinstance(x, np.ndarray) else x)
    df_salvar['category_binary'] = df_salvar['category_binary'].apply(lambda x: json.dumps(x.tolist()) if isinstance(x, np.ndarray) else x)
    df_salvar[['processed_summary', 'summary_embedding', 'model_embedding', 'word_count', 'char_count',
        'sentiment', 'category_binary']].to_csv('./data/processed/df_final_model.csv',index=False)

    additional_features= data_processor.process_train_data(df_final)

     # Split data into train and test sets
    X_emb_train, X_emb_val, X_add_train, X_add_val, y_train, y_val = train_test_split(
        embeddings, additional_features, category_binary, test_size=0.3, random_state=42
    )

    # Take the first 10 rows from the validation set for the test set
    X_emb_test = X_emb_val[:10]
    X_add_test = X_add_val[:10]
    y_test = y_val[:10]

    # Remove those 10 rows from the validation set
    X_emb_val = X_emb_val[10:]
    X_add_val = X_add_val[10:]
    y_val = y_val[10:]

    # Verify the shapes of the data
    print("X_emb_train shape:", X_emb_train.shape)  # Should match input shape for embedding input
    print("X_add_train shape:", X_add_train.shape)  # Should match input shape for additional input
    print("y_train shape:", y_train.shape)       # Should be (num_samples, 5) if 5 labels are expected

    set_test_df(X_emb_test, X_add_test, y_test)
    
    # Defining the model

    tuner = get_model_tuner(X_emb_train, X_add_train, n_categories=y_train.shape[1])

    # Define callbacks
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    best_model_path = "models/best_model.h5"
    model_checkpoint = ModelCheckpoint(filepath=best_model_path, monitor='val_loss', save_best_only=True, mode='min', verbose=1)
    lr_scheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.25, patience=3, min_lr=1e-6)

    # Start searching for the best hyperparameters
    tuner.search(
        {"embedding_input": X_emb_train, "additional_input": X_add_train},
        y_train,
        validation_data=({"embedding_input": X_emb_val, "additional_input": X_add_val}, y_val),
        epochs=50,
        batch_size=128,
        callbacks=[early_stopping, model_checkpoint, lr_scheduler]
    )

    # Get the best hyperparameters from the search
    best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
    print(f"Best hyperparameters found: {best_hps.values}")

    # Train the final model using the best hyperparameters
    model = tuner.hypermodel.build(best_hps)
    history = model.fit(
        {"embedding_input": X_emb_train, "additional_input": X_add_train},
        y_train,
        validation_data=({"embedding_input": X_emb_val, "additional_input": X_add_val}, y_val),
        epochs=200,  
        batch_size=128,
        callbacks=[early_stopping, model_checkpoint, lr_scheduler]
    )


