
import pandas as pd
import tensorflow as tf
import numpy as np
import ast

from tensorflow.keras.models import load_model

if __name__ == "__main__":

    """ Seeding """
    np.random.seed(42)
    tf.random.set_seed(42)

    #Loading the test df
    df_final = pd.read_csv('data/processed/test_data.csv')

    # Convert the object columns to a list
    df_final["embeddings"] = df_final["embeddings"].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)
    df_final["additional_features"] = df_final["additional_features"].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)
    df_final['category_binary'] = df_final['category_binary'].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)

    #Convert columns to NumPy arrays
    embeddings = np.array(df_final['embeddings'].to_list())
    additional_features =  np.array(df_final['additional_features'].to_list())

    #Loading the model
    custom_model = load_model('models/best_bayesian_model.h5')

    # Make predictions
    prediction = custom_model.predict({
        "embedding_input": embeddings,
        "additional_input": additional_features
    })

   # Setting the threshold for predictions
    threshold = 0.5
    binary_predictions = (prediction >= threshold).astype(int)

    # Formating binary predictions 
    categories = ["Engine and Performance", "Electrical System", "Other Problems", "Structure and Control", "Safety and Brakes"]
    formatted_predictions = []
    formatted_category = []

    # For each element in the row, check where the value is 1
    # Append the list of categories to the 'formatted' list
    for row in binary_predictions:
        predicted_categories = [categories[i] for i, c in enumerate(row) if c == 1]
        formatted_predictions.append(predicted_categories)

    for row in df_final['category_binary']:
        binary_categories = [categories[i] for i, c in enumerate(row) if c == 1]
        formatted_category.append(binary_categories)
        
    # Setting the Final Df
    df_final['binary_predictions'] = [list(row) for row in binary_predictions]
    df_final['predictions'] = formatted_predictions
    df_final['category'] =  formatted_category
    df_final = df_final[['category_binary', 'category','binary_predictions','predictions']]

    # Print the Df
    print(df_final)

    # Initialize counter for correct predictions
    correct_predictions = 0

    # Iterate over the predictions and actual labels to check how many match
    for pred, true in zip(df_final['predictions'], df_final['category']):
        if set(pred) == set(true):  # Check if the predicted categories match the true categories
            correct_predictions += 1

    # Calculate accuracy 
    accuracy = (correct_predictions / len(df_final)) * 100

    # Print Accuracy
    print(f"Accuracy: {accuracy:.2f}%")

