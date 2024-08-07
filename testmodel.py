import pickle
import numpy as np
import pandas as pd

# User-defined prediction functionpassword123
def predict():
    password = input("Enter a password: ")
    
    # Load the saved model
    with open('password_strength_model.pkl', 'rb') as model_file:
        clf = pickle.load(model_file)
        
    with open('vectorizer.pkl', 'rb') as vec_file:
        vectorizer = pickle.load(vec_file)
        
    sample_array = np.array([password])
    sample_matrix = vectorizer.transform(sample_array) 
    
    length_pass = len(password)
    length_normalised_lowercase = len([char for char in password if char.islower()]) / len(password)
    length_normalised_uppercase = len([char for char in password if char.isupper()]) / len(password)
    length_normalised_digit = len([char for char in password if char.isdigit()]) / len(password)
    length_normalised_special = len([char for char in password if not char.isalpha() and not char.isdigit()]) / len(password)
    
    new_features = np.append(sample_matrix.toarray(), [length_pass, length_normalised_lowercase, length_normalised_uppercase, length_normalised_digit, length_normalised_special])
    
    # Create a DataFrame with the correct feature names
    feature_names = list(vectorizer.get_feature_names_out()) + ['length', 'lowercase_freq', 'uppercase_freq', 'digit_freq', 'special_char_freq']
    new_matrix2 = pd.DataFrame([new_features], columns=feature_names)
    
    result = clf.predict(new_matrix2)
    
    if result == 0:
        return "Password is weak"
    elif result == 1:
        return "Password is normal"
    else:
        return "Password is strong"

print(predict())