import re
import pandas as pd
import joblib
from openai import OpenAI

def generate_hs_code(item_description, model_inp):
    # Initialize the OpenAI client
    api_key = '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'
    openai_client = OpenAI(api_key=api_key)

    # Create a prompt using the provided item description
    prompt = f"Generate the 8-digit HS code for an item described as: {item_description}"

    # Make an API request using the new method
    response = openai_client.completions.create(
        model="gpt-3.5-turbo-instruct",
        prompt=prompt,
        temperature=0.7,
        max_tokens=100,
    )

    # Extract and print the generated text
    generated_text = response.choices[0].text
    print("Generated HS Code:", generated_text)

    # Loading the saved model
    loaded_pipeline = joblib.load(model_inp)

    # Input HS Code as a string
    INP = generated_text

    # Remove non-numeric characters
    cleaned_input = re.sub(r'[^0-9.]', '', INP)

    # Check if the cleaned input is not empty
    if cleaned_input:
        # Remove trailing zeros after decimals
        # cleaned_input = re.sub(r'\.0*$', '', cleaned_input)

        # Remove all dots from the cleaned input
        cleaned_input = cleaned_input.replace('.', '')

        print("Cleaned input_HS_code:", cleaned_input)

        # Using the cleaned input as a string without conversion
        user_input = pd.Series({'HS Code': cleaned_input})
        predictions = loaded_pipeline.predict(user_input.values.reshape(-1, 1))[0]

        # Output predictions
        print("[+] Total duty with SWS of 10% on BCD:", predictions)
    else:
        print("Cleaned input is empty. Please provide a valid input.")

trained_model_path = input("Enter your trained model PATH: ")
while True :
  item_description_input = input("Please provide a detailed description of the item: ")


  generate_hs_code(item_description_input, trained_model_path)
