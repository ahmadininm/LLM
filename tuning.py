import os
import requests
import json
from datasets import load_dataset, Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model, prepare_model_for_int8_training

# --------------------------------------------------------------------------------
# CONFIGURATION
# --------------------------------------------------------------------------------

# Define API endpoints for data collection.
# These endpoints provide abstracts and metadata related to sustainability topics.
API_ENDPOINTS = {
    "pubmed": "https://api.ncbi.nlm.nih.gov/lit/ctxp/v1/pubmed/",
    "core": "https://core.ac.uk/api-v2/articles/search",
    "crossref": "https://api.crossref.org/works"
}

# If you have API keys, place them here. Some services may be free but require keys
# for higher rate limits or better access.
API_KEYS = {
    "pubmed": "<YOUR_PUBMED_API_KEY>",  
    "core": "<YOUR_CORE_API_KEY>",
    "crossref": None
}

# The chosen search query to fetch data related to "sustainable manufacturing".
SEARCH_QUERY = "sustainable manufacturing"

# Name of the base model to be used. For demonstration, we use Mistral-7B.
MODEL_NAME = "mistralai/Mistral-7B"

# --------------------------------------------------------------------------------
# STEP 1: DATA COLLECTION FUNCTIONS
# --------------------------------------------------------------------------------

def fetch_pubmed_data(query, max_results=100):
    """
    Fetch articles from the PubMed API based on a given query.
    
    Args:
        query (str): The search phrase for PubMed, e.g., 'sustainable manufacturing'.
        max_results (int): Maximum number of articles to retrieve.
        
    Returns:
        dict: A JSON response from the PubMed API.
    """
    url = API_ENDPOINTS['pubmed']
    # 'retmax' controls how many results to return.
    params = {"db": "pubmed", "term": query, "retmax": max_results}
    headers = {"Content-Type": "application/json"}
    response = requests.get(url, params=params, headers=headers)
    response.raise_for_status()  # Raises an error if the request fails
    return response.json()

def fetch_core_data(query, max_results=100):
    """
    Fetch articles from the CORE API based on a given query.
    
    Args:
        query (str): The search phrase for CORE, e.g., 'sustainable manufacturing'.
        max_results (int): Maximum number of articles to retrieve.
        
    Returns:
        dict: A JSON response from the CORE API.
    """
    url = API_ENDPOINTS['core']
    params = {
        "q": query,
        "limit": max_results,
        "apiKey": API_KEYS['core']
    }
    response = requests.get(url, params=params)
    response.raise_for_status()
    return response.json()

def fetch_crossref_data(query, max_results=100):
    """
    Fetch article metadata from the CrossRef API based on a given query.
    
    Args:
        query (str): The search phrase for CrossRef, e.g., 'sustainable manufacturing'.
        max_results (int): Maximum number of works to retrieve.
        
    Returns:
        dict: A JSON response from the CrossRef API.
    """
    url = API_ENDPOINTS['crossref']
    params = {"query": query, "rows": max_results}
    response = requests.get(url, params=params)
    response.raise_for_status()
    return response.json()

# --------------------------------------------------------------------------------
# STEP 1 (CONTINUED): DATA COLLECTION WORKFLOW
# --------------------------------------------------------------------------------

def collect_and_preprocess_data():
    """
    Collect data from three APIs (PubMed, CORE, CrossRef) using the specified query.
    Extract relevant text fields (e.g., abstracts), combine them into one list,
    and return the combined data for further processing.
    
    Returns:
        list of str: A list of text entries (abstracts, descriptions, etc.).
    """
    query = SEARCH_QUERY
    
    print("Fetching data from APIs...")
    pubmed_data = fetch_pubmed_data(query)
    core_data = fetch_core_data(query)
    crossref_data = fetch_crossref_data(query)

    combined_data = []
    
    # For demonstration, we assume that PubMed returns a structure where
    # we can extract abstracts from an 'articles' field. Adjust as needed.
    for item in pubmed_data.get("articles", []):
        # Use the 'get' method with a default of an empty string to avoid errors
        combined_data.append(item.get("abstract", ""))
    
    # CORE often provides a 'description' or 'fullText'. Here, we use 'description'.
    for item in core_data.get("data", []):
        combined_data.append(item.get("description", ""))
    
    # CrossRef typically stores abstracts in 'message.items.abstract' but it is
    # not always guaranteed to exist.
    for item in crossref_data.get("message", {}).get("items", []):
        combined_data.append(item.get("abstract", ""))

    print(f"Collected {len(combined_data)} entries.")
    return combined_data

# --------------------------------------------------------------------------------
# STEP 2: SAVING AND LOADING PREPROCESSED DATA
# --------------------------------------------------------------------------------

def save_preprocessed_data(data, filename="sustainability_dataset.json"):
    """
    Save a list of text entries to a JSON file.
    
    Args:
        data (list of str): The text data to be saved.
        filename (str): The name of the JSON file.
    """
    with open(filename, "w") as f:
        json.dump(data, f)
    print(f"Saved preprocessed data to {filename}")

def load_preprocessed_data(filename="sustainability_dataset.json"):
    """
    Load the preprocessed data from a JSON file.
    
    Args:
        filename (str): The name of the JSON file to load from.
        
    Returns:
        list of str: A list of text entries (abstracts or full text).
    """
    with open(filename, "r") as f:
        return json.load(f)

# --------------------------------------------------------------------------------
# STEP 3: TOKENISATION AND MODEL PREPARATION
# --------------------------------------------------------------------------------

def tokenize_data(data, tokenizer, max_length=512):
    """
    Tokenise a list of text entries using a specified tokenizer.
    Each entry is truncated or padded to a given maximum length.
    
    Args:
        data (list of str): Text data (e.g. abstracts) to be tokenised.
        tokenizer (PreTrainedTokenizer): The tokenizer from a Hugging Face model.
        max_length (int): Maximum sequence length for truncation or padding.
        
    Returns:
        list of dict: A list of tokenised outputs with input IDs and attention masks.
    """
    print("Tokenising data...")
    tokenised_dataset = []
    for text in data:
        # We convert each piece of text into a token ID sequence. 
        # 'truncation=True' ensures we do not exceed 'max_length'.
        # 'padding=max_length' ensures uniform shape for batches.
        tokenised = tokenizer(text, truncation=True, max_length=max_length, padding="max_length")
        tokenised_dataset.append(tokenised)
    return tokenised_dataset

def load_model_and_tokenizer():
    """
    Load a pre-trained model and tokenizer. 
    This example uses Mistral-7B, but you can change it as needed.
    
    Returns:
        tuple: (model, tokenizer) ready for further fine-tuning.
    """
    print(f"Loading model and tokenizer from {MODEL_NAME}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, device_map="auto")
    return model, tokenizer

# --------------------------------------------------------------------------------
# STEP 4: LoRA CONFIGURATION AND FINE-TUNING
# --------------------------------------------------------------------------------

def prepare_model_for_fine_tuning(model):
    """
    Prepare the given model for Low-Rank Adaptation (LoRA) training.
    This includes configuring LoRA parameters and converting the model 
    for efficient int8 training if available.
    
    Args:
        model (PreTrainedModel): The model to prepare for LoRA.
        
    Returns:
        PreTrainedModel: The model wrapped with LoRA modules.
    """
    print("Configuring model for LoRA training...")
    # LoRA parameters:
    #   - 'r': The rank of the low-rank adaptation.
    #   - 'lora_alpha': A scaling factor for the LoRA parameters.
    #   - 'target_modules': The modules to apply LoRA to (e.g., q_proj, v_proj).
    #   - 'lora_dropout': Dropout rate for LoRA layers to prevent overfitting.
    #   - 'bias': Whether bias parameters are included in LoRA or not.
    lora_config = LoraConfig(
        r=16, 
        lora_alpha=32, 
        target_modules=["q_proj", "v_proj"], 
        lora_dropout=0.1, 
        bias="none", 
        task_type="CAUSAL_LM"
    )
    
    # Prepare the model for int8 training if you have suitable hardware.
    # This can greatly reduce memory use.
    model = prepare_model_for_int8_training(model)
    
    # Wrap the model with LoRA modules.
    model = get_peft_model(model, lora_config)
    return model

def fine_tune_model(model, tokenizer, tokenised_data):
    """
    Fine-tune the model using the provided tokenised data and specified 
    training arguments. Saves the final model to a local folder.
    
    Args:
        model (PreTrainedModel): The model pre-wrapped with LoRA modules.
        tokenizer (PreTrainedTokenizer): The tokenizer to use for processing data.
        tokenised_data (list of dict): Tokenised inputs including 'input_ids' and 'attention_mask'.
    """
    print("Fine-tuning the model...")

    # Convert the tokenised data into a Hugging Face 'Dataset'.
    train_dataset = Dataset.from_dict({
        "input_ids": [d["input_ids"] for d in tokenised_data], 
        "attention_mask": [d["attention_mask"] for d in tokenised_data]
    })

    # Configure your training hyperparameters.
    training_args = TrainingArguments(
        output_dir="./mistral_finetuned",
        per_device_train_batch_size=4,
        num_train_epochs=3,
        learning_rate=1e-4,
        logging_dir="./logs",
        save_strategy="epoch",
        evaluation_strategy="epoch",
        fp16=True,   # Enables mixed-precision training if your hardware supports it.
        push_to_hub=False
    )

    # Instantiate the Hugging Face 'Trainer' with the model, data, and arguments.
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset
        # If you want to add a validation dataset, pass it as 'eval_dataset=...'
    )

    # Start the training process.
    trainer.train()

    # Save the fine-tuned model locally.
    model.save_pretrained("./mistral_finetuned")
    print("Model fine-tuned and saved to './mistral_finetuned'.")

# --------------------------------------------------------------------------------
# STEP 5: MAIN WORKFLOW
# --------------------------------------------------------------------------------

def main():
    """
    Orchestrate the entire process:
      1. Collect and preprocess data from PubMed, CORE, and CrossRef.
      2. Save and then load the preprocessed data for reusability.
      3. Load a pre-trained model and tokenizer.
      4. Tokenise the data.
      5. Prepare the model for LoRA training.
      6. Fine-tune the model using the tokenised data.
    """
    # STEP 1: Data collection and simple preprocessing
    data = collect_and_preprocess_data()
    save_preprocessed_data(data)

    # STEP 2: Load the preprocessed data
    preprocessed_data = load_preprocessed_data()

    # STEP 3: Load the model and tokenizer
    model, tokenizer = load_model_and_tokenizer()

    # STEP 4: Tokenise the loaded data
    tokenised_data = tokenize_data(preprocessed_data, tokenizer)

    # STEP 5: Prepare model for LoRA
    model = prepare_model_for_fine_tuning(model)

    # STEP 6: Fine-tune the model
    fine_tune_model(model, tokenizer, tokenised_data)

# Execute the main function when the script is run
if __name__ == "__main__":
    main()


# --------------------------------------------------------------------------------
# PROMOTIONAL / SUMMARY REMARKS
# --------------------------------------------------------------------------------
"""
PROMO: Comprehensive Pipeline for Fine-Tuning Mistral-7B on Sustainable Manufacturing Data

Objective:
This Python script gathers domain-specific data on sustainable manufacturing from three well-known APIs (PubMed, CORE, and CrossRef) and fine-tunes the Mistral-7B language model using LoRA (Low-Rank Adaptation). The pipeline is carefully designed to be modular and can be adapted to other fields or larger datasets.

Key Features:
1. Automated Data Retrieval:
   - Fetches abstracts and related metadata from PubMed, CORE, and CrossRef for reliable sustainability insights.
2. Flexible Preprocessing:
   - Cleans and tokenises text data, saving it locally for repeated use or further enhancement.
3. Efficient Fine-Tuning:
   - Applies LoRA, which allows the model to learn task-specific knowledge in a memory-efficient manner.
4. Easy to Modify:
   - Adjust the search query, add new APIs, or tweak training parameters with minimal difficulty.

Outcomes:
- A domain-specialised language model tailored to sustainable manufacturing.
- A ready-to-use codebase for academics or industry professionals wanting to incorporate AI-driven insights into decision-making for sustainability.

This script thus represents a thorough example for novices and experts alike, demonstrating how to merge data gathering, model customisation, and training processes into a single workflow.
"""
