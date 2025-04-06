import pandas as pd
import os


def load_sms_spam_dataset(data_path, seed=123):
    """
    Loads the dataset of emails that are categorized into spam and ham
    # Arguments
        data_path: string, path for the data directory
        seed: int, seed for randomizer
    # Returns 
    """
    sms_data_path = os.path.join(data_path, 'SMSSpamCollection')


load_sms_spam_dataset("C:\Users\ester\OneDrive\Documentos\QuintoPeríodo\Treinamento LapPi\smsspamcollection")