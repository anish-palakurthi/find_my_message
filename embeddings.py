import os
import sqlite3
from datetime import datetime, timedelta
import numpy as np
from annoy import AnnoyIndex
import spacy
import json
from spacy.lang.en.stop_words import STOP_WORDS
from itertools import islice


# Define constants
MACOS_EPOCH = datetime(2001, 1, 1)


#used for storing/updating which time messages are updated to
def load_config(config_path='config.json'):
    with open(config_path, 'r') as config_file:
        config = json.load(config_file)
    return config['localStore']['lastUpdatedTime']

def update_last_updated_time(new_time, config_path='config.json'):
    with open(config_path, 'r') as config_file:
        config = json.load(config_file)
    config['localStore']['lastUpdatedTime'] = new_time
    with open(config_path, 'w') as config_file:
        json.dump(config, config_file, indent=4)


def update_vector_db(config_path='config.json', messages_db_path='~/Library/Messages/chat.db', contacts_db_path='~/Library/Application Support/AddressBook/AddressBook-v22.abcddb'):
    # Load the last updated time from the config file
    last_updated_time = load_config(config_path)

    # Extract messages with contact names after the last updated time
    messages = extract_messages_with_contact_names(messages_db_path, contacts_db_path, last_updated_time)

    # Only use the first 10 messages
    # messages = messages[:10]


    # Generate embeddings and create Annoy index for the messages
    generate_and_save_embeddings(messages)

    # Update the last updated time in the config file
    new_time = messages[-1][2]
    
    update_last_updated_time(new_time, config_path)
    

def format_phone_number(phone_number):
    if phone_number is None:
        return None
    return phone_number.replace(' ', '').replace('-', '').replace('(', '').replace(')', '')

def extract_contacts(db_path):
    db_path = os.path.expanduser(db_path)
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("""
    SELECT ZABCDPHONENUMBER.ZFULLNUMBER, ZABCDRECORD.ZFIRSTNAME, ZABCDRECORD.ZLASTNAME
    FROM ZABCDPHONENUMBER
    JOIN ZABCDRECORD ON ZABCDPHONENUMBER.ZOWNER = ZABCDRECORD.Z_PK;
    """)
    contacts = cursor.fetchall()
    conn.close()
    return contacts

def extract_messages_with_contact_names(messages_db_path, contacts_db_path, last_updated_time):
    messages_db_path = os.path.expanduser(messages_db_path)
    contacts_db_path = os.path.expanduser(contacts_db_path)
    
    contacts = extract_contacts(contacts_db_path)
    contacts_dict = {format_phone_number(contact[0]): f"{contact[1]} {contact[2]}".strip() for contact in contacts}
    
    messages_conn = sqlite3.connect(messages_db_path)
    messages_cursor = messages_conn.cursor()
    query = """
    SELECT message.text, handle.id, handle.uncanonicalized_id, message.date
    FROM message
    JOIN handle ON message.handle_id = handle.ROWID
    WHERE message.text IS NOT NULL AND message.date > ?
    ORDER BY message.date ASC
    """
    messages_cursor.execute(query, (last_updated_time,))
    messages = messages_cursor.fetchall()
    messages_conn.close()
    
    messages_with_contacts = []
    for message in messages:
        text, handle_id, contact_name, timestamp = message
        formatted_contact_name = format_phone_number(contact_name)
        full_name = contacts_dict.get(formatted_contact_name, contact_name)
        full_name = full_name if full_name else contact_name
        messages_with_contacts.append((text, full_name, timestamp))
    
    return messages_with_contacts


def convert_timestamp(macos_timestamp):
    if macos_timestamp > 1e12:
        macos_timestamp = macos_timestamp / 1e9
    readable_date = MACOS_EPOCH + timedelta(seconds=macos_timestamp)
    return readable_date.strftime('%Y-%m-%d %H:%M:%S')


def generate_embeddings(texts):
    nlp = spacy.load('en_core_web_lg')
    embeddings = []
    for doc in nlp.pipe(texts):
        embeddings.append(doc.vector)
    return np.array(embeddings)


def generate_and_save_embeddings(messages):
    texts = [message[0] for message in messages]
    embeddings = generate_embeddings(texts)
    dimension = embeddings.shape[1]
    
    # Load existing embeddings and additional data if they exist
    if os.path.exists("message_embeddings.ann") and os.path.exists("mappings.json"):
        old_index = AnnoyIndex(dimension, 'angular')
        old_index.load("message_embeddings.ann")
        with open('mappings.json', 'r') as f:
            additional_data = json.load(f)
        old_embeddings = [old_index.get_item_vector(i) for i in range(old_index.get_n_items())]
        embeddings = old_embeddings + list(embeddings)
    else:
        additional_data = {}

    # Create new Annoy index with combined embeddings
    new_index = AnnoyIndex(dimension, 'angular')
    for i, embedding in enumerate(embeddings):
        new_index.add_item(i, embedding)

    # Add new additional data
    for i, message in enumerate(messages, start=len(additional_data)):
        text, contact_name, timestamp = message
        additional_data[i] = {'text': text, 'phone_number': contact_name}

    new_index.build(10)
    new_index.save("message_embeddings.ann")

    # Save the combined additional data to a JSON file
    with open('mappings.json', 'w') as f:
        json.dump(additional_data, f)



def test_annoy_index(query_text):
    # Load the Annoy index
    index = AnnoyIndex(300, 'angular')  # 300 is the dimension of the vectors
    index.load('message_embeddings.ann')

    # Load the spaCy model
    nlp = spacy.load('en_core_web_lg')

    # Prepare the search query vector
    query_vector = nlp(query_text).vector

    # Load the additional data from mappings.json
    with open('mappings.json', 'r') as f:
        additional_data = json.load(f)

    # Query the index for the top 10 nearest neighbors
    num_neighbors = 10
    nearest_neighbors = index.get_nns_by_vector(query_vector, num_neighbors)

    # Print the nearest neighbors and their info
    for neighbor in nearest_neighbors:
        info = additional_data.get(str(neighbor), {})
        print(f"Neighbor ID: {neighbor}, Info: {info}")
        
""" 
def test_annoy_index(query_text):
    keywords = query_text.split(" ")
    # Load the Annoy index
    index = AnnoyIndex(300, 'angular')  # 300 is the dimension of the vectors
    index.load('message_embeddings.ann')

    # Load the spaCy model
    nlp = spacy.load('en_core_web_lg')

    # Prepare the search query vector
    query_vector = nlp(query_text).vector

    # Load the additional data from mappings.json
    with open('mappings.json', 'r') as f:
        additional_data = json.load(f)

    # Convert keywords to a set for faster lookup
    keywords_set = set(keywords)
    print(f"Keywords: {keywords}")

    # Score each data point based on semantic similarity and keyword presence
    scores = []
    num_items = int(len(additional_data) * 0.2)  # calculate 20% of the total number of items
    for neighbor, info in islice(additional_data.items(), num_items):
        text = info.get('text', '')
        print(f"Text: {text}")
        # Remove stop words from the text
        text_words = set(text.split()) - STOP_WORDS
        print(f"Text words: {text_words}")

        # Calculate semantic similarity
        text_vector = nlp(text).vector
        distance = 1 - index.get_distance(index.get_nns_by_vector(query_vector, 1)[0], index.get_nns_by_vector(text_vector, 1)[0])

        # Decrease the distance if any of the keywords are present in the text
        if keywords_set & text_words:
            print(f"Matched keywords: {keywords_set & text_words}")
            distance = max(0, distance - 0.0005)  # ensure the score doesn't go below 0
        # Decrease the distance even more if any subarray of the joined keywords is present in the text
        # joined_keywords = ' '.join(keywords)
        # dp = [[False] * (len(text) + 1) for _ in range(len(joined_keywords) + 1)]
        # dp[0] = [True] * (len(text) + 1)
        # for i in range(1, len(joined_keywords) + 1):
        #     for j in range(i, len(text) + 1):
        #         if text[j - i:j] == joined_keywords[:i]:
        #             dp[i][j] = dp[i - 1][j - i]
        # if any(dp[len(joined_keywords)]):
        #     distance -= 0.1  # adjust this value as needed
        #     print(f"Matched: {text}")
        scores.append((neighbor, distance, info))

    # Sort the results by score (lower distance is better)
    scores.sort(key=lambda x: x[1])

    # Print the scored results
    for neighbor, score, info in scores:
        print(f"Neighbor ID: {neighbor}, Score: {score}, Info: {info}")
 """
if __name__ == "__main__":
    # update_vector_db()
    test_annoy_index("leetcode problem capacitor")
    