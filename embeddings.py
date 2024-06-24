import os
import sqlite3
from datetime import datetime, timedelta
import numpy as np
from annoy import AnnoyIndex
import spacy

# Define constants
MACOS_EPOCH = datetime(2001, 1, 1)

def main():
    # Path to the AddressBook database
    db_path = '~/Library/Application Support/AddressBook/AddressBook-v22.abcddb'
    
    # Query the AddressBook database
    results = query_address_book(db_path)
    print("Query results:")
    for row in results:
        print(row)

    # Extract messages with contact names
    messages = extract_messages_with_contact_names(
        '~/Library/Messages/chat.db',
        db_path
    )

    # Print the earliest message by date
    if messages:
        text, contact_name, timestamp = messages[0]
        readable_date = convert_timestamp(timestamp)
        print(f"Earliest Message - Text: {text}, Contact: {contact_name}, Date: {readable_date}")
    else:
        print("No messages found.")

    # Extract messages
    messages = extract_messages('~/Library/Messages/chat.db')

    # Print all messages sorted by date
    for message in messages:
        text, contact_name, timestamp = message
        readable_date = convert_timestamp(timestamp)
        print(f"Text: {text}, Contact: {contact_name}, Date: {readable_date}")

    # Generate embeddings and create Annoy index
    generate_and_save_embeddings(messages)

def query_address_book(db_path):
    db_path = os.path.expanduser(db_path)
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("""
    SELECT ZABCDRECORD.ZFIRSTNAME, ZABCDRECORD.ZLASTNAME, ZABCDPHONENUMBER.ZFULLNUMBER
    FROM ZABCDRECORD
    JOIN ZABCDPHONENUMBER ON ZABCDPHONENUMBER.ZOWNER = ZABCDRECORD.Z_PK
    LIMIT 10;
    """)
    results = cursor.fetchall()
    conn.close()
    return results

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

def extract_messages_with_contact_names(messages_db_path, contacts_db_path):
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
    WHERE message.text IS NOT NULL
    ORDER BY message.date ASC
    """
    messages_cursor.execute(query)
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

def extract_messages(messages_db_path):
    messages_db_path = os.path.expanduser(messages_db_path)
    conn = sqlite3.connect(messages_db_path)
    cursor = conn.cursor()
    query = """
    SELECT message.text, handle.id, message.date
    FROM message
    JOIN handle ON message.handle_id = handle.ROWID
    WHERE message.text IS NOT NULL
    ORDER BY message.date ASC
    """
    cursor.execute(query)
    messages = cursor.fetchall()
    conn.close()
    return messages

def convert_timestamp(macos_timestamp):
    if macos_timestamp > 1e12:
        macos_timestamp = macos_timestamp / 1e9
    readable_date = MACOS_EPOCH + timedelta(seconds=macos_timestamp)
    return readable_date.strftime('%Y-%m-%d %H:%M:%S')

def generate_embeddings(texts):
    nlp = spacy.load('en_core_web_sm')
    embeddings = []
    for doc in nlp.pipe(texts):
        embeddings.append(doc.vector)
    return np.array(embeddings)

def generate_and_save_embeddings(messages):
    texts = [message[0] for message in messages]
    embeddings = generate_embeddings(texts)
    dimension = embeddings.shape[1]
    index = AnnoyIndex(dimension, 'angular')
    for i, embedding in enumerate(embeddings):
        index.add_item(i, embedding)
    index.build(10)
    index.save("message_embeddings.ann")
    for i, message in enumerate(messages[:10]):
        text, contact_name, timestamp = message
        readable_date = convert_timestamp(timestamp)
        print(f"ID: {i}, Text: {text}, Contact: {contact_name}, Date: {readable_date}, Embedding: {embeddings[i]}")

if __name__ == "__main__":
    main()