import configparser
import os

# Create a config parser object
config = configparser.ConfigParser()

config.read('config.ini')
DATABASE_PATH = config.get('paths', 'database_directory', fallback='data')

if not os.path.exists(DATABASE_PATH):
    print(f"INFO: Database directory not found. Creating it at: {DATABASE_PATH}")
    os.makedirs(DATABASE_PATH)