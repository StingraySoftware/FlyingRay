import configparser
import os

# Create a config parser object
config = configparser.ConfigParser()

# Read the config file from the project's root directory
# Note: This assumes you run the app from the root, which you do.
config.read('config.ini')

# Get the path from the file and store it in a global variable
DATABASE_PATH = config.get('paths', 'database_directory', fallback='data')

# --- Important: Create the directory if it doesn't exist ---
if not os.path.exists(DATABASE_PATH):
    print(f"INFO: Database directory not found. Creating it at: {DATABASE_PATH}")
    os.makedirs(DATABASE_PATH)