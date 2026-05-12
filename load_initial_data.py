import os
import django
from django.core.management import call_command
from django.db import connection

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'lumos_career.settings')
django.setup()

from django.contrib.auth import get_user_model
User = get_user_model()

def run():
    # Check if the database has already been populated
    # We use the existence of any User as a proxy for whether the initial data has been loaded
    try:
        if not User.objects.exists():
            print("Database is empty. Loading data from datadump.json...")
            if os.path.exists('datadump.json'):
                call_command('loaddata', 'datadump.json')
                print("Successfully loaded initial data.")
            else:
                print("datadump.json not found. Skipping data load.")
        else:
            print("Database already contains data. Skipping datadump.json load.")
    except Exception as e:
        print(f"An error occurred while checking or loading data: {e}")

if __name__ == '__main__':
    run()
