#!/bin/bash
# entrypoint.sh

# Apply database migrations
echo "Apply database migrations"
python manage.py migrate --noinput

# Load initial data if database is empty
echo "Checking and loading initial data if needed"
python load_initial_data.py

# Start Gunicorn processes
echo "Starting Gunicorn."
exec gunicorn lumos_career.wsgi:application --bind 0.0.0.0:8000
