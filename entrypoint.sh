#!/bin/bash
# entrypoint.sh

# Apply database migrations
echo "Apply database migrations"
python manage.py migrate --noinput

# Start Gunicorn processes
echo "Starting Gunicorn."
exec gunicorn lumos_career.wsgi:application --bind 0.0.0.0:8000
