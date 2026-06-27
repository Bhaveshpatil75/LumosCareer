import os
import django
import urllib.parse

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'lumos_career.settings')
django.setup()

from core.models import CareerPath

def seed_resources():
    paths = CareerPath.objects.filter(is_predefined=True)
    updated_count = 0
    
    for path in paths:
        roadmap = path.roadmap_data
        steps = roadmap.get('steps', [])
        modified = False
        
        for step in steps:
            resources = step.get('resources', [])
            new_resources = []
            
            for res in resources:
                # If the resource is just a string, convert it to a rich dictionary object
                if isinstance(res, str):
                    safe_query = urllib.parse.quote_plus(res)
                    url = f"https://www.google.com/search?q={safe_query}"
                    new_resources.append({
                        "title": res,
                        "url": url
                    })
                    modified = True
                else:
                    new_resources.append(res)
                    
            step['resources'] = new_resources
            
        if modified:
            roadmap['steps'] = steps
            # Sync back to roadmap key if it exists
            if 'roadmap' in roadmap:
                roadmap['roadmap'] = steps
            path.roadmap_data = roadmap
            path.save()
            updated_count += 1
            print(f"Updated resources for path: {path.title}")
            
    print(f"Total paths updated: {updated_count}")

if __name__ == "__main__":
    seed_resources()
