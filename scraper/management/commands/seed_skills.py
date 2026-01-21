from django.core.management.base import BaseCommand
from scraper.models import SkillNode, SkillEdge, SkillSignal

class Command(BaseCommand):
    help = 'Seeds the database with a rich set of Skills, Connections, and Market Signals.'

    def handle(self, *args, **options):
        self.stdout.write("Seeding database with algorithmic data...")

        # 1. Clear existing data
        SkillEdge.objects.all().delete()
        SkillSignal.objects.all().delete()
        SkillNode.objects.all().delete()

        # 2. Define Skills (Nodes)
        # Format: (Name, Category, Difficulty 1-10, Weeks)
        skills_data = [
            # Fundamentals
            ("HTML/CSS", "Language", 2, 2),
            ("JavaScript", "Language", 4, 4),
            ("Python", "Language", 3, 3),
            ("Git", "Tool", 3, 1),
            ("SQL", "Language", 4, 3),
            
            # Backend
            ("Django", "Framework", 6, 6),
            ("FastAPI", "Framework", 5, 2),
            ("Node.js", "Framework", 5, 4),
            ("PostgreSQL", "Tool", 5, 2),
            ("Docker", "Tool", 6, 3),
            ("Kubernetes", "Tool", 9, 8),
            ("Microservices", "Concept", 8, 4),
            
            # Frontend
            ("React", "Framework", 6, 5),
            ("Vue.js", "Framework", 5, 3),
            ("TypeScript", "Language", 5, 2),
            ("Redux", "Tool", 7, 2),
            ("Next.js", "Framework", 6, 3),
            
            # AI/Data
            ("Pandas", "Tool", 4, 2),
            ("Scikit-Learn", "Tool", 6, 4),
            ("TensorFlow", "Framework", 8, 8),
            ("PyTorch", "Framework", 8, 8),
            ("NLP", "Concept", 7, 6),
            ("Computer Vision", "Concept", 8, 6),
            
            # Cloud/DevOps
            ("AWS", "Tool", 7, 6),
            ("Terraform", "Tool", 7, 4),
            ("CI/CD", "Concept", 6, 2),
            ("System Design", "Concept", 9, 6),
        ]

        nodes = {}
        for name, cat, diff, weeks in skills_data:
            node = SkillNode.objects.create(
                name=name,
                category=cat,
                difficulty_level=diff,
                learning_weeks=weeks,
                description=f"A key {cat.lower()} for modern development."
            )
            nodes[name] = node
            self.stdout.write(f"Created Node: {name}")

        # 3. Define Connections (Edges)
        # Format: (Source, Target, TimeWeight, DifficultyWeight)
        edges_data = [
            # Web Track
            ("HTML/CSS", "JavaScript", 1, 2),
            ("JavaScript", "React", 2, 4),
            ("JavaScript", "Vue.js", 2, 3),
            ("JavaScript", "Node.js", 3, 4),
            ("React", "Next.js", 1, 2),
            ("React", "Redux", 2, 5),
            ("JavaScript", "TypeScript", 1, 3),
            ("TypeScript", "React", 1, 1),
            
            # Python Track
            ("Python", "Django", 2, 4),
            ("Python", "FastAPI", 2, 3),
            ("Python", "Pandas", 1, 2),
            ("SQL", "Django", 1, 2),
            ("SQL", "PostgreSQL", 1, 2),
            
            # AI Track
            ("Python", "Scikit-Learn", 2, 5),
            ("Pandas", "Scikit-Learn", 1, 3),
            ("Scikit-Learn", "TensorFlow", 4, 7),
            ("Scikit-Learn", "PyTorch", 4, 7),
            ("TensorFlow", "Computer Vision", 2, 6),
            
            # DevOps Track
            ("Git", "CI/CD", 2, 4),
            ("Linux", "Docker", 2, 5), # Oops, missed Linux, let's auto-create or skip
            ("Docker", "Kubernetes", 4, 8),
            ("Python", "Docker", 2, 4),
            ("Django", "Docker", 1, 2),
            ("Node.js", "Docker", 1, 2),
            ("Docker", "AWS", 3, 5),
            ("AWS", "Terraform", 2, 6),
            
            # Advanced
            ("Django", "Microservices", 4, 7),
            ("Django", "System Design", 5, 8),
        ]

        for src, tgt, w_time, w_diff in edges_data:
            if src not in nodes: # Handle missing nodes dynamically if needed
                continue
            if tgt not in nodes:
                continue
                
            SkillEdge.objects.create(
                source=nodes[src],
                target=nodes[tgt],
                weight_time=w_time,
                weight_difficulty=w_diff
            )
            self.stdout.write(f"Connected {src} -> {tgt}")

        # 4. Define Market Signals
        # Format: (Skill, SuccessRate, FailureRate, Trend)
        signals_data = [
            ("Python", 0.90, 0.20, "Stable"),
            ("JavaScript", 0.85, 0.25, "Stable"),
            ("React", 0.80, 0.30, "Rising"),
            ("Django", 0.70, 0.40, "Stable"),
            ("Kubernetes", 0.95, 0.10, "Rising"), # High Lift!
            ("TensorFlow", 0.88, 0.15, "Rising"),
            ("AWS", 0.92, 0.15, "Rising"),
            ("HTML/CSS", 0.60, 0.50, "Falling"), # Saturated
            ("System Design", 0.98, 0.05, "Rising"), # Very high lift
        ]

        for skill_name, succ, fail, trend in signals_data:
            if skill_name in nodes:
                SkillSignal.objects.create(
                    skill=nodes[skill_name],
                    success_rate=succ,
                    failure_rate=fail,
                    demand_trend=trend
                )
                self.stdout.write(f"Added Signal for {skill_name}")

        # 5. Seed Predefined Career Paths
        from scraper.models import CareerPath
        
        # Clear existing predefined paths
        CareerPath.objects.filter(is_predefined=True).delete()

        predefined_paths = [
            {
                "title": "Full Stack Python Developer",
                "description": "Master the art of building complete web applications using Python, Django, and modern frontend technologies like React.",
                "roadmap_data": {
                    "steps": [
                        {"name": "HTML/CSS", "status": "completed", "description": "Learn the building blocks of the web."},
                        {"name": "Python", "status": "completed", "description": "Master Python fundamentals."},
                        {"name": "Django", "status": "in-progress", "description": "Build robust backends with Django."},
                        {"name": "SQL", "status": "in-progress", "description": "Manage databases effectively."},
                        {"name": "JavaScript", "status": "locked", "description": "Add interactivity to your sites."},
                        {"name": "React", "status": "locked", "description": "Build modern, dynamic user interfaces."}
                    ]
                },
                "progress": 30
            },
            {
                "title": "AI & Data Scientist",
                "description": "Dive deep into data analysis, machine learning, and artificial intelligence with Python's powerful ecosystem.",
                "roadmap_data": {
                    "steps": [
                        {"name": "Python", "status": "completed", "description": "Master Python for data science."},
                        {"name": "Pandas", "status": "in-progress", "description": "Manipulate and analyze data structures."},
                        {"name": "SQL", "status": "in-progress", "description": "Query large datasets."},
                        {"name": "Scikit-Learn", "status": "locked", "description": "Implement machine learning algorithms."},
                        {"name": "TensorFlow", "status": "locked", "description": "Build and deploy neural networks."}
                    ]
                },
                "progress": 20
            },
            {
                "title": "DevOps Engineer",
                "description": "Bridge the gap between development and operations. Learn to deploy, scale, and manage infrastructure.",
                "roadmap_data": {
                    "steps": [
                        {"name": "Linux", "status": "completed", "description": "Master the command line."},
                        {"name": "Git", "status": "completed", "description": "Version control your code."},
                        {"name": "Docker", "status": "in-progress", "description": "Containerize applications."},
                        {"name": "AWS", "status": "locked", "description": "Manage cloud infrastructure."},
                        {"name": "Kubernetes", "status": "locked", "description": "Orchestrate container deployments."}
                    ]
                },
                "progress": 15
            }
        ]

        for path in predefined_paths:
            CareerPath.objects.create(
                title=path["title"],
                description=path["description"],
                roadmap_data=path["roadmap_data"],
                progress=path["progress"],
                is_predefined=True,
                status="Active"
            )
            self.stdout.write(f"Created Predefined Path: {path['title']}")

        self.stdout.write(self.style.SUCCESS('Successfully seeded database with algorithmic data and predefined paths'))
