import json
from django.core.management.base import BaseCommand
from scraper.models import CareerPath, User

class Command(BaseCommand):
    help = 'Seeds the database with rich career path data'

    def handle(self, *args, **kwargs):
        self.stdout.write('Seeding Career Paths...')
        
        # Clear existing PREDEFINED paths to avoid duplicates/mess
        CareerPath.objects.filter(is_predefined=True).delete()

        # 1. Full Stack Developer Path
        fs_steps = [
            {
                "step_id": 1,
                "name": "HTML & CSS Fundamentals",
                "description": "Master the building blocks of the web. specific focus on Semantic HTML5, CSS Grid, Flexbox, and Responsive Design principles.",
                "resources": [
                    {"title": "MDN Web Docs - HTML Basics", "url": "https://developer.mozilla.org/en-US/docs/Learn/HTML"},
                    {"title": "CSS Tricks - Flexbox Guide", "url": "https://css-tricks.com/snippets/css/a-guide-to-flexbox/"}
                ],
                "status": "open",
                "duration": "2 Weeks"
            },
            {
                "step_id": 2,
                "name": "JavaScript Deep Dive",
                "description": "Learn the core language of the web. Understand ES6+, Async/Await, DOM manipulation, and functional programming concepts.",
                "resources": [
                    {"title": "JavaScript.info", "url": "https://javascript.info/"},
                    {"title": "Eloquent JavaScript", "url": "https://eloquentjavascript.net/"}
                ],
                "status": "locked",
                "duration": "4 Weeks"
            },
            {
                "step_id": 3,
                "name": "Frontend Frameworks (React)",
                "description": "Build dynamic, single-page applications using React. Learn Hooks, State Management (Redux/Context), and component lifecycle.",
                "resources": [
                    {"title": "React Official Docs", "url": "https://react.dev/"},
                    {"title": "Fullstack Open", "url": "https://fullstackopen.com/en/"}
                ],
                "status": "locked",
                "duration": "6 Weeks"
            },
            {
                "step_id": 4,
                "name": "Backend Development (Node/Python)",
                "description": "Understand server-side logic, RESTful APIs, and database interactions. Choose Node.js or Python (Django/FastAPI).",
                "resources": [
                    {"title": "Django Girls Tutorial", "url": "https://tutorial.djangogirls.org/"},
                    {"title": "Node.js Crash Course", "url": "https://nodejs.org/en/docs/guides/getting-started-guide/"}
                ],
                "status": "locked",
                "duration": "5 Weeks"
            },
            {
                "step_id": 5,
                "name": "Databases & Deployment",
                "description": "Learn SQL (PostgreSQL) vs NoSQL (MongoDB). deploy your applications using Docker, CI/CD pipelines, and Cloud services (AWS/Vercel).",
                "resources": [
                    {"title": "PostgreSQL Tutorial", "url": "https://www.postgresqltutorial.com/"},
                    {"title": "Docker for Beginners", "url": "https://docker-curriculum.com/"}
                ],
                "status": "locked",
                "duration": "3 Weeks"
            }
        ]

        CareerPath.objects.create(
            title="Full Stack Developer",
            description="Become a versatile developer capable of building entire web applications from scratch. This comprehensive path covers everything from frontend aesthetics to backend logic and infrastructure.",
            roadmap_data={"steps": fs_steps},
            progress=0,
            is_predefined=True,
            status="Active"
        )

        # 2. Data Scientist Path
        ds_steps = [
            {
                "step_id": 1,
                "name": "Python for Data Science",
                "description": "Master Python with a focus on data manipulation libraries like NumPy and Pandas. Understand virtual environments and Jupyter Notebooks.",
                "resources": [
                    {"title": "Kaggle Python Course", "url": "https://www.kaggle.com/learn/python"},
                    {"title": "Pandas Documentation", "url": "https://pandas.pydata.org/docs/"}
                ],
                "status": "open",
                "duration": "3 Weeks"
            },
            {
                "step_id": 2,
                "name": "Exploratory Data Analysis & Visualization",
                "description": "Learn to tell stories with data. Master Matplotlib, Seaborn, and Plotly to discover patterns and insights.",
                "resources": [
                    {"title": "Data Visualization with Python", "url": "https://realpython.com/python-data-visualization-seaborn/"},
                    {"title": "Storytelling with Data", "url": "https://www.storytellingwithdata.com/"}
                ],
                "status": "locked",
                "duration": "3 Weeks"
            },
            {
                "step_id": 3,
                "name": "Machine Learning Fundamentals",
                "description": "Understand core algorithms: Regression, Classification, Clustering. Learn Scikit-Learn and model evaluation metrics.",
                "resources": [
                    {"title": "Google ML Crash Course", "url": "https://developers.google.com/machine-learning/crash-course"},
                    {"title": "Scikit-Learn User Guide", "url": "https://scikit-learn.org/stable/user_guide.html"}
                ],
                "status": "locked",
                "duration": "6 Weeks"
            },
            {
                "step_id": 4,
                "name": "Deep Learning & Neural Networks",
                "description": "Dive into the black box. Learn TensorFlow or PyTorch for computer vision (CNNs) and natural language processing (RNNs/Transformers).",
                "resources": [
                    {"title": "DeepLearning.AI", "url": "https://www.deeplearning.ai/"},
                    {"title": "PyTorch Tutorials", "url": "https://pytorch.org/tutorials/"}
                ],
                "status": "locked",
                "duration": "8 Weeks"
            }
        ]

        CareerPath.objects.create(
            title="Data Scientist",
            description="Extract knowledge and insights from data. This path guides you through the statistical and computational tools needed to solve complex problems with AI.",
            roadmap_data={"steps": ds_steps},
            progress=0,
            is_predefined=True,
            status="Active"
        )

        self.stdout.write(self.style.SUCCESS('Successfully seeded career paths.'))
