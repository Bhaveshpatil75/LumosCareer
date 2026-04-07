import os
import django

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "lumos_career.settings")
django.setup()

from core.models import Company, SkillNode, SkillEdge, SkillSignal
from core.algorithms.pagerank import PageRank

def populate():
    print("Clearing old data...")
    SkillSignal.objects.all().delete()
    SkillEdge.objects.all().delete()
    SkillNode.objects.all().delete()
    Company.objects.all().delete()

    print("Populating Companies (50+)...")
    companies_data = [
        # FAANG / Big Tech
        ("Google", "Tech", "Python, Go, C++, Kubernetes, Angular, TensorFlow, Cloud, Java, gRPC, Spanner, BigTable, Protobuf"),
        ("Amazon", "Tech", "Java, AWS, DynamoDB, React, C++, Node.js, Python, TypeScript, Kotlin, Lambda, ECS, SQS"),
        ("Netflix", "Tech", "Java, Spring Boot, RxJava, Node.js, React, AWS, Cassandra, Kafka, Python, GraphQL, Zuul"),
        ("Meta", "Tech", "Hack, React, GraphQL, PyTorch, C++, Android, Python, Presto, Cassandra, PHP, TypeScript"),
        ("Apple", "Tech", "Swift, Objective-C, C++, Python, AWS, SiriKit, Metal, CoreML, SwiftUI, Xcode"),
        ("Microsoft", "Tech", "C#, .NET, Azure, TypeScript, React, SQL Server, Python, PowerShell, VS Code, GitHub"),
        # Growth Tech
        ("Tesla", "Tech", "Python, C++, Rust, PyTorch, CUDA, React, Linux, Embedded Systems, ROS"),
        ("Spotify", "Tech", "Python, Java, C++, GCP, BigQuery, TensorFlow, React, Kafka, gRPC"),
        ("Stripe", "FinTech", "Ruby, Java, Go, Python, React, TypeScript, AWS, Kafka, GraphQL"),
        ("Airbnb", "Tech", "Ruby, React, Java, Kotlin, Python, AWS, Kafka, GraphQL, TypeScript"),
        ("Uber", "Tech", "Go, Java, Python, Node.js, React, Kafka, Cassandra, Redis, Kubernetes, Presto"),
        ("LinkedIn", "Tech", "Java, Scala, Python, React, Kafka, Spark, Hadoop, REST APIs, GraphQL"),
        ("Twitter/X", "Tech", "Scala, Java, Python, React, GraphQL, Kafka, Manhattan DB, gRPC"),
        # Cloud & Data
        ("Snowflake", "Cloud/Data", "Java, C++, Python, AWS, Azure, GCP, SQL, React, Go"),
        ("Databricks", "Cloud/Data", "Scala, Python, Spark, Delta Lake, MLflow, React, TypeScript, AWS, Azure"),
        ("Palantir", "Tech", "Java, TypeScript, React, Python, Go, Spark, PostgreSQL, GraphQL"),
        ("Cloudflare", "Tech", "Go, Rust, C, JavaScript, TypeScript, Lua, Kubernetes, Linux, Workers"),
        ("HashiCorp", "Tech", "Go, Python, React, TypeScript, Terraform, Vault, Consul, Nomad"),
        # FinTech
        ("Goldman Sachs", "Finance", "Java, Python, Slang, React, AWS, Kubernetes, C++, SQL"),
        ("JPMorgan Chase", "Finance", "Java, Python, Spring, React, Cloud, Kubernetes, SQL, Angular"),
        ("Coinbase", "FinTech/Crypto", "Go, Ruby, React, TypeScript, PostgreSQL, AWS, Kubernetes, Solidity"),
        ("Square/Block", "FinTech", "Java, Kotlin, Ruby, Python, React, AWS, Kubernetes, Go"),
        # Enterprise
        ("Salesforce", "Enterprise SaaS", "Java, Apex, Lightning Web Components, React, Python, Heroku, PostgreSQL"),
        ("Oracle", "Enterprise Tech", "Java, Python, SQL, OCI, Kubernetes, React, C++, PL/SQL"),
        ("SAP", "Enterprise Tech", "Java, ABAP, Python, React, HANA, Cloud Foundry, TypeScript"),
        ("Adobe", "Creative Tech", "JavaScript, React, C++, Python, Java, Node.js, GraphQL, AWS"),
        ("Atlassian", "DevTools", "Java, TypeScript, React, Python, AWS, Kotlin, GraphQL, Forge"),
        ("Shopify", "E-Commerce", "Ruby, React, TypeScript, Go, Python, GraphQL, Kubernetes, Rails"),
        # Productivity
        ("Notion", "Productivity", "TypeScript, React, Kotlin, Rust, PostgreSQL, AWS, Redis, Go"),
        ("Figma", "Design Tech", "TypeScript, C++, React, WebAssembly, Rust, Node.js, PostgreSQL"),
        ("Slack", "Communication", "Java, PHP, React, TypeScript, Go, MySQL, AWS, Kafka, Electron"),
        ("Discord", "Communication", "Rust, Python, React, TypeScript, Elixir, Cassandra, GCP, C++"),
        ("Zoom", "Communication", "C++, Python, JavaScript, React, AWS, WebRTC, Kotlin, Swift"),
        # Semiconductor / Hardware
        ("Nvidia", "Semiconductor/AI", "C++, CUDA, Python, TensorFlow, PyTorch, Linux, Vulkan, C"),
        ("AMD", "Semiconductor", "C++, C, Python, Verilog, VHDL, Linux, ROCm, OpenCL"),
        ("Intel", "Semiconductor", "C++, C, Python, Verilog, SystemVerilog, Linux, oneAPI, FPGA"),
        ("Qualcomm", "Semiconductor", "C++, C, Python, Linux, Android, ARM, DSP, Machine Learning"),
        # AI/Research
        ("OpenAI", "AI", "Python, PyTorch, Kubernetes, React, Ray, Rust, CUDA, Triton, Go"),
        ("Anthropic", "AI", "Python, PyTorch, Rust, TypeScript, React, Kubernetes, AWS"),
        ("DeepMind", "AI", "Python, JAX, TensorFlow, C++, Go, Kubernetes, Research"),
        # Asian Tech Giants
        ("ByteDance", "Tech", "Go, Python, Java, React, Flutter, Kafka, Redis, Kubernetes, C++"),
        ("Samsung", "Electronics", "C++, Java, Kotlin, Python, Tizen, Android, Linux, AI/ML"),
        # Consulting / Services
        ("Accenture", "Consulting", "Java, Python, .NET, React, Angular, AWS, Azure, SAP, Salesforce"),
        ("Deloitte", "Consulting", "Java, Python, SAP, .NET, Azure, AWS, Tableau, SQL, React"),
        ("McKinsey", "Consulting", "Python, R, SQL, Tableau, PowerBI, Excel, AWS, React"),
        # Indian IT
        ("Infosys", "IT Services", "Java, Python, .NET, Angular, React, AWS, Azure, Spring Boot, SQL"),
        ("TCS", "IT Services", "Java, Python, .NET, Angular, React, AWS, Azure, Microservices, SQL"),
        ("Wipro", "IT Services", "Java, Python, .NET, React, Angular, AWS, Azure, DevOps, SQL"),
        ("LTIMindtree", "IT Services", "Java, Python, .NET, React, AWS, Azure, Data Engineering, SQL"),
        ("Cognizant", "IT Services", "Java, Python, .NET, AWS, Azure, Spring Boot, SQL, Angular"),
        ("HCLTech", "IT Services", "Java, Python, C++, .NET, React, Cloud, DevOps, SQL"),
        ("Tech Mahindra", "IT Services", "Java, Python, Telecom, Cloud, .NET, AWS, Oracle, SQL"),
        # Aerospace
        ("NASA", "Aerospace", "C++, Python, Fortran, MATLAB, Linux, ROS, Simulation, AI/ML"),
        ("SpaceX", "Aerospace", "C++, Python, JavaScript, React, Linux, Embedded Systems, Kubernetes"),
    ]

    # Extended info for major companies (name -> extra fields dict)
    company_extra = {
        "Google": {"description": "Global technology leader in search, cloud, AI, and advertising.", "headquarters": "Mountain View, CA", "employee_count": "180000+", "culture_notes": "Known for innovation-driven culture, 20% time for personal projects, flat hierarchy, data-driven decisions, and comprehensive benefits.", "interview_process": "Phone screen → Technical phone interviews (2) → On-site (4-5 rounds: coding, system design, behavioral, Googleyness). Focus on problem-solving and scalability.", "avg_salary_range": "$150k-$400k", "careers_url": "https://careers.google.com"},
        "Amazon": {"description": "World's largest e-commerce and cloud computing (AWS) company.", "headquarters": "Seattle, WA", "employee_count": "1500000+", "culture_notes": "Leadership Principles-driven culture, customer obsession, bias for action, ownership mentality, frugality.", "interview_process": "Online assessment → Phone screen → On-site loop (5-6 rounds). Heavy focus on Leadership Principles (STAR format) and system design.", "avg_salary_range": "$130k-$350k", "careers_url": "https://amazon.jobs"},
        "Netflix": {"description": "Leading streaming entertainment service with 200M+ subscribers globally.", "headquarters": "Los Gatos, CA", "employee_count": "12000+", "culture_notes": "Freedom and Responsibility culture, no vacation tracking, keeper test, radical candor, top-of-market pay.", "interview_process": "Recruiter screen → Technical phone → On-site (4-5 rounds). Focus on senior-level thinking, system design, and culture fit.", "avg_salary_range": "$200k-$500k", "careers_url": "https://jobs.netflix.com"},
        "Meta": {"description": "Social media and metaverse technology company (Facebook, Instagram, WhatsApp).", "headquarters": "Menlo Park, CA", "employee_count": "70000+", "culture_notes": "Move fast culture, open office, hackathons, focus on impact and scale, strong engineering culture.", "interview_process": "Recruiter call → Technical screens (2) → On-site (3-4 rounds: coding, system design, behavioral). Ninja and Pirate rounds for E5+.", "avg_salary_range": "$160k-$450k", "careers_url": "https://metacareers.com"},
        "Apple": {"description": "Consumer electronics and software company known for iPhone, Mac, and ecosystem.", "headquarters": "Cupertino, CA", "employee_count": "160000+", "culture_notes": "Secrecy-driven culture, design excellence, cross-functional collaboration, attention to detail, product-first mindset.", "interview_process": "Phone screen → Technical phone → On-site (4-6 rounds). Strong focus on domain expertise, design thinking, and cultural fit.", "avg_salary_range": "$150k-$380k", "careers_url": "https://jobs.apple.com"},
        "Microsoft": {"description": "Enterprise software, cloud (Azure), and AI leader.", "headquarters": "Redmond, WA", "employee_count": "220000+", "culture_notes": "Growth mindset culture under Satya Nadella, inclusive, learning-oriented, work-life balance, hybrid work.", "interview_process": "Recruiter screen → Technical phone → On-site (4-5 rounds: coding, system design, behavioral). 'As Appropriate' final interview with hiring manager.", "avg_salary_range": "$140k-$380k", "careers_url": "https://careers.microsoft.com"},
        "Tesla": {"description": "Electric vehicle and clean energy company pushing sustainable transport.", "headquarters": "Austin, TX", "employee_count": "130000+", "culture_notes": "Mission-driven, fast-paced, high-intensity, flat structure, hands-on engineering culture.", "interview_process": "Phone screen → Technical assessment → On-site (3-4 rounds). Focus on practical problem solving and first-principles thinking.", "avg_salary_range": "$120k-$300k", "careers_url": "https://tesla.com/careers"},
        "Stripe": {"description": "Payment infrastructure for the internet, powering millions of businesses.", "headquarters": "San Francisco, CA", "employee_count": "8000+", "culture_notes": "Writing culture (Stripe Press), intellectual rigor, user-first, remote-friendly, high talent density.", "interview_process": "Recruiter call → Coding exercise → Technical interviews (2-3) → Final round. Known for thoughtful, well-structured interviews.", "avg_salary_range": "$170k-$400k", "careers_url": "https://stripe.com/jobs"},
        "OpenAI": {"description": "AI research lab building safe and beneficial artificial general intelligence.", "headquarters": "San Francisco, CA", "employee_count": "3000+", "culture_notes": "Mission-driven, research-oriented, fast iteration, high-impact work, collaborative and intellectually rigorous.", "interview_process": "Recruiter screen → Technical phone → On-site (3-4 rounds: coding, ML/AI depth, system design). Research presentations for research roles.", "avg_salary_range": "$200k-$500k+", "careers_url": "https://openai.com/careers"},
        "Goldman Sachs": {"description": "Leading global investment banking, securities, and management firm.", "headquarters": "New York, NY", "employee_count": "45000+", "culture_notes": "Meritocratic, high-performance, collaborative, emphasis on teamwork and client service.", "interview_process": "HireVue → Superday (multiple rounds in one day: technical, behavioral, case study). Focus on problem-solving and financial acumen.", "avg_salary_range": "$120k-$350k", "careers_url": "https://goldmansachs.com/careers"},
        "Nvidia": {"description": "Leading GPU and AI computing platform company.", "headquarters": "Santa Clara, CA", "employee_count": "29000+", "culture_notes": "Engineering-first culture, innovation-driven, collaborative, focus on pushing computing boundaries.", "interview_process": "Phone screen → Technical interviews (2-3) → On-site. Heavy focus on C++, GPU architecture, and parallel computing.", "avg_salary_range": "$150k-$400k", "careers_url": "https://nvidia.com/en-us/about-nvidia/careers"},
        "Spotify": {"description": "World's most popular audio streaming subscription service.", "headquarters": "Stockholm, Sweden", "employee_count": "9000+", "culture_notes": "Squad model (autonomous teams), band metaphor, innovation time, distributed-first, strong D&I focus.", "interview_process": "Phone screen → Technical assessment → On-site (3-4 rounds). Focus on system design, data structures, and culture fit.", "avg_salary_range": "$130k-$300k", "careers_url": "https://lifeatspotify.com"},
    }

    for name, industry, stack in companies_data:
        extra = company_extra.get(name, {})
        Company.objects.create(
            name=name,
            industry=industry,
            tech_stack=stack,
            description=extra.get('description', ''),
            headquarters=extra.get('headquarters', ''),
            employee_count=extra.get('employee_count', ''),
            culture_notes=extra.get('culture_notes', ''),
            interview_process=extra.get('interview_process', ''),
            avg_salary_range=extra.get('avg_salary_range', ''),
            careers_url=extra.get('careers_url', ''),
        )

    print("Populating Skill Graph (80+ nodes)...")
    skills_data = [
        # Languages
        ("Python", "Language", 3, 4), ("Java", "Language", 4, 6), ("C++", "Language", 7, 10),
        ("JavaScript", "Language", 3, 4), ("TypeScript", "Language", 4, 3), ("Go", "Language", 5, 6),
        ("Rust", "Language", 8, 10), ("Scala", "Language", 6, 8), ("Kotlin", "Language", 4, 5),
        ("Swift", "Language", 4, 6), ("Ruby", "Language", 3, 4), ("PHP", "Language", 3, 4),
        ("R", "Language", 4, 4), ("SQL", "Language", 3, 3), ("Solidity", "Language", 6, 6),
        ("Dart", "Language", 3, 3), ("Bash", "Language", 3, 2), ("MATLAB", "Language", 5, 4),
        ("HTML/CSS", "Language", 1, 2),
        # Frameworks - Frontend
        ("React", "Framework", 5, 6), ("Angular", "Framework", 6, 8), ("Vue.js", "Framework", 4, 4),
        ("Svelte", "Framework", 4, 3), ("Next.js", "Framework", 5, 4),
        # Frameworks - Backend
        ("Django", "Framework", 4, 4), ("Flask", "Framework", 3, 2), ("FastAPI", "Framework", 4, 3),
        ("Spring Boot", "Framework", 6, 8), ("Express.js", "Framework", 3, 3),
        ("NestJS", "Framework", 5, 5), ("Rails", "Framework", 4, 5), ("Laravel", "Framework", 4, 5),
        ("Node.js", "Framework", 4, 4),
        # Frameworks - Mobile
        ("Flutter", "Framework", 5, 5), ("React Native", "Framework", 5, 5),
        # Frameworks - ML
        ("PyTorch", "Framework", 7, 8), ("TensorFlow", "Framework", 7, 8),
        ("Scikit-learn", "Framework", 5, 4),
        # Frameworks - Game
        ("Unity", "Framework", 6, 6), ("Unreal Engine", "Framework", 8, 10),
        # Tools - Databases
        ("PostgreSQL", "Tool", 4, 4), ("MongoDB", "Tool", 4, 3), ("Redis", "Tool", 4, 3),
        ("MySQL", "Tool", 3, 3), ("Cassandra", "Tool", 6, 6), ("Elasticsearch", "Tool", 5, 5),
        ("DynamoDB", "Tool", 5, 4),
        # Tools - DevOps
        ("Docker", "Tool", 4, 3), ("Kubernetes", "Tool", 7, 8), ("Terraform", "Tool", 6, 6),
        ("Ansible", "Tool", 5, 4), ("Jenkins", "Tool", 4, 3), ("GitHub Actions", "Tool", 3, 2),
        ("Prometheus", "Tool", 5, 4), ("Grafana", "Tool", 4, 3), ("Git", "Tool", 2, 1),
        # Tools - Cloud
        ("AWS", "Tool", 6, 8), ("Azure", "Tool", 6, 8), ("GCP", "Tool", 6, 7),
        # Tools - Streaming
        ("Kafka", "Tool", 7, 6), ("RabbitMQ", "Tool", 5, 4),
        # Tools - Data
        ("Spark", "Tool", 7, 8), ("Hadoop", "Tool", 6, 6), ("Airflow", "Tool", 5, 4),
        # Concepts
        ("Machine Learning", "Concept", 7, 10), ("Deep Learning", "Concept", 8, 12),
        ("System Design", "Concept", 8, 12), ("Algorithms", "Concept", 7, 8),
        ("Data Structures", "Concept", 6, 6), ("Microservices", "Concept", 6, 6),
        ("API Design", "Concept", 5, 4), ("CI/CD", "Concept", 4, 3),
        ("Computer Vision", "Concept", 8, 10), ("NLP", "Concept", 8, 10),
        ("Reinforcement Learning", "Concept", 9, 12), ("LLMs", "Concept", 8, 8),
        ("RAG", "Concept", 7, 5), ("Prompt Engineering", "Concept", 4, 2),
        ("DevOps", "Concept", 6, 8), ("GraphQL", "Concept", 4, 3),
        ("Networking", "Concept", 6, 6), ("Linux", "Tool", 4, 4),
        ("NoSQL", "Concept", 5, 4), ("Data Engineering", "Concept", 7, 8),
        ("Site Reliability Engineering", "Concept", 7, 8),
        # Roles
        ("Full Stack Developer", "Role", 7, 16), ("Frontend Developer", "Role", 5, 12),
        ("Backend Developer", "Role", 6, 14), ("Data Scientist", "Role", 8, 20),
        ("ML Engineer", "Role", 8, 20), ("DevOps Engineer", "Role", 7, 16),
        ("Cloud Architect", "Role", 8, 20), ("Mobile Developer", "Role", 6, 14),
        # Soft Skills
        ("Communication", "Soft Skill", 3, 4), ("Leadership", "Soft Skill", 5, 8),
        ("Problem Solving", "Soft Skill", 4, 4), ("Teamwork", "Soft Skill", 2, 2),
    ]
    
    nodes = {}
    for name, cat, diff, weeks in skills_data:
        node, _ = SkillNode.objects.get_or_create(
            name=name, 
            defaults={'category': cat, 'difficulty_level': diff, 'learning_weeks': weeks}
        )
        nodes[name] = node

    print("Populating Edges (120+)...")
    edges_data = [
        # Frontend Track
        ("HTML/CSS", "JavaScript", 1, 1), ("JavaScript", "TypeScript", 2, 2),
        ("JavaScript", "React", 2, 3), ("JavaScript", "Vue.js", 2, 2),
        ("JavaScript", "Angular", 3, 3), ("JavaScript", "Svelte", 2, 2),
        ("JavaScript", "Node.js", 2, 2), ("TypeScript", "React", 1, 2),
        ("TypeScript", "Angular", 2, 2), ("TypeScript", "NestJS", 2, 3),
        ("React", "Next.js", 2, 2), ("React", "React Native", 2, 3),
        ("Dart", "Flutter", 2, 2),
        ("HTML/CSS", "Frontend Developer", 1, 1),
        ("React", "Frontend Developer", 2, 2),
        ("React", "Full Stack Developer", 2, 2),
        # Backend Track
        ("Python", "Django", 2, 2), ("Python", "Flask", 1, 1),
        ("Python", "FastAPI", 2, 2), ("Java", "Spring Boot", 3, 3),
        ("Ruby", "Rails", 2, 2), ("PHP", "Laravel", 2, 2),
        ("Node.js", "Express.js", 1, 1), ("Node.js", "NestJS", 2, 2),
        ("SQL", "Django", 1, 1), ("SQL", "Spring Boot", 1, 1),
        ("SQL", "PostgreSQL", 1, 1), ("SQL", "MySQL", 1, 1),
        ("Django", "Backend Developer", 2, 2),
        ("Spring Boot", "Backend Developer", 2, 2),
        ("Node.js", "Backend Developer", 2, 2),
        ("Backend Developer", "Full Stack Developer", 3, 3),
        ("Frontend Developer", "Full Stack Developer", 3, 3),
        # ML/AI Track
        ("Python", "Machine Learning", 3, 4), ("Python", "Scikit-learn", 2, 2),
        ("Machine Learning", "Deep Learning", 3, 4),
        ("Deep Learning", "PyTorch", 2, 3), ("Deep Learning", "TensorFlow", 2, 3),
        ("Deep Learning", "Computer Vision", 3, 4), ("Deep Learning", "NLP", 3, 4),
        ("Deep Learning", "Reinforcement Learning", 4, 5),
        ("NLP", "LLMs", 3, 3), ("LLMs", "RAG", 2, 2),
        ("LLMs", "Prompt Engineering", 1, 1),
        ("Python", "Data Scientist", 2, 2),
        ("Machine Learning", "Data Scientist", 3, 3),
        ("Machine Learning", "ML Engineer", 3, 3),
        ("Deep Learning", "ML Engineer", 3, 4),
        # Data Engineering Track
        ("Python", "Spark", 3, 4), ("Python", "Airflow", 2, 3),
        ("SQL", "Data Engineering", 2, 2), ("Spark", "Data Engineering", 3, 3),
        ("Kafka", "Data Engineering", 3, 3), ("Airflow", "Data Engineering", 2, 2),
        ("Hadoop", "Spark", 2, 2),
        # DevOps/Cloud Track
        ("Linux", "Bash", 1, 1), ("Linux", "Docker", 2, 2),
        ("Docker", "Kubernetes", 3, 4), ("Kubernetes", "Terraform", 2, 2),
        ("Git", "CI/CD", 1, 1), ("CI/CD", "GitHub Actions", 1, 1),
        ("CI/CD", "Jenkins", 2, 2), ("Kubernetes", "AWS", 2, 3),
        ("Kubernetes", "GCP", 2, 3), ("Kubernetes", "Azure", 2, 3),
        ("Prometheus", "Grafana", 1, 1), ("Kubernetes", "Prometheus", 2, 2),
        ("Docker", "DevOps", 2, 2), ("Kubernetes", "DevOps", 3, 3),
        ("Terraform", "DevOps", 2, 2), ("CI/CD", "DevOps", 2, 2),
        ("DevOps", "DevOps Engineer", 3, 3),
        ("AWS", "Cloud Architect", 3, 4), ("Azure", "Cloud Architect", 3, 4),
        ("System Design", "Cloud Architect", 3, 3),
        ("DevOps", "Site Reliability Engineering", 3, 3),
        # CS Fundamentals Track
        ("Data Structures", "Algorithms", 2, 3),
        ("Algorithms", "System Design", 3, 4),
        ("C++", "Algorithms", 2, 2), ("C++", "System Design", 3, 3),
        # Database Track
        ("PostgreSQL", "System Design", 2, 2), ("MongoDB", "NoSQL", 1, 1),
        ("Redis", "System Design", 2, 2), ("Cassandra", "NoSQL", 2, 3),
        ("Elasticsearch", "System Design", 2, 2),
        # Architecture Track
        ("API Design", "Microservices", 2, 2), ("API Design", "GraphQL", 2, 2),
        ("Microservices", "System Design", 2, 3), ("Kafka", "Microservices", 3, 3),
        ("RabbitMQ", "Microservices", 2, 2), ("Docker", "Microservices", 2, 2),
        # Cross-cutting
        ("Python", "AWS", 2, 3), ("Java", "AWS", 2, 3),
        ("Go", "Kubernetes", 2, 3), ("Go", "Docker", 2, 2),
        ("Go", "Microservices", 2, 3), ("Rust", "System Design", 3, 3),
        ("Scala", "Spark", 2, 3), ("Scala", "Kafka", 2, 3),
        ("Kotlin", "Android", 2, 3) if "Android" in nodes else ("Kotlin", "Spring Boot", 2, 3),
        ("Swift", "Mobile Developer", 3, 3),
        ("React Native", "Mobile Developer", 3, 3),
        ("Flutter", "Mobile Developer", 3, 3),
        # Systems / Game
        ("C++", "Unity", 3, 4), ("C++", "Unreal Engine", 4, 5),
        # Networking
        ("Networking", "System Design", 2, 2), ("Linux", "Networking", 2, 3),
        # Soft Skills
        ("Communication", "Leadership", 3, 4),
        ("Problem Solving", "Algorithms", 1, 1),
    ]
    
    for u, v, wt, wd in edges_data:
        if u in nodes and v in nodes:
            SkillEdge.objects.create(source=nodes[u], target=nodes[v], weight_time=wt, weight_difficulty=wd)

    print("Populating Signals (45+ market data)...")
    signals = [
        ("React", 0.90, 0.20, "Rising"), ("Python", 0.88, 0.18, "Rising"),
        ("AWS", 0.92, 0.15, "Rising"), ("Kubernetes", 0.88, 0.30, "Rising"),
        ("System Design", 0.95, 0.25, "Rising"), ("Rust", 0.78, 0.08, "Rising"),
        ("Machine Learning", 0.85, 0.28, "Rising"), ("TensorFlow", 0.75, 0.30, "Stable"),
        ("Docker", 0.87, 0.25, "Stable"), ("TypeScript", 0.88, 0.20, "Rising"),
        ("Go", 0.82, 0.15, "Rising"), ("Java", 0.85, 0.30, "Stable"),
        ("PostgreSQL", 0.80, 0.25, "Stable"), ("MongoDB", 0.72, 0.30, "Stable"),
        ("Redis", 0.78, 0.22, "Rising"), ("Kafka", 0.82, 0.18, "Rising"),
        ("PyTorch", 0.88, 0.22, "Rising"), ("Deep Learning", 0.85, 0.25, "Rising"),
        ("Node.js", 0.80, 0.30, "Stable"), ("GraphQL", 0.75, 0.25, "Rising"),
        ("Terraform", 0.85, 0.15, "Rising"), ("CI/CD", 0.82, 0.30, "Stable"),
        ("Microservices", 0.83, 0.28, "Stable"), ("SQL", 0.90, 0.40, "Stable"),
        ("Git", 0.95, 0.60, "Stable"), ("Algorithms", 0.92, 0.35, "Stable"),
        ("Data Structures", 0.90, 0.38, "Stable"), ("Azure", 0.82, 0.20, "Rising"),
        ("GCP", 0.78, 0.22, "Rising"), ("Spark", 0.80, 0.18, "Stable"),
        ("Angular", 0.72, 0.28, "Falling"), ("Vue.js", 0.70, 0.25, "Stable"),
        ("Next.js", 0.82, 0.18, "Rising"), ("Django", 0.78, 0.22, "Stable"),
        ("Spring Boot", 0.80, 0.25, "Stable"), ("FastAPI", 0.80, 0.12, "Rising"),
        ("Linux", 0.85, 0.35, "Stable"), ("LLMs", 0.92, 0.08, "Rising"),
        ("RAG", 0.90, 0.06, "Rising"), ("Prompt Engineering", 0.85, 0.10, "Rising"),
        ("NLP", 0.82, 0.20, "Rising"), ("Computer Vision", 0.78, 0.22, "Stable"),
        ("Scala", 0.72, 0.15, "Falling"), ("Kotlin", 0.78, 0.20, "Rising"),
        ("Swift", 0.75, 0.22, "Stable"), ("Flutter", 0.74, 0.20, "Rising"),
    ]
    
    for name, suc, fail, trend in signals:
        if name in nodes:
            SkillSignal.objects.create(
                skill=nodes[name], success_rate=suc, failure_rate=fail, demand_trend=trend
            )

    print("Updating PageRank Scores...")
    pr = PageRank()
    pr.update_db_scores()
    
    print(f"Population Done! Companies: {Company.objects.count()}, Skills: {SkillNode.objects.count()}, Edges: {SkillEdge.objects.count()}, Signals: {SkillSignal.objects.count()}")

if __name__ == '__main__':
    populate()
