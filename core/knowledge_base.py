"""
Massive Knowledge Base for LumosCareer RAG Engine.
Covers: career roles, technologies, interview prep, company profiles, wellness, industry trends, personality-career mapping.
"""

KNOWLEDGE_BASE = {

# ===================== CAREER ROLES =====================

"role_frontend_developer": {
    "title": "Frontend Developer",
    "description": "Frontend developers build the user-facing side of web applications using HTML, CSS, and JavaScript frameworks. They translate UI/UX designs into responsive, performant, accessible interfaces. Senior frontend devs architect component systems, manage state, optimize bundle sizes, and mentor juniors.",
    "salary_range": "$70,000 - $160,000 USD",
    "growth_trajectory": "Junior Frontend → Mid Frontend → Senior Frontend → Staff Engineer / Frontend Architect → Engineering Manager",
    "key_skills": ["HTML5", "CSS3", "JavaScript", "TypeScript", "React", "Vue.js", "Angular", "Webpack", "Vite", "Testing", "Accessibility", "Performance Optimization"],
    "industry_demand": "Very High — every company with a web presence needs frontend engineers. Remote-friendly roles widely available.",
    "day_in_life": "Morning standup, code reviews, implement new UI features, write unit tests, collaborate with designers, debug cross-browser issues, optimize Core Web Vitals, deploy to staging.",
},

"role_backend_developer": {
    "title": "Backend Developer",
    "description": "Backend developers design and build server-side logic, databases, APIs, and system architecture. They handle authentication, data processing, business logic, integrations with third-party services, and ensure systems scale under load. Focus areas include API design, database optimization, caching strategies, and microservices architecture.",
    "salary_range": "$75,000 - $170,000 USD",
    "growth_trajectory": "Junior Backend → Mid Backend → Senior Backend → Staff Engineer / Architect → Principal Engineer → VP Engineering",
    "key_skills": ["Python", "Java", "Node.js", "Go", "SQL", "PostgreSQL", "Redis", "Docker", "REST APIs", "GraphQL", "Microservices", "Message Queues"],
    "industry_demand": "Very High — backend engineering is foundational to all software products.",
},

"role_fullstack_developer": {
    "title": "Full-Stack Developer",
    "description": "Full-stack developers work across the entire web application stack — frontend UI, backend APIs, databases, and deployment. They are versatile engineers who can build complete features end-to-end. Startups particularly value full-stack developers for their ability to ship independently.",
    "salary_range": "$80,000 - $175,000 USD",
    "key_skills": ["JavaScript", "TypeScript", "React", "Node.js", "Python", "Django", "PostgreSQL", "Docker", "AWS", "Git", "CI/CD"],
    "industry_demand": "Extremely High — startups and mid-size companies prefer full-stack engineers for velocity.",
},

"role_data_scientist": {
    "title": "Data Scientist",
    "description": "Data scientists extract actionable insights from large datasets using statistical analysis, machine learning, and data visualization. They design experiments, build predictive models, communicate findings to stakeholders, and drive data-informed decisions. Core work involves hypothesis testing, feature engineering, model training/evaluation, and deploying ML pipelines.",
    "salary_range": "$85,000 - $180,000 USD",
    "key_skills": ["Python", "R", "SQL", "Pandas", "NumPy", "Scikit-learn", "TensorFlow", "PyTorch", "Statistics", "Data Visualization", "A/B Testing", "Feature Engineering"],
    "industry_demand": "High — growing across finance, healthcare, e-commerce, and tech.",
},

"role_ml_engineer": {
    "title": "Machine Learning Engineer",
    "description": "ML engineers bridge the gap between data science research and production systems. They build scalable ML pipelines, optimize model inference, deploy models to production, monitor model performance, and handle data drift. They work with MLOps tools, distributed training, and model serving infrastructure.",
    "salary_range": "$100,000 - $220,000 USD",
    "key_skills": ["Python", "PyTorch", "TensorFlow", "MLflow", "Kubeflow", "Docker", "Kubernetes", "Spark", "Feature Stores", "Model Serving", "A/B Testing"],
    "industry_demand": "Very High — one of the fastest growing roles in tech, especially with the AI boom.",
},

"role_devops_engineer": {
    "title": "DevOps Engineer",
    "description": "DevOps engineers build and maintain CI/CD pipelines, infrastructure automation, monitoring systems, and deployment workflows. They bridge development and operations, ensuring reliable, fast, and secure software delivery. Key responsibilities include infrastructure as code, container orchestration, observability, and incident response.",
    "salary_range": "$90,000 - $175,000 USD",
    "key_skills": ["Linux", "Docker", "Kubernetes", "Terraform", "Ansible", "Jenkins", "GitHub Actions", "AWS", "GCP", "Azure", "Prometheus", "Grafana", "Bash"],
    "industry_demand": "Very High — critical for any organization practicing continuous delivery.",
},

"role_cloud_architect": {
    "title": "Cloud Architect",
    "description": "Cloud architects design and oversee an organization's cloud computing strategy, including cloud adoption plans, cloud application design, and cloud management and monitoring. They evaluate cloud services, design highly available and fault-tolerant architectures, optimize costs, and ensure security compliance.",
    "salary_range": "$120,000 - $220,000 USD",
    "key_skills": ["AWS", "Azure", "GCP", "Terraform", "Networking", "Security", "Microservices", "Serverless", "Cost Optimization", "Multi-cloud"],
    "industry_demand": "High — enterprises are accelerating cloud migration.",
},

"role_mobile_developer": {
    "title": "Mobile Developer",
    "description": "Mobile developers build native or cross-platform applications for iOS and Android devices. They handle UI implementation, platform APIs, offline storage, push notifications, app store deployment, and performance optimization for resource-constrained devices.",
    "salary_range": "$75,000 - $165,000 USD",
    "key_skills": ["Swift", "Kotlin", "React Native", "Flutter", "Dart", "iOS SDK", "Android SDK", "REST APIs", "SQLite", "Firebase"],
    "industry_demand": "Steady High — mobile-first strategies continue to dominate.",
},

"role_cybersecurity_analyst": {
    "title": "Cybersecurity Analyst",
    "description": "Cybersecurity analysts protect an organization's computer systems and networks from cyber threats. They monitor for security breaches, investigate incidents, implement security measures, conduct vulnerability assessments, and ensure compliance with security standards like SOC2, ISO27001, and GDPR.",
    "salary_range": "$80,000 - $160,000 USD",
    "key_skills": ["Network Security", "SIEM", "Penetration Testing", "Incident Response", "Firewalls", "Encryption", "Linux", "Python", "Compliance", "Threat Intelligence"],
    "industry_demand": "Critical — cybersecurity talent shortage is one of the largest in tech.",
},

"role_product_manager": {
    "title": "Product Manager (Technical)",
    "description": "Technical product managers define product vision, strategy, and roadmap. They prioritize features based on user research, business metrics, and technical feasibility. They work at the intersection of business, design, and engineering — writing PRDs, running sprints, analyzing metrics, and making trade-off decisions.",
    "salary_range": "$95,000 - $200,000 USD",
    "key_skills": ["Product Strategy", "User Research", "Data Analysis", "SQL", "A/B Testing", "Agile/Scrum", "Stakeholder Management", "Technical Communication", "Roadmapping"],
    "industry_demand": "High — PMs are critical for product-led growth companies.",
},

"role_ai_llm_engineer": {
    "title": "AI/LLM Engineer",
    "description": "AI/LLM engineers specialize in building applications powered by large language models. They design RAG pipelines, fine-tune models, implement prompt engineering strategies, build agent systems, integrate vector databases, and optimize inference costs. This is one of the newest and most in-demand roles in tech.",
    "salary_range": "$120,000 - $250,000 USD",
    "key_skills": ["Python", "LLMs", "RAG", "Prompt Engineering", "LangChain", "Vector Databases", "Fine-tuning", "OpenAI API", "Hugging Face", "Embeddings", "Agent Frameworks"],
    "industry_demand": "Explosive — the fastest growing role in tech as of 2024-2026.",
},

"role_data_engineer": {
    "title": "Data Engineer",
    "description": "Data engineers design, build, and maintain the infrastructure and pipelines that collect, store, and serve data at scale. They work with ETL/ELT processes, data warehouses, streaming systems, and ensure data quality and governance. They enable data scientists and analysts to do their work effectively.",
    "salary_range": "$90,000 - $185,000 USD",
    "key_skills": ["Python", "SQL", "Spark", "Airflow", "Kafka", "dbt", "Snowflake", "BigQuery", "AWS", "Docker", "Data Modeling", "ETL"],
    "industry_demand": "Very High — data infrastructure is foundational to modern companies.",
},

"role_sre": {
    "title": "Site Reliability Engineer (SRE)",
    "description": "SREs apply software engineering principles to infrastructure and operations problems. They build automated systems for deployment, monitoring, and incident response. They define SLIs/SLOs/SLAs, manage on-call rotations, conduct postmortems, and optimize system reliability. Pioneered by Google.",
    "salary_range": "$100,000 - $200,000 USD",
    "key_skills": ["Linux", "Python", "Go", "Kubernetes", "Terraform", "Prometheus", "Grafana", "Incident Management", "Chaos Engineering", "Distributed Systems"],
    "industry_demand": "High — companies adopt SRE practices as they scale.",
},

"role_game_developer": {
    "title": "Game Developer",
    "description": "Game developers create interactive entertainment software for various platforms. They work with game engines, implement gameplay mechanics, physics simulations, AI behavior, rendering pipelines, and multiplayer networking. Specializations include gameplay programming, engine programming, graphics programming, and tools development.",
    "salary_range": "$60,000 - $150,000 USD",
    "key_skills": ["C++", "C#", "Unity", "Unreal Engine", "3D Math", "Physics", "Shaders", "Networking", "Game Design", "Optimization"],
    "industry_demand": "Moderate — competitive but growing with mobile and indie markets.",
},

"role_blockchain_developer": {
    "title": "Blockchain / Web3 Developer",
    "description": "Blockchain developers build decentralized applications (dApps), smart contracts, and DeFi protocols. They work with blockchain platforms like Ethereum, Solana, and Polygon, implementing token standards, consensus mechanisms, and cryptographic protocols.",
    "salary_range": "$90,000 - $200,000 USD",
    "key_skills": ["Solidity", "Rust", "JavaScript", "Web3.js", "Ethers.js", "Smart Contracts", "DeFi", "NFTs", "Cryptography", "Consensus Algorithms"],
    "industry_demand": "Cyclical — high during crypto booms, stable demand for infrastructure builders.",
},

# ===================== TECHNOLOGY DEEP DIVES =====================

"tech_python": {
    "name": "Python",
    "category": "Programming Language",
    "description": "Python is the world's most popular general-purpose programming language, known for its readable syntax, vast ecosystem, and versatility. It dominates in data science, machine learning, web development (Django/Flask/FastAPI), automation, scripting, and scientific computing. Python's simplicity makes it ideal for beginners while its powerful libraries make it essential for experts.",
    "why_learn": "Python appears in more job descriptions than any other language. It's the #1 language for AI/ML, data science, and automation. Learning Python opens doors to the widest range of tech careers.",
    "key_frameworks": ["Django", "Flask", "FastAPI", "Pandas", "NumPy", "PyTorch", "TensorFlow", "Scikit-learn", "SQLAlchemy", "Celery"],
    "learning_path": "Syntax basics → Data structures → OOP → File I/O → Libraries (requests, os) → Web framework (Django/Flask) → Specialization (ML, Data, DevOps)",
    "industry_adoption": "Used by Google, Instagram, Spotify, Netflix, Dropbox, Reddit, NASA, and virtually every AI/ML company.",
},

"tech_javascript": {
    "name": "JavaScript",
    "category": "Programming Language",
    "description": "JavaScript is the language of the web — the only programming language that runs natively in web browsers. With Node.js, it also runs on servers. JavaScript powers interactive web pages, server APIs, mobile apps (React Native), desktop apps (Electron), and even machine learning (TensorFlow.js). Its event-driven, non-blocking architecture makes it ideal for real-time applications.",
    "why_learn": "JavaScript is unavoidable for web development. It has the largest package ecosystem (npm) and one of the largest developer communities worldwide.",
    "key_frameworks": ["React", "Vue.js", "Angular", "Next.js", "Express.js", "NestJS", "Svelte", "Electron", "React Native"],
    "industry_adoption": "Used by every company with a web presence. Meta, Netflix, Uber, Airbnb, and LinkedIn are heavy JavaScript shops.",
},

"tech_react": {
    "name": "React",
    "category": "Frontend Framework",
    "description": "React is a JavaScript library for building user interfaces, created by Meta (Facebook). It uses a component-based architecture with a virtual DOM for efficient rendering. React's ecosystem includes React Router for navigation, Redux/Zustand for state management, and Next.js for server-side rendering. It's the most popular frontend framework with the largest job market.",
    "why_learn": "React dominates the frontend job market with 40%+ of frontend roles requiring it. Its component model is the industry standard.",
    "learning_path": "JavaScript fundamentals → JSX → Components & Props → State & Hooks → Context API → React Router → State Management → Next.js → Testing",
},

"tech_docker": {
    "name": "Docker",
    "category": "DevOps Tool",
    "description": "Docker is a containerization platform that packages applications and their dependencies into lightweight, portable containers. Containers ensure consistency across development, testing, and production environments. Docker eliminates 'works on my machine' problems and is foundational to modern microservices architecture and CI/CD pipelines.",
    "why_learn": "Docker is a baseline requirement for most senior engineering roles. It's essential for DevOps, cloud deployment, and modern development workflows.",
    "learning_path": "Docker basics → Dockerfiles → Docker Compose → Multi-stage builds → Networking → Volumes → Docker in CI/CD → Kubernetes",
},

"tech_kubernetes": {
    "name": "Kubernetes",
    "category": "Container Orchestration",
    "description": "Kubernetes (K8s) is an open-source container orchestration platform originally designed by Google. It automates deployment, scaling, and management of containerized applications. K8s handles load balancing, service discovery, storage orchestration, automated rollouts/rollbacks, and self-healing. It's the industry standard for running microservices at scale.",
    "why_learn": "Kubernetes is the backbone of cloud-native infrastructure. Knowledge of K8s is highly valued and commands premium salaries.",
    "learning_path": "Docker → K8s concepts (Pods, Services, Deployments) → kubectl → Helm → Ingress → Storage → RBAC → Operators → Service Mesh",
},

"tech_aws": {
    "name": "Amazon Web Services (AWS)",
    "category": "Cloud Platform",
    "description": "AWS is the world's leading cloud computing platform with 200+ services spanning compute (EC2, Lambda), storage (S3, EBS), databases (RDS, DynamoDB), networking (VPC, CloudFront), AI/ML (SageMaker), and more. AWS certifications (Solutions Architect, Developer, DevOps) are among the most valuable in IT.",
    "why_learn": "AWS holds 31% cloud market share. Most companies use AWS, making it the most in-demand cloud skill.",
    "key_services": ["EC2", "S3", "Lambda", "RDS", "DynamoDB", "SQS", "SNS", "CloudFormation", "ECS", "EKS", "CloudWatch", "IAM"],
},

"tech_tensorflow": {
    "name": "TensorFlow",
    "category": "ML Framework",
    "description": "TensorFlow is Google's open-source machine learning framework for building and deploying ML models. It supports deep learning, neural networks, computer vision, NLP, and reinforcement learning. TensorFlow provides production-grade tools including TensorFlow Serving, TensorFlow Lite (mobile), and TensorFlow.js (browser).",
    "why_learn": "TensorFlow is widely used in production ML systems. Google, Airbnb, Intel, and many enterprises use TensorFlow.",
},

"tech_pytorch": {
    "name": "PyTorch",
    "category": "ML Framework",
    "description": "PyTorch is Meta's open-source deep learning framework known for its dynamic computation graph and Pythonic API. It's the dominant framework in ML research and increasingly in production. PyTorch excels at rapid prototyping, debugging, and experimentation. The ecosystem includes torchvision, torchaudio, and Hugging Face integration.",
    "why_learn": "PyTorch is the #1 framework in ML/AI research. 80%+ of new ML papers use PyTorch. Essential for AI/LLM engineering roles.",
},

"tech_sql": {
    "name": "SQL",
    "category": "Query Language",
    "description": "SQL (Structured Query Language) is the standard language for managing and querying relational databases. Every backend developer, data scientist, data engineer, and analyst needs SQL. Key concepts include JOIN operations, subqueries, window functions, indexing, query optimization, transactions, and stored procedures.",
    "why_learn": "SQL is the most universally required technical skill across all data-related roles. It appears in job descriptions more than any other skill.",
},

"tech_git": {
    "name": "Git",
    "category": "Version Control",
    "description": "Git is the industry-standard distributed version control system. It tracks changes in source code, enables collaboration through branching and merging, and is essential for team-based software development. GitHub, GitLab, and Bitbucket are built on Git. Key concepts include branching strategies (GitFlow, trunk-based), rebasing, cherry-picking, and conflict resolution.",
    "why_learn": "Git is non-negotiable for software development. Every engineering role requires Git proficiency.",
},

"tech_system_design": {
    "name": "System Design",
    "category": "Architecture Concept",
    "description": "System design is the process of defining the architecture, components, modules, interfaces, and data flow of a system to satisfy specified requirements. Topics include: load balancing, caching (Redis, CDN), database sharding, message queues (Kafka, RabbitMQ), microservices vs monoliths, CAP theorem, consistency patterns, rate limiting, and API gateway design. System design interviews are the most important round for senior engineering roles.",
    "why_learn": "System design knowledge separates senior engineers from juniors. It's the key interview topic for L5+ roles at FAANG companies.",
},

"tech_algorithms_ds": {
    "name": "Data Structures & Algorithms",
    "category": "Computer Science Fundamentals",
    "description": "Data structures (arrays, linked lists, trees, graphs, hash tables, heaps, tries) and algorithms (sorting, searching, BFS/DFS, dynamic programming, greedy, divide-and-conquer, backtracking) are the foundation of computer science. They are the primary topic in technical coding interviews at top companies. Mastery requires understanding time/space complexity (Big-O notation) and pattern recognition.",
    "why_learn": "DSA is the gatekeeper for technical interviews at FAANG, unicorns, and most tech companies. LeetCode-style problems test these skills.",
},

"tech_graphql": {
    "name": "GraphQL",
    "category": "API Technology",
    "description": "GraphQL is a query language and runtime for APIs developed by Meta. Unlike REST, GraphQL lets clients request exactly the data they need in a single query, eliminating over-fetching and under-fetching. It uses a strong type system, introspection, and real-time subscriptions.",
    "why_learn": "GraphQL is increasingly adopted by companies needing flexible, efficient APIs. Meta, GitHub, Shopify, and Twitter use GraphQL.",
},

"tech_rust": {
    "name": "Rust",
    "category": "Systems Programming Language",
    "description": "Rust is a systems programming language focused on safety, speed, and concurrency. Its ownership model eliminates memory bugs at compile time without a garbage collector. Rust is used for performance-critical systems, WebAssembly, operating systems, game engines, and blockchain. It's been voted the most loved programming language for 7+ consecutive years.",
    "why_learn": "Rust skills command premium salaries. It's increasingly adopted for infrastructure, cloud services, and performance-critical applications.",
},

"tech_go": {
    "name": "Go (Golang)",
    "category": "Programming Language",
    "description": "Go is Google's statically typed, compiled language designed for simplicity, performance, and concurrency. Its goroutines make concurrent programming accessible. Go excels at building microservices, CLI tools, DevOps tooling, and cloud infrastructure. Docker, Kubernetes, Terraform, and Prometheus are all written in Go.",
    "why_learn": "Go is the language of cloud infrastructure. If you want to work in DevOps, SRE, or cloud-native development, Go is essential.",
},

"tech_typescript": {
    "name": "TypeScript",
    "category": "Programming Language",
    "description": "TypeScript is a typed superset of JavaScript that compiles to plain JavaScript. It adds static type checking, interfaces, generics, and enums to JavaScript, catching errors at compile time rather than runtime. TypeScript is now the standard for large-scale JavaScript applications.",
    "why_learn": "TypeScript is rapidly replacing JavaScript in enterprise codebases. Most React, Angular, and Node.js projects now use TypeScript.",
},

"tech_redis": {
    "name": "Redis",
    "category": "In-Memory Database",
    "description": "Redis is an in-memory data structure store used as a database, cache, message broker, and queue. It supports strings, hashes, lists, sets, sorted sets, streams, and more. Redis is critical for caching, session management, real-time leaderboards, rate limiting, and pub/sub messaging.",
    "why_learn": "Redis is used by virtually every large-scale web application. Understanding caching patterns is essential for system design interviews.",
},

"tech_mongodb": {
    "name": "MongoDB",
    "category": "NoSQL Database",
    "description": "MongoDB is a document-oriented NoSQL database that stores data in flexible, JSON-like BSON documents. It's designed for scalability, high availability, and developer productivity. MongoDB Atlas provides a managed cloud service. Use cases include content management, real-time analytics, IoT, and mobile apps.",
    "why_learn": "MongoDB is the most popular NoSQL database. It's widely used in JavaScript/Node.js stacks and startups.",
},

"tech_kafka": {
    "name": "Apache Kafka",
    "category": "Streaming Platform",
    "description": "Apache Kafka is a distributed event streaming platform used for high-throughput, fault-tolerant, real-time data pipelines. It handles publish-subscribe messaging, event sourcing, stream processing, and log aggregation. Kafka is critical infrastructure at companies processing millions of events per second.",
    "why_learn": "Kafka knowledge is essential for data engineering, backend architecture, and event-driven microservices roles.",
},

"tech_terraform": {
    "name": "Terraform",
    "category": "Infrastructure as Code",
    "description": "Terraform is HashiCorp's infrastructure as code tool that lets you define cloud resources in declarative configuration files. It supports AWS, Azure, GCP, and 3000+ providers. Terraform enables version-controlled, reproducible, and auditable infrastructure management.",
    "why_learn": "Terraform is the de facto standard for infrastructure as code. Essential for DevOps, SRE, and cloud engineering roles.",
},

"tech_llms_rag": {
    "name": "Large Language Models & RAG",
    "category": "AI Technology",
    "description": "Large Language Models (LLMs) like GPT-4, Claude, Gemini, and Llama are transformer-based neural networks trained on vast text corpora. RAG (Retrieval-Augmented Generation) enhances LLMs by retrieving relevant documents from a knowledge base before generating responses, reducing hallucinations and enabling domain-specific expertise. Key concepts include embeddings, vector databases, chunking strategies, prompt engineering, fine-tuning, and agent frameworks.",
    "why_learn": "LLM/RAG engineering is the hottest skill in tech. Companies are investing billions in AI integration, creating unprecedented demand.",
},

"tech_mcp": {
    "name": "Model Context Protocol (MCP)",
    "category": "AI Architecture",
    "description": "MCP (Model Context Protocol) is an open standard developed by Anthropic that standardizes how AI models interact with external data sources and tools. Like USB-C for AI, MCP provides a universal protocol for connecting LLMs to databases, APIs, file systems, and services. MCP uses a client-server architecture with three core primitives: Tools (executable actions), Resources (read-only data), and Prompts (reusable templates). MCP enables building composable, interoperable AI agent systems.",
    "why_learn": "MCP is becoming the standard protocol for AI agent integration. Understanding MCP architecture is essential for modern AI engineering.",
},

# ===================== INTERVIEW PREPARATION =====================

"interview_faang_preparation": {
    "title": "FAANG Interview Preparation Guide",
    "description": "FAANG (Facebook/Meta, Amazon, Apple, Netflix, Google) interviews follow a structured multi-round process: Phone Screen → Technical Rounds (2-4) → System Design → Behavioral. Each round is typically 45-60 minutes. Technical rounds focus on data structures, algorithms, and coding fluency. System design rounds test architecture skills for senior roles. Behavioral rounds use the STAR method (Situation, Task, Action, Result) extensively.",
    "coding_tips": "Practice 150+ LeetCode problems covering arrays, strings, trees, graphs, dynamic programming, and sliding window patterns. Focus on medium difficulty. Time yourself to 25 minutes per problem. Always discuss time/space complexity.",
    "system_design_tips": "Study: URL shortener, Twitter/X feed, chat system, notification service, rate limiter, distributed cache. Use the framework: Requirements → API Design → Data Model → High-Level Design → Deep Dive → Bottlenecks.",
    "behavioral_tips": "Prepare 8-10 STAR stories covering: leadership, conflict resolution, failure, ambiguity, tight deadlines, cross-team collaboration, customer obsession (Amazon), and innovation.",
},

"interview_behavioral_star": {
    "title": "STAR Method for Behavioral Interviews",
    "description": "The STAR method is the gold standard for answering behavioral interview questions. Situation: Set the context (where, when, what role). Task: Describe your specific responsibility. Action: Detail the steps YOU took (use 'I' not 'we'). Result: Quantify the outcome (metrics, impact, learning). Common behavioral questions: Tell me about a time you disagreed with your manager. Describe a situation where you had to meet a tight deadline. Give an example of when you failed. How do you handle ambiguity?",
},

"interview_system_design_patterns": {
    "title": "System Design Interview Patterns",
    "description": "Key system design patterns every senior engineer should know: (1) Load Balancing — distribute traffic across servers (round robin, least connections, consistent hashing). (2) Caching — reduce latency with Redis/Memcached (cache-aside, write-through, write-behind). (3) Database Sharding — horizontal partitioning for scale (range-based, hash-based, directory-based). (4) Message Queues — decouple services with Kafka/RabbitMQ (pub-sub, point-to-point). (5) API Gateway — single entry point for microservices (rate limiting, auth, routing). (6) CDN — serve static content from edge locations. (7) CQRS — separate read and write models. (8) Event Sourcing — store state changes as events.",
},

"interview_coding_patterns": {
    "title": "Top Coding Interview Patterns",
    "description": "Master these 15 coding patterns to solve 90% of interview problems: (1) Two Pointers (2) Sliding Window (3) Fast & Slow Pointers (4) Merge Intervals (5) Cyclic Sort (6) In-place Linked List Reversal (7) Tree BFS (8) Tree DFS (9) Two Heaps (10) Subsets/Backtracking (11) Modified Binary Search (12) Top K Elements (13) K-way Merge (14) Topological Sort (15) Dynamic Programming (0/1 Knapsack, Unbounded, LCS, LIS). Practice identifying which pattern applies to each problem before coding.",
},

"interview_technical_questions_common": {
    "title": "Most Common Technical Interview Questions",
    "description": "Frequently asked questions across tech companies: Explain the difference between processes and threads. What is a deadlock and how do you prevent it? Explain REST vs GraphQL. What is the CAP theorem? Describe ACID properties. What is eventual consistency? Explain TCP vs UDP. What is DNS and how does it work? Describe the HTTP request lifecycle. What is a CDN? Explain indexing in databases. What is normalization vs denormalization? Describe OAuth 2.0 flow. What is the difference between authentication and authorization? Explain microservices vs monolith tradeoffs.",
},

"interview_salary_negotiation": {
    "title": "Salary Negotiation Strategies for Tech",
    "description": "Research market rates on levels.fyi, Glassdoor, and Blind. Always negotiate — initial offers are rarely final. Key strategies: (1) Never give a number first, ask for the range. (2) Use competing offers as leverage. (3) Negotiate total compensation (base + equity + signing bonus + benefits). (4) Ask for the offer in writing. (5) Practice your negotiation script. (6) Be willing to walk away. (7) Negotiate for equity refresh, remote flexibility, or title if salary is capped. (8) Typical counter-offer success rate is 85%+ — companies expect negotiation.",
},

# ===================== COMPANY PROFILES =====================

"company_google": {
    "name": "Google (Alphabet)",
    "culture": "Engineering-driven culture focused on innovation, data-driven decisions, and '20% time' for side projects. Known for high autonomy, extensive internal tooling, strong emphasis on code quality and design docs. L3-L8 leveling system.",
    "interview_process": "Phone screen → 4-5 onsite rounds (2 coding, 1 system design for senior, 1 behavioral/Googleyness, 1 mixed). Focus on algorithms, scalability, and 'Googleyness' (leadership, role-related knowledge).",
    "tech_stack": "Python, Go, C++, Java, Kubernetes, Borg, Spanner, BigTable, TensorFlow, Angular, Protobuf, gRPC",
    "tips": "Study Google's engineering practices blog. Practice with Neetcode 150. Prepare for open-ended system design. Demonstrate 'Googleyness' — intellectual humility, bias to action, collaborative problem-solving.",
},

"company_amazon": {
    "name": "Amazon",
    "culture": "Highly customer-obsessed culture driven by 16 Leadership Principles. Fast-paced, autonomous '2-pizza teams', strong ownership mentality. Promotion-driven. Bar raiser program in interviews.",
    "interview_process": "Phone screen → 4-5 loop interviews (2 coding, 1-2 behavioral focused on Leadership Principles, 1 system design). Bar raiser is an independent evaluator.",
    "tech_stack": "Java, Python, AWS (all services), DynamoDB, React, TypeScript, Kotlin",
    "tips": "Memorize all 16 Leadership Principles and prepare 2 STAR stories per principle. 'Customer Obsession', 'Ownership', and 'Dive Deep' are most tested. Use the Amazon interview prep guide.",
},

"company_meta": {
    "name": "Meta (Facebook)",
    "culture": "Move fast culture (formerly 'move fast and break things', now 'move fast with stable infrastructure'). Strong engineering culture, internal bootcamp for new hires, emphasis on impact and velocity. IC track up to E9.",
    "interview_process": "Phone screen → 3-4 onsite (2 coding, 1 system design, 1 behavioral). Coding rounds heavily focus on optimal solutions and clean code. System design uses a specific Meta framework.",
    "tech_stack": "Hack/PHP, React, GraphQL, PyTorch, C++, Python, Presto, Cassandra",
    "tips": "Practice on LeetCode focusing on Meta-tagged problems. Master React concepts. Prepare for product sense questions. Emphasize impact metrics in behavioral answers.",
},

"company_apple": {
    "name": "Apple",
    "culture": "Secrecy-oriented, design-obsessed culture. Strong focus on craftsmanship and attention to detail. Siloed teams with limited cross-team visibility. Hardware-software integration mindset.",
    "interview_process": "Phone screen → Multiple onsite rounds (varies by team). Mix of coding, system design, and domain-specific questions. Behavioral focuses on passion for Apple products.",
    "tech_stack": "Swift, Objective-C, C++, Python, Metal, CoreML, SwiftUI",
},

"company_microsoft": {
    "name": "Microsoft",
    "culture": "Growth mindset culture (transformed under Satya Nadella). Emphasis on learning, empathy, and collaboration. Strong work-life balance compared to other FAANG. Azure and AI are strategic priorities.",
    "interview_process": "Phone screen → 4-5 onsite (coding, system design, behavioral). 'As-appropriate' round with hiring manager. Focus on problem-solving process as much as solution.",
    "tech_stack": "C#, .NET, TypeScript, Azure, React, Python, SQL Server, PowerShell",
},

"company_netflix": {
    "name": "Netflix",
    "culture": "High-performance culture with extreme freedom and responsibility. 'Keeper test' — managers ask if they'd fight to keep each employee. Top-of-market compensation, no formal performance reviews. Adults-only culture with minimal process.",
    "interview_process": "Recruiter phone screen → Technical phone screen → Onsite (4-5 rounds). Emphasis on culture fit, self-direction, and senior-level thinking even for mid-level roles.",
    "tech_stack": "Java, Spring Boot, Python, Node.js, React, AWS, Cassandra, Kafka, Zuul",
},

"company_openai": {
    "name": "OpenAI",
    "culture": "Mission-driven culture focused on ensuring AGI benefits all of humanity. Research-oriented with strong emphasis on safety and alignment. Fast-moving with high expectations for impact.",
    "interview_process": "Highly selective. Multiple technical rounds focusing on ML fundamentals, systems programming, and research contributions. Strong emphasis on alignment with mission.",
    "tech_stack": "Python, PyTorch, Kubernetes, Ray, React, Rust, CUDA, Triton",
},

"company_stripe": {
    "name": "Stripe",
    "culture": "Intensely analytical, writing-oriented culture. Strong emphasis on clear communication, rigorous thinking, and 'increasing the GDP of the internet'. Known for exceptional engineering talent density.",
    "interview_process": "Phone screen → Take-home project → Onsite (coding, system design, debugging, collaboration, manager). The debugging round is unique to Stripe.",
    "tech_stack": "Ruby, Java, Go, Python, React, TypeScript, AWS, Kafka",
},

# ===================== MENTAL WELLNESS =====================

"wellness_imposter_syndrome": {
    "title": "Dealing with Imposter Syndrome in Tech",
    "description": "Imposter syndrome affects 70% of tech professionals at some point. Signs include: feeling like a fraud despite achievements, attributing success to luck, fear of being 'found out', comparing yourself to others constantly. Strategies to overcome: (1) Keep a 'wins journal' documenting achievements. (2) Recognize that not knowing everything is normal — even senior engineers Google basics daily. (3) Share your feelings — most people experience this. (4) Focus on growth, not perfection. (5) Mentor others — teaching solidifies your own knowledge. (6) Remember: you were hired because someone believed in you.",
},

"wellness_burnout_prevention": {
    "title": "Preventing and Recovering from Burnout",
    "description": "Tech burnout manifests as chronic exhaustion, cynicism, and reduced effectiveness. WHO classifies it as an occupational phenomenon. Prevention strategies: (1) Set hard boundaries — log off at a consistent time. (2) Take all your PTO. (3) Practice 'digital sabbath' — one day per week offline. (4) Exercise regularly — 30 min/day reduces burnout risk by 40%. (5) Build non-work identity — hobbies, relationships, community. (6) Say no — protect your time and energy. Recovery: Take extended time off, seek therapy, consider a role change if the environment is toxic. Remember: no job is worth your health.",
},

"wellness_career_anxiety": {
    "title": "Managing Career Transition Anxiety",
    "description": "Career transitions trigger anxiety because they involve identity, financial security, and social status. Common during layoffs, pivots, or first job searches. CBT techniques that help: (1) Cognitive restructuring — challenge catastrophic thoughts ('I'll never find a job' → 'Job searches take time, and I'm building skills daily'). (2) Behavioral activation — small daily actions (one application, one networking message). (3) Exposure hierarchy — gradually tackle feared activities. (4) Mindfulness — observe anxiety without judgment. (5) Social support — lean on your network. (6) Professional help — therapy is a sign of strength.",
},

"wellness_interview_anxiety": {
    "title": "Managing Interview Anxiety",
    "description": "Interview anxiety is one of the most common forms of performance anxiety. Physiological symptoms include racing heart, sweaty palms, and mind blanks. Preparation strategies: (1) Mock interviews — practice reduces anxiety by 60%. (2) Box breathing before interviews — 4 seconds in, 4 hold, 4 out, 4 hold. (3) Power posing — 2 minutes of expansive postures reduces cortisol. (4) Reframe anxiety as excitement — your body can't tell the difference. (5) Prepare questions for the interviewer — gives you control. (6) Remember: interviewing is a skill that improves with practice, not an IQ test.",
},

"wellness_cbt_techniques": {
    "title": "CBT Techniques for Career Stress",
    "description": "Cognitive Behavioral Therapy (CBT) is evidence-based therapy highly effective for stress and anxiety. Core techniques: (1) Thought records — write down negative thoughts, identify distortions (catastrophizing, black-and-white thinking, mind reading), and generate balanced alternatives. (2) Behavioral experiments — test your predictions ('If I ask for help, people will think I'm incompetent' → ask and observe the actual response). (3) Activity scheduling — plan pleasurable and mastery activities. (4) Graded exposure — face feared situations gradually. (5) Problem-solving — define the problem, brainstorm solutions, evaluate, implement, review.",
},

# ===================== INDUSTRY TRENDS =====================

"trend_ai_ml_market": {
    "title": "AI/ML Industry Trends 2024-2026",
    "description": "The AI market is projected to reach $1.8 trillion by 2030. Key trends: (1) LLM commoditization — open-source models approaching proprietary quality. (2) RAG over fine-tuning — enterprises prefer RAG for domain-specific AI. (3) AI agents — autonomous systems that plan, reason, and execute multi-step tasks. (4) Multimodal AI — models handling text, image, video, and audio. (5) Edge AI — running models on devices. (6) AI safety and regulation — EU AI Act, responsible AI practices. (7) Synthetic data — AI-generated training data. Job impact: AI/ML engineer roles grew 74% year-over-year.",
},

"trend_cloud_computing": {
    "title": "Cloud Computing Trends",
    "description": "Cloud computing market exceeds $600B annually. Key trends: (1) Multi-cloud and hybrid cloud adoption increasing. (2) Serverless computing growing 25% annually. (3) FinOps — cloud cost optimization becoming a discipline. (4) Platform engineering — internal developer platforms (IDPs) replacing raw cloud access. (5) Cloud-native security (CNAPP, CSPM). (6) AWS leads with 31%, Azure at 25%, GCP at 11%. (7) Edge computing converging with cloud.",
},

"trend_remote_work": {
    "title": "Remote Work in Tech",
    "description": "Post-pandemic tech workforce: 28% fully remote, 53% hybrid, 19% in-office. Remote-first companies: GitLab, Automattic, Zapier, Buffer, InVision. Key data: Remote workers report 20% higher productivity. Salary adjustment for location is controversial. Remote roles receive 3x more applications. Best practices: async communication, documentation-first culture, regular video touchpoints, intentional social bonding.",
},

# ===================== PERSONALITY-CAREER MAPPING =====================

"mbti_intj": {
    "type": "INTJ - The Architect",
    "career_fit": "INTJ personalities excel in roles requiring strategic thinking, systems analysis, and independent problem-solving. Top career matches: Software Architect, Data Scientist, Research Engineer, Technical Lead, Cybersecurity Analyst, Quantitative Analyst. INTJs prefer depth over breadth, autonomous work environments, and intellectually challenging problems.",
    "strengths": "Strategic vision, analytical thinking, independence, determination, high standards",
    "work_style": "Prefer minimal meetings, deep focus time, written communication over verbal, structured environments with clear goals",
    "interview_approach": "INTJs should prepare structured, logical answers. Leverage strategic thinking in system design. Be aware of coming across as too blunt — practice showing empathy in behavioral answers.",
},

"mbti_enfp": {
    "type": "ENFP - The Campaigner",
    "career_fit": "ENFPs thrive in creative, people-oriented roles with variety and meaning. Top career matches: Product Manager, UX Designer, Developer Advocate, Technical Writer, Startup Founder, Solutions Engineer. ENFPs need roles with human connection and creative problem-solving.",
    "strengths": "Creativity, enthusiasm, empathy, adaptability, communication skills",
    "work_style": "Prefer collaborative environments, brainstorming sessions, flexible schedules, meaningful work",
    "interview_approach": "ENFPs should channel their enthusiasm while staying focused. Use storytelling in behavioral answers. Show how creativity leads to practical solutions.",
},

"mbti_istp": {
    "type": "ISTP - The Virtuoso",
    "career_fit": "ISTPs excel in hands-on, practical roles requiring troubleshooting and technical expertise. Top career matches: DevOps Engineer, SRE, Security Engineer, Systems Administrator, Embedded Systems Developer, Network Engineer. ISTPs prefer working with tangible systems and solving concrete problems.",
    "strengths": "Troubleshooting, practical skills, adaptability, crisis management, technical depth",
    "work_style": "Prefer hands-on work, minimal bureaucracy, flexible problem-solving, learning by doing",
},

"mbti_enfj": {
    "type": "ENFJ - The Protagonist",
    "career_fit": "ENFJs are natural leaders who excel in roles requiring team leadership, mentoring, and stakeholder management. Top career matches: Engineering Manager, Technical Program Manager, Developer Relations, Scrum Master, CTO, VP of Engineering. ENFJs build strong teams and drive alignment.",
    "strengths": "Leadership, communication, empathy, vision, team building",
    "work_style": "Prefer collaborative environments, mentoring opportunities, leadership roles, making organizational impact",
},

"mbti_intp": {
    "type": "INTP - The Logician",
    "career_fit": "INTPs thrive in roles requiring deep analytical thinking, research, and theoretical problem-solving. Top career matches: Research Scientist, Compiler Engineer, Algorithm Developer, Data Scientist, AI Researcher, Security Researcher. INTPs love complex theoretical challenges.",
    "strengths": "Analytical depth, objectivity, innovation, pattern recognition, theoretical thinking",
    "work_style": "Prefer autonomous work, intellectual freedom, minimal bureaucracy, deep technical challenges",
},

"mbti_estj": {
    "type": "ESTJ - The Executive",
    "career_fit": "ESTJs excel in organized, structured leadership roles. Top career matches: Project Manager, QA Manager, IT Director, Technical Program Manager, Operations Manager. ESTJs bring order, efficiency, and clear processes to teams.",
    "strengths": "Organization, reliability, leadership, decisiveness, process management",
    "work_style": "Prefer clear hierarchies, defined processes, measurable goals, team leadership",
},

"mbti_infp": {
    "type": "INFP - The Mediator",
    "career_fit": "INFPs seek meaningful, value-aligned work. Top career matches: UX Researcher, Technical Writer, Accessibility Engineer, EdTech Developer, Non-profit Tech Lead. INFPs bring empathy and creativity to human-centered design.",
    "strengths": "Creativity, empathy, writing, values-driven work, user advocacy",
    "work_style": "Prefer meaningful projects, creative freedom, small teams, non-competitive environments",
},

"mbti_entj": {
    "type": "ENTJ - The Commander",
    "career_fit": "ENTJs are driven leaders who excel at scaling organizations and making tough decisions. Top career matches: CTO, VP Engineering, Startup Founder, Solutions Architect, Management Consultant. ENTJs drive strategy and execution.",
    "strengths": "Strategic leadership, decision-making, efficiency, confidence, long-term vision",
    "work_style": "Prefer leadership positions, high-impact projects, fast-paced environments, strategic thinking",
},

}
