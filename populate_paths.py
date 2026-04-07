import os
import django

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "lumos_career.settings")
django.setup()

from core.models import CareerPath

def populate_paths():
    print("Clearing old predefined career paths...")
    CareerPath.objects.filter(is_predefined=True).delete()

    paths_data = [
        {
            "title": "Frontend Developer",
            "description": "Master the art of building beautiful, interactive, and performant user interfaces for the modern web.",
            "roadmap_data": {
                "steps": [
                    {"step_id": 1, "name": "HTML, CSS & Web Fundamentals", "description": "Learn semantic HTML5, CSS3 (Flexbox, Grid, animations), responsive design, accessibility (WCAG), and browser DevTools. Build 3 responsive landing pages.", "status": "open", "duration": "4 Weeks", "skills_covered": ["HTML5", "CSS3", "Responsive Design", "Accessibility"], "resources": ["MDN Web Docs", "freeCodeCamp", "CSS-Tricks"]},
                    {"step_id": 2, "name": "JavaScript Essentials", "description": "Master core JavaScript: ES6+ features (arrow functions, destructuring, modules), DOM manipulation, event handling, async/await, Promises, Fetch API, and error handling.", "status": "locked", "duration": "6 Weeks", "skills_covered": ["JavaScript", "ES6+", "Async Programming", "DOM"], "resources": ["JavaScript.info", "Eloquent JavaScript", "JavaScript30"]},
                    {"step_id": 3, "name": "TypeScript & Modern Tooling", "description": "Learn TypeScript (types, interfaces, generics, type guards), npm/yarn, Vite, ESLint, Prettier, and modern build tooling. Migrate a JS project to TypeScript.", "status": "locked", "duration": "3 Weeks", "skills_covered": ["TypeScript", "Vite", "npm", "ESLint"], "resources": ["TypeScript Handbook", "Vite Documentation"]},
                    {"step_id": 4, "name": "React.js Ecosystem", "description": "Master React: components, hooks (useState, useEffect, useContext, useReducer, custom hooks), React Router, state management (Zustand/Redux), and React DevTools.", "status": "locked", "duration": "8 Weeks", "skills_covered": ["React", "Hooks", "State Management", "React Router"], "resources": ["React Official Docs", "Epic React by Kent C. Dodds", "React Patterns"]},
                    {"step_id": 5, "name": "Next.js & Server-Side Rendering", "description": "Learn Next.js: App Router, Server Components, SSR/SSG/ISR, API routes, middleware, image optimization, and deployment on Vercel.", "status": "locked", "duration": "4 Weeks", "skills_covered": ["Next.js", "SSR", "SSG", "Vercel"], "resources": ["Next.js Documentation", "Lee Robinson's Blog"]},
                    {"step_id": 6, "name": "Testing & Quality Assurance", "description": "Master frontend testing: unit tests with Jest/Vitest, component tests with React Testing Library, E2E tests with Playwright/Cypress, visual regression testing.", "status": "locked", "duration": "3 Weeks", "skills_covered": ["Jest", "Testing Library", "Playwright", "E2E Testing"], "resources": ["Testing JavaScript by Kent C. Dodds"]},
                    {"step_id": 7, "name": "Performance & Deployment", "description": "Optimize Core Web Vitals (LCP, FID, CLS), implement code splitting, lazy loading, caching strategies, CDN deployment, and CI/CD pipelines for frontend.", "status": "locked", "duration": "3 Weeks", "skills_covered": ["Performance", "Core Web Vitals", "CI/CD", "CDN"], "resources": ["web.dev", "Lighthouse", "Vercel Docs"]}
                ]
            }
        },
        {
            "title": "Backend Developer",
            "description": "Build robust, scalable server-side applications, APIs, and data-driven systems.",
            "roadmap_data": {
                "steps": [
                    {"step_id": 1, "name": "Python & Programming Fundamentals", "description": "Master Python: data types, control flow, functions, OOP, error handling, file I/O, virtual environments, and pip. Build CLI tools and automation scripts.", "status": "open", "duration": "5 Weeks", "skills_covered": ["Python", "OOP", "CLI Tools"], "resources": ["Automate the Boring Stuff", "Python Official Tutorial"]},
                    {"step_id": 2, "name": "SQL & Database Design", "description": "Master SQL: CRUD, JOINs, subqueries, window functions, indexing, normalization, ER diagrams. Work with PostgreSQL. Design schemas for real-world applications.", "status": "locked", "duration": "4 Weeks", "skills_covered": ["SQL", "PostgreSQL", "Database Design", "Normalization"], "resources": ["SQLBolt", "PostgreSQL Documentation", "Use The Index, Luke"]},
                    {"step_id": 3, "name": "Django Web Framework", "description": "Build full-stack web apps with Django: models, views, templates, forms, authentication, admin, ORM queries, middleware, signals, and management commands.", "status": "locked", "duration": "6 Weeks", "skills_covered": ["Django", "ORM", "Authentication", "REST"], "resources": ["Django Documentation", "Django for Beginners by William Vincent"]},
                    {"step_id": 4, "name": "REST API Design & FastAPI", "description": "Design RESTful APIs following best practices: versioning, pagination, filtering, error handling, HATEOAS. Build high-performance APIs with FastAPI and Pydantic.", "status": "locked", "duration": "4 Weeks", "skills_covered": ["REST APIs", "FastAPI", "Pydantic", "API Design"], "resources": ["FastAPI Documentation", "RESTful Web APIs by Leonard Richardson"]},
                    {"step_id": 5, "name": "Authentication, Security & Caching", "description": "Implement JWT, OAuth2, session-based auth. Learn security: OWASP Top 10, SQL injection, XSS, CSRF prevention. Add Redis caching for performance.", "status": "locked", "duration": "4 Weeks", "skills_covered": ["JWT", "OAuth2", "Redis", "Security"], "resources": ["OWASP Documentation", "Redis University"]},
                    {"step_id": 6, "name": "Docker & Deployment", "description": "Containerize applications with Docker and Docker Compose. Deploy to AWS (EC2, RDS, S3) or Railway/Render. Set up CI/CD with GitHub Actions.", "status": "locked", "duration": "3 Weeks", "skills_covered": ["Docker", "AWS", "CI/CD", "GitHub Actions"], "resources": ["Docker Documentation", "AWS Free Tier"]}
                ]
            }
        },
        {
            "title": "Full-Stack Developer",
            "description": "Become a versatile engineer who can build complete web applications end-to-end.",
            "roadmap_data": {
                "steps": [
                    {"step_id": 1, "name": "Web Fundamentals & JavaScript", "description": "HTML5, CSS3, JavaScript ES6+, TypeScript basics. Build responsive, accessible web pages.", "status": "open", "duration": "6 Weeks", "skills_covered": ["HTML/CSS", "JavaScript", "TypeScript"], "resources": ["freeCodeCamp", "JavaScript.info"]},
                    {"step_id": 2, "name": "React & Frontend Architecture", "description": "React components, hooks, state management, React Router, Next.js basics. Build 2 complete frontend applications.", "status": "locked", "duration": "7 Weeks", "skills_covered": ["React", "Next.js", "State Management"], "resources": ["React Docs", "Next.js Docs"]},
                    {"step_id": 3, "name": "Node.js & Express Backend", "description": "Server-side JavaScript with Node.js, Express.js REST APIs, middleware, error handling, and authentication.", "status": "locked", "duration": "5 Weeks", "skills_covered": ["Node.js", "Express.js", "REST APIs"], "resources": ["Node.js Documentation", "Express.js Guide"]},
                    {"step_id": 4, "name": "Databases & Data Modeling", "description": "PostgreSQL for relational data, MongoDB for document data. ORMs (Prisma/Sequelize). Database design patterns.", "status": "locked", "duration": "4 Weeks", "skills_covered": ["PostgreSQL", "MongoDB", "Prisma", "Data Modeling"], "resources": ["Prisma Documentation", "MongoDB University"]},
                    {"step_id": 5, "name": "Authentication & Real-time Features", "description": "JWT/OAuth2 authentication flows, WebSockets for real-time features, third-party API integrations.", "status": "locked", "duration": "3 Weeks", "skills_covered": ["JWT", "OAuth2", "WebSockets"], "resources": ["Auth0 Documentation"]},
                    {"step_id": 6, "name": "DevOps & Deployment", "description": "Docker, CI/CD with GitHub Actions, deployment to Vercel/Railway/AWS, monitoring basics.", "status": "locked", "duration": "3 Weeks", "skills_covered": ["Docker", "CI/CD", "Deployment"], "resources": ["Docker Docs", "GitHub Actions Docs"]},
                    {"step_id": 7, "name": "Capstone: Full-Stack Application", "description": "Build and deploy a complete full-stack application with auth, real-time features, and CI/CD. Document and present.", "status": "locked", "duration": "4 Weeks", "skills_covered": ["System Design", "Project Architecture"], "resources": ["Portfolio Best Practices"]}
                ]
            }
        },
        {
            "title": "Data Scientist",
            "description": "Extract actionable insights from data using statistics, programming, and machine learning.",
            "roadmap_data": {
                "steps": [
                    {"step_id": 1, "name": "Python & Data Handling", "description": "Master Python, Pandas, NumPy for data manipulation. Jupyter notebooks, data cleaning, feature engineering basics.", "status": "open", "duration": "6 Weeks", "skills_covered": ["Python", "Pandas", "NumPy", "Jupyter"], "resources": ["Python for Data Analysis by Wes McKinney"]},
                    {"step_id": 2, "name": "Statistics & Probability", "description": "Descriptive statistics, probability distributions, hypothesis testing, confidence intervals, Bayesian thinking, A/B testing.", "status": "locked", "duration": "6 Weeks", "skills_covered": ["Statistics", "Probability", "A/B Testing", "Bayesian"], "resources": ["Think Stats", "Khan Academy Statistics"]},
                    {"step_id": 3, "name": "Data Visualization & EDA", "description": "Master Matplotlib, Seaborn, Plotly. Exploratory Data Analysis techniques, storytelling with data, dashboard creation.", "status": "locked", "duration": "3 Weeks", "skills_covered": ["Matplotlib", "Seaborn", "Plotly", "EDA"], "resources": ["Storytelling with Data by Cole Nussbaumer"]},
                    {"step_id": 4, "name": "SQL & Data Warehousing", "description": "Advanced SQL: window functions, CTEs, optimization. Data warehouse concepts, star/snowflake schemas, ETL basics.", "status": "locked", "duration": "4 Weeks", "skills_covered": ["SQL", "Data Warehousing", "ETL"], "resources": ["Mode Analytics SQL Tutorial"]},
                    {"step_id": 5, "name": "Machine Learning Foundations", "description": "Scikit-Learn: regression, classification, clustering, ensemble methods. Model evaluation, cross-validation, hyperparameter tuning.", "status": "locked", "duration": "8 Weeks", "skills_covered": ["Scikit-Learn", "ML Algorithms", "Model Evaluation"], "resources": ["Hands-On ML by Aurélien Géron", "Kaggle Courses"]},
                    {"step_id": 6, "name": "Deep Learning & Neural Networks", "description": "PyTorch/TensorFlow basics: CNNs, RNNs, transformers. NLP fundamentals, computer vision basics, transfer learning.", "status": "locked", "duration": "8 Weeks", "skills_covered": ["PyTorch", "Deep Learning", "NLP", "CNNs"], "resources": ["Fast.ai", "Deep Learning Specialization by Andrew Ng"]},
                    {"step_id": 7, "name": "ML Deployment & MLOps", "description": "Model serving with FastAPI/Flask, containerization, model monitoring, experiment tracking (MLflow), basic MLOps pipeline.", "status": "locked", "duration": "4 Weeks", "skills_covered": ["MLflow", "Model Serving", "Docker", "MLOps"], "resources": ["MLflow Documentation", "Made With ML"]}
                ]
            }
        },
        {
            "title": "Machine Learning Engineer",
            "description": "Bridge the gap between ML research and production systems at scale.",
            "roadmap_data": {
                "steps": [
                    {"step_id": 1, "name": "Python & Software Engineering", "description": "Advanced Python: design patterns, testing, packaging, clean code. Software engineering fundamentals for ML systems.", "status": "open", "duration": "4 Weeks", "skills_covered": ["Python", "Software Engineering", "Testing"], "resources": ["Clean Code", "Effective Python"]},
                    {"step_id": 2, "name": "Mathematics for ML", "description": "Linear algebra, calculus, optimization, probability theory. Matrix operations, gradient descent, convex optimization.", "status": "locked", "duration": "6 Weeks", "skills_covered": ["Linear Algebra", "Calculus", "Optimization"], "resources": ["Mathematics for Machine Learning (book)", "3Blue1Brown"]},
                    {"step_id": 3, "name": "Classical ML & Feature Engineering", "description": "Master all classical ML algorithms, feature engineering, feature stores, data pipelines, model selection strategies.", "status": "locked", "duration": "6 Weeks", "skills_covered": ["ML Algorithms", "Feature Engineering", "Scikit-Learn"], "resources": ["Kaggle Competitions", "Feature Engineering for ML"]},
                    {"step_id": 4, "name": "Deep Learning & PyTorch", "description": "PyTorch mastery: custom datasets, training loops, distributed training, model architectures (CNNs, transformers, GANs).", "status": "locked", "duration": "8 Weeks", "skills_covered": ["PyTorch", "Deep Learning", "Transformers", "Distributed Training"], "resources": ["PyTorch Documentation", "d2l.ai"]},
                    {"step_id": 5, "name": "LLMs, RAG & Prompt Engineering", "description": "Work with LLMs: API integration, RAG pipelines, vector databases, embeddings, fine-tuning, prompt engineering, agent frameworks.", "status": "locked", "duration": "6 Weeks", "skills_covered": ["LLMs", "RAG", "Embeddings", "Prompt Engineering"], "resources": ["OpenAI Cookbook", "LangChain Documentation"]},
                    {"step_id": 6, "name": "MLOps & Production ML", "description": "ML pipelines, experiment tracking (MLflow/W&B), model serving (TorchServe/Triton), monitoring, A/B testing, Kubernetes for ML.", "status": "locked", "duration": "6 Weeks", "skills_covered": ["MLOps", "MLflow", "Kubernetes", "Model Serving"], "resources": ["Designing ML Systems by Chip Huyen", "Made With ML"]}
                ]
            }
        },
        {
            "title": "DevOps Engineer",
            "description": "Build and maintain CI/CD pipelines, infrastructure automation, and reliable deployment systems.",
            "roadmap_data": {
                "steps": [
                    {"step_id": 1, "name": "Linux & Shell Scripting", "description": "Master Linux commands, filesystem, processes, networking, permissions. Write Bash scripts for automation. Understand systemd, cron, SSH.", "status": "open", "duration": "4 Weeks", "skills_covered": ["Linux", "Bash", "Shell Scripting"], "resources": ["Linux Command Line by William Shotts", "OverTheWire Wargames"]},
                    {"step_id": 2, "name": "Networking & Security Fundamentals", "description": "TCP/IP, DNS, HTTP/HTTPS, firewalls, VPNs, SSL/TLS, load balancers. Network troubleshooting tools (tcpdump, wireshark, netstat).", "status": "locked", "duration": "4 Weeks", "skills_covered": ["Networking", "Security", "TCP/IP", "DNS"], "resources": ["Computer Networking: A Top-Down Approach"]},
                    {"step_id": 3, "name": "Docker & Container Orchestration", "description": "Docker fundamentals, Dockerfiles, multi-stage builds, Docker Compose. Introduction to Kubernetes: pods, services, deployments, ConfigMaps, Secrets.", "status": "locked", "duration": "6 Weeks", "skills_covered": ["Docker", "Kubernetes", "Containers"], "resources": ["Docker Documentation", "KodeKloud"]},
                    {"step_id": 4, "name": "CI/CD Pipeline Design", "description": "Design and implement CI/CD pipelines with GitHub Actions, Jenkins, or GitLab CI. Automated testing, artifact management, deployment strategies (blue-green, canary).", "status": "locked", "duration": "4 Weeks", "skills_covered": ["CI/CD", "GitHub Actions", "Jenkins"], "resources": ["GitHub Actions Documentation"]},
                    {"step_id": 5, "name": "Infrastructure as Code", "description": "Terraform for cloud infrastructure provisioning, Ansible for configuration management. State management, modules, best practices.", "status": "locked", "duration": "5 Weeks", "skills_covered": ["Terraform", "Ansible", "IaC"], "resources": ["Terraform Documentation", "Terraform Up & Running"]},
                    {"step_id": 6, "name": "Cloud Platforms (AWS/GCP/Azure)", "description": "Master core cloud services: compute (EC2/GCE), storage (S3), databases (RDS), networking (VPC), IAM, and serverless (Lambda).", "status": "locked", "duration": "6 Weeks", "skills_covered": ["AWS", "Cloud Architecture", "IAM"], "resources": ["AWS Certified Solutions Architect Study Guide"]},
                    {"step_id": 7, "name": "Monitoring & Observability", "description": "Prometheus for metrics, Grafana for dashboards, ELK stack for logging. Alerting strategies, SLOs/SLIs, incident response.", "status": "locked", "duration": "4 Weeks", "skills_covered": ["Prometheus", "Grafana", "Observability", "SLOs"], "resources": ["Prometheus Documentation", "Google SRE Book"]}
                ]
            }
        },
        {
            "title": "Cloud Architect",
            "description": "Design and oversee enterprise cloud computing strategy, architecture, and governance.",
            "roadmap_data": {
                "steps": [
                    {"step_id": 1, "name": "Cloud Fundamentals & AWS Core", "description": "Cloud computing models (IaaS/PaaS/SaaS), AWS core services: EC2, S3, VPC, IAM, RDS, Lambda. AWS Well-Architected Framework.", "status": "open", "duration": "6 Weeks", "skills_covered": ["AWS", "Cloud Computing", "IAM"], "resources": ["AWS Cloud Practitioner", "A Cloud Guru"]},
                    {"step_id": 2, "name": "Networking & Security Architecture", "description": "VPC design, subnets, NACLs, security groups, WAF, CloudFront, Route 53. Zero-trust architecture, encryption at rest and in transit.", "status": "locked", "duration": "5 Weeks", "skills_covered": ["Networking", "Security", "VPC"], "resources": ["AWS Security Specialty"]},
                    {"step_id": 3, "name": "High Availability & Disaster Recovery", "description": "Multi-AZ and multi-region architectures, auto-scaling, load balancing (ALB/NLB), backup strategies, RPO/RTO planning.", "status": "locked", "duration": "5 Weeks", "skills_covered": ["High Availability", "Auto-scaling", "DR"], "resources": ["AWS Solutions Architect Professional"]},
                    {"step_id": 4, "name": "Infrastructure as Code & DevOps", "description": "Terraform at scale, CloudFormation, CDK. GitOps workflows, CI/CD for infrastructure, policy as code (OPA, Sentinel).", "status": "locked", "duration": "5 Weeks", "skills_covered": ["Terraform", "CloudFormation", "GitOps"], "resources": ["Terraform Documentation"]},
                    {"step_id": 5, "name": "Microservices & Serverless Architecture", "description": "Design patterns for microservices, API Gateway, service mesh (Istio), serverless architectures (Lambda, Step Functions, EventBridge).", "status": "locked", "duration": "5 Weeks", "skills_covered": ["Microservices", "Serverless", "API Gateway"], "resources": ["Building Microservices by Sam Newman"]},
                    {"step_id": 6, "name": "Cost Optimization & Governance", "description": "FinOps principles, cost allocation, reserved instances, savings plans, AWS Organizations, landing zones, compliance frameworks.", "status": "locked", "duration": "4 Weeks", "skills_covered": ["FinOps", "Cost Optimization", "Governance"], "resources": ["FinOps Foundation", "AWS Cost Management"]}
                ]
            }
        },
        {
            "title": "Mobile Developer",
            "description": "Build native and cross-platform mobile applications for iOS and Android.",
            "roadmap_data": {
                "steps": [
                    {"step_id": 1, "name": "Programming Fundamentals", "description": "Master Dart (for Flutter) or JavaScript/TypeScript (for React Native). OOP, async patterns, state management concepts.", "status": "open", "duration": "4 Weeks", "skills_covered": ["Dart", "TypeScript", "OOP"], "resources": ["Dart Documentation", "TypeScript Handbook"]},
                    {"step_id": 2, "name": "Flutter / React Native Framework", "description": "Choose your framework and master it: widget trees, navigation, state management (Provider/Riverpod or Redux), platform channels.", "status": "locked", "duration": "8 Weeks", "skills_covered": ["Flutter", "React Native", "State Management"], "resources": ["Flutter Documentation", "React Native Documentation"]},
                    {"step_id": 3, "name": "UI/UX & Platform Design Guidelines", "description": "Material Design (Android), Human Interface Guidelines (iOS), responsive layouts, animations, gestures, accessibility.", "status": "locked", "duration": "3 Weeks", "skills_covered": ["Material Design", "HIG", "Animations"], "resources": ["Material Design Guidelines", "Apple HIG"]},
                    {"step_id": 4, "name": "Backend Integration & APIs", "description": "REST/GraphQL API consumption, authentication (OAuth, Firebase Auth), real-time data (WebSockets, Firebase Realtime DB).", "status": "locked", "duration": "4 Weeks", "skills_covered": ["REST APIs", "Firebase", "GraphQL"], "resources": ["Firebase Documentation"]},
                    {"step_id": 5, "name": "Local Storage & Offline Support", "description": "SQLite, Hive/SharedPreferences, offline-first architecture, data synchronization, caching strategies.", "status": "locked", "duration": "3 Weeks", "skills_covered": ["SQLite", "Offline-first", "Caching"], "resources": ["SQLite Documentation"]},
                    {"step_id": 6, "name": "Testing & App Store Deployment", "description": "Unit, widget, and integration testing. App store submission (Google Play, Apple App Store), CI/CD for mobile (Fastlane, Codemagic).", "status": "locked", "duration": "3 Weeks", "skills_covered": ["Testing", "App Store", "CI/CD"], "resources": ["Fastlane Documentation", "Codemagic"]}
                ]
            }
        },
        {
            "title": "Cybersecurity Analyst",
            "description": "Protect organizations from cyber threats through monitoring, analysis, and security implementation.",
            "roadmap_data": {
                "steps": [
                    {"step_id": 1, "name": "Networking & OS Fundamentals", "description": "TCP/IP deep dive, OSI model, DNS, DHCP, firewalls, VPNs. Linux and Windows administration, command-line tools.", "status": "open", "duration": "5 Weeks", "skills_covered": ["Networking", "Linux", "Windows", "TCP/IP"], "resources": ["CompTIA Network+", "TryHackMe"]},
                    {"step_id": 2, "name": "Security Fundamentals & CIA Triad", "description": "Confidentiality, Integrity, Availability. Cryptography basics, PKI, authentication methods, access control models (RBAC, ABAC).", "status": "locked", "duration": "4 Weeks", "skills_covered": ["Cryptography", "PKI", "Access Control"], "resources": ["CompTIA Security+", "Cybrary"]},
                    {"step_id": 3, "name": "Threat Intelligence & SIEM", "description": "Threat landscape, MITRE ATT&CK framework, SIEM tools (Splunk, ELK), log analysis, indicator of compromise (IoC) detection.", "status": "locked", "duration": "5 Weeks", "skills_covered": ["SIEM", "Threat Intelligence", "MITRE ATT&CK"], "resources": ["Splunk Free Training", "MITRE Documentation"]},
                    {"step_id": 4, "name": "Vulnerability Assessment & Pen Testing", "description": "Vulnerability scanning (Nessus, OpenVAS), penetration testing methodology, OWASP Top 10, web application security, Burp Suite.", "status": "locked", "duration": "6 Weeks", "skills_covered": ["Pen Testing", "OWASP", "Burp Suite"], "resources": ["PortSwigger Web Security Academy", "Hack The Box"]},
                    {"step_id": 5, "name": "Incident Response & Digital Forensics", "description": "Incident response lifecycle, evidence collection, chain of custody, memory forensics, malware analysis basics, post-incident reporting.", "status": "locked", "duration": "5 Weeks", "skills_covered": ["Incident Response", "Forensics", "Malware Analysis"], "resources": ["SANS Incident Handler's Handbook"]},
                    {"step_id": 6, "name": "Compliance & Governance", "description": "SOC 2, ISO 27001, GDPR, HIPAA, PCI-DSS. Risk assessment frameworks, security auditing, policy development.", "status": "locked", "duration": "4 Weeks", "skills_covered": ["SOC 2", "ISO 27001", "GDPR", "Risk Assessment"], "resources": ["NIST Cybersecurity Framework"]}
                ]
            }
        },
        {
            "title": "Product Manager (Technical)",
            "description": "Define product vision, strategy, and roadmap at the intersection of business, design, and engineering.",
            "roadmap_data": {
                "steps": [
                    {"step_id": 1, "name": "Product Management Fundamentals", "description": "Product lifecycle, product-market fit, user personas, jobs-to-be-done framework, competitive analysis, product vision and strategy.", "status": "open", "duration": "4 Weeks", "skills_covered": ["Product Strategy", "User Personas", "Market Analysis"], "resources": ["Inspired by Marty Cagan", "Product School"]},
                    {"step_id": 2, "name": "User Research & Discovery", "description": "User interviews, surveys, usability testing, A/B testing, analytics tools (Mixpanel, Amplitude), customer journey mapping.", "status": "locked", "duration": "4 Weeks", "skills_covered": ["User Research", "A/B Testing", "Analytics"], "resources": ["The Mom Test", "Mixpanel Documentation"]},
                    {"step_id": 3, "name": "Technical Fluency", "description": "Understand APIs, databases, system architecture, CI/CD, cloud services. Read code, participate in sprint planning, write technical specs.", "status": "locked", "duration": "5 Weeks", "skills_covered": ["APIs", "System Architecture", "SQL", "Technical Communication"], "resources": ["System Design Interview by Alex Xu"]},
                    {"step_id": 4, "name": "Data-Driven Decision Making", "description": "SQL for product analytics, defining metrics (North Star, KPIs, OKRs), cohort analysis, funnel analysis, experimentation frameworks.", "status": "locked", "duration": "4 Weeks", "skills_covered": ["SQL", "Product Metrics", "OKRs", "Experimentation"], "resources": ["Lean Analytics", "Mode SQL Tutorial"]},
                    {"step_id": 5, "name": "Agile & Stakeholder Management", "description": "Scrum and Kanban, sprint planning, backlog grooming, stakeholder communication, cross-functional team leadership, roadmap prioritization (RICE, MoSCoW).", "status": "locked", "duration": "3 Weeks", "skills_covered": ["Agile/Scrum", "Prioritization", "Stakeholder Management"], "resources": ["Scrum Guide", "SVPG Blog"]},
                    {"step_id": 6, "name": "Product Execution & Launch", "description": "PRD writing, go-to-market strategy, feature flags, phased rollouts, post-launch metrics tracking, iteration based on data.", "status": "locked", "duration": "4 Weeks", "skills_covered": ["PRDs", "GTM Strategy", "Feature Flags"], "resources": ["Lenny's Newsletter", "Product Hunt"]}
                ]
            }
        },
        {
            "title": "AI/LLM Engineer",
            "description": "Build production applications powered by large language models, RAG pipelines, and AI agents.",
            "roadmap_data": {
                "steps": [
                    {"step_id": 1, "name": "Python & ML Foundations", "description": "Advanced Python, NumPy, basic ML concepts (supervised/unsupervised learning), evaluation metrics, data preprocessing.", "status": "open", "duration": "5 Weeks", "skills_covered": ["Python", "NumPy", "ML Basics"], "resources": ["Fast.ai", "Scikit-learn Tutorial"]},
                    {"step_id": 2, "name": "NLP & Transformer Architecture", "description": "Text preprocessing, word embeddings (Word2Vec, GloVe), attention mechanism, transformer architecture, BERT, GPT fundamentals.", "status": "locked", "duration": "6 Weeks", "skills_covered": ["NLP", "Transformers", "Attention", "BERT"], "resources": ["Attention Is All You Need (paper)", "Hugging Face Course"]},
                    {"step_id": 3, "name": "LLM APIs & Prompt Engineering", "description": "OpenAI/Gemini/Claude API integration, prompt engineering techniques (few-shot, CoT, ReAct), structured outputs, token management, cost optimization.", "status": "locked", "duration": "4 Weeks", "skills_covered": ["LLMs", "Prompt Engineering", "API Integration"], "resources": ["OpenAI Cookbook", "Anthropic Prompt Engineering Guide"]},
                    {"step_id": 4, "name": "RAG Pipeline Development", "description": "Embeddings (OpenAI, Sentence Transformers), vector databases (Pinecone, Chroma, pgvector), chunking strategies, retrieval optimization, hybrid search.", "status": "locked", "duration": "5 Weeks", "skills_covered": ["RAG", "Embeddings", "Vector Databases", "Chunking"], "resources": ["LangChain Documentation", "LlamaIndex Documentation"]},
                    {"step_id": 5, "name": "AI Agent Frameworks & MCP", "description": "Build autonomous AI agents: tool use, planning, memory, multi-agent systems. Model Context Protocol (MCP) for standardized AI-tool integration.", "status": "locked", "duration": "5 Weeks", "skills_covered": ["AI Agents", "MCP", "Tool Use", "Multi-Agent"], "resources": ["LangGraph Documentation", "MCP Specification"]},
                    {"step_id": 6, "name": "Fine-tuning & Model Optimization", "description": "LoRA/QLoRA fine-tuning, RLHF concepts, model quantization, distillation, inference optimization (vLLM, TensorRT).", "status": "locked", "duration": "5 Weeks", "skills_covered": ["Fine-tuning", "LoRA", "Quantization", "RLHF"], "resources": ["Hugging Face PEFT", "vLLM Documentation"]},
                    {"step_id": 7, "name": "Production LLM Systems", "description": "LLM observability (LangSmith), guardrails, evaluation frameworks (RAGAS), cost management, A/B testing for prompts, scaling with Kubernetes.", "status": "locked", "duration": "4 Weeks", "skills_covered": ["LLMOps", "Evaluation", "Guardrails", "Observability"], "resources": ["LangSmith Documentation", "RAGAS Documentation"]}
                ]
            }
        },
        {
            "title": "Data Engineer",
            "description": "Design, build, and maintain data infrastructure and pipelines that power analytics and ML.",
            "roadmap_data": {
                "steps": [
                    {"step_id": 1, "name": "Python & SQL Mastery", "description": "Advanced Python (generators, decorators, multiprocessing). Advanced SQL (window functions, CTEs, query optimization, execution plans).", "status": "open", "duration": "5 Weeks", "skills_covered": ["Python", "SQL", "Query Optimization"], "resources": ["High Performance Python", "Use The Index, Luke"]},
                    {"step_id": 2, "name": "Data Modeling & Warehousing", "description": "Dimensional modeling (star/snowflake schemas), data vault, slowly changing dimensions, data catalog, data governance fundamentals.", "status": "locked", "duration": "4 Weeks", "skills_covered": ["Data Modeling", "Data Warehousing", "Dimensional Modeling"], "resources": ["The Data Warehouse Toolkit by Ralph Kimball"]},
                    {"step_id": 3, "name": "ETL/ELT & Orchestration", "description": "Build ETL/ELT pipelines with Apache Airflow, dbt for transformations, data quality frameworks (Great Expectations), scheduling and monitoring.", "status": "locked", "duration": "6 Weeks", "skills_covered": ["Airflow", "dbt", "ETL/ELT", "Data Quality"], "resources": ["Apache Airflow Documentation", "dbt Documentation"]},
                    {"step_id": 4, "name": "Big Data Processing (Spark)", "description": "Apache Spark: RDDs, DataFrames, Spark SQL, PySpark, partitioning, broadcast joins, performance tuning. Delta Lake for ACID transactions.", "status": "locked", "duration": "6 Weeks", "skills_covered": ["Spark", "PySpark", "Delta Lake", "Big Data"], "resources": ["Learning Spark by Jules Damji"]},
                    {"step_id": 5, "name": "Streaming Data (Kafka)", "description": "Apache Kafka: producers, consumers, topics, partitions, consumer groups. Kafka Streams, Kafka Connect, schema registry. Real-time pipeline design.", "status": "locked", "duration": "5 Weeks", "skills_covered": ["Kafka", "Stream Processing", "Real-time Pipelines"], "resources": ["Kafka: The Definitive Guide", "Confluent Training"]},
                    {"step_id": 6, "name": "Cloud Data Platforms & DataOps", "description": "Snowflake/BigQuery/Redshift, cloud-native data architectures, lakehouse pattern, DataOps practices, CI/CD for data pipelines.", "status": "locked", "duration": "5 Weeks", "skills_covered": ["Snowflake", "BigQuery", "Lakehouse", "DataOps"], "resources": ["Snowflake Documentation", "Fundamentals of Data Engineering"]}
                ]
            }
        },
    ]

    print("Populating 12 predefined career paths...")
    for path in paths_data:
        CareerPath.objects.create(
            user=None,
            title=path["title"],
            description=path["description"],
            roadmap_data=path["roadmap_data"],
            progress=0,
            status='Active',
            is_predefined=True
        )

    print(f"Predefined CareerPaths population done! Total: {CareerPath.objects.filter(is_predefined=True).count()}")

if __name__ == '__main__':
    populate_paths()
