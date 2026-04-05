"""
MCP (Model Context Protocol) Server for LumosCareer.

Implements the MCP specification pattern with three core primitives:
- Tools: Executable actions AI agents can invoke
- Resources: Read-only data sources for AI context
- Prompts: Reusable, parameterized prompt templates

This server acts as the central orchestration layer between AI agents
and the application's algorithms, data, and knowledge base.
"""

import json
from .mcp_tools import MCPTools
from .rag_engine import get_rag_engine
from . import mcp_prompts


class MCPServer:
    """
    Model Context Protocol Server.
    Provides standardized discovery and invocation of Tools, Resources, and Prompts.
    Uses JSON-RPC-style method dispatch.
    """

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        self.tools = MCPTools()
        self.rag = get_rag_engine()
        self._resource_handlers = {
            "company_profiles": self._get_company_profiles,
            "skill_graph": self._get_skill_graph,
            "career_knowledge": self._get_career_knowledge,
            "user_profile": self._get_user_profile,
            "market_trends": self._get_market_trends,
            "interview_patterns": self._get_interview_patterns,
        }
        self._prompt_handlers = {
            "interviewer_prompt": self._build_interviewer_prompt,
            "therapist_prompt": self._build_therapist_prompt,
            "career_advisor_prompt": self._build_career_advisor_prompt,
            "report_generator_prompt": self._build_report_generator_prompt,
            "interview_prep_prompt": self._build_interview_prep_prompt,
        }

    # ===================== DISCOVERY =====================

    def list_tools(self):
        """Returns metadata for all available MCP tools."""
        return list(MCPTools.TOOL_DEFINITIONS.values())

    def list_resources(self):
        """Returns metadata for all available MCP resources."""
        return [
            {"uri": "company_profiles", "name": "Company Profiles", "description": "All company data with tech stacks, culture, and interview notes", "mimeType": "application/json"},
            {"uri": "skill_graph", "name": "Skill Graph", "description": "Full skill node/edge/signal knowledge graph", "mimeType": "application/json"},
            {"uri": "career_knowledge", "name": "Career Knowledge", "description": "RAG knowledge base documents covering careers, tech, and wellness", "mimeType": "application/json"},
            {"uri": "user_profile", "name": "User Profile", "description": "Current user's resume, bio, targets, and personality type", "mimeType": "application/json"},
            {"uri": "market_trends", "name": "Market Trends", "description": "Aggregated demand trends across skills and industries", "mimeType": "application/json"},
            {"uri": "interview_patterns", "name": "Interview Patterns", "description": "Company-specific interview patterns and preparation guides", "mimeType": "application/json"},
        ]

    def list_prompts(self):
        """Returns metadata for all available MCP prompt templates."""
        return [
            {"name": "interviewer_prompt", "description": "Dynamic interview simulation prompt with company-specific context", "arguments": ["company_name", "job_description", "resume_text", "personality_type"]},
            {"name": "therapist_prompt", "description": "Therapeutic companion prompt with CBT and wellness knowledge", "arguments": []},
            {"name": "career_advisor_prompt", "description": "Career advisory prompt with personality-career mapping", "arguments": ["personality_type", "resume_text", "career_answers"]},
            {"name": "report_generator_prompt", "description": "Company compatibility report generation prompt", "arguments": ["company_name", "resume_text", "personality_type", "similarity_score", "job_description"]},
            {"name": "interview_prep_prompt", "description": "Interview preparation guide generation prompt", "arguments": ["company_name", "resume_text"]},
        ]

    # ===================== EXECUTION =====================

    def call_tool(self, name, args=None):
        """Execute an MCP tool by name with given arguments."""
        if args is None:
            args = {}
        tool_method = getattr(self.tools, name, None)
        if not tool_method:
            return {"error": f"Tool '{name}' not found"}
        try:
            return tool_method(**args)
        except Exception as e:
            return {"error": f"Tool execution failed: {str(e)}"}

    def read_resource(self, uri, context=None):
        """Read an MCP resource by URI."""
        handler = self._resource_handlers.get(uri)
        if not handler:
            return {"error": f"Resource '{uri}' not found"}
        try:
            return handler(context or {})
        except Exception as e:
            return {"error": f"Resource read failed: {str(e)}"}

    def get_prompt(self, name, args=None):
        """Get a rendered MCP prompt template by name."""
        if args is None:
            args = {}
        handler = self._prompt_handlers.get(name)
        if not handler:
            return {"error": f"Prompt '{name}' not found"}
        try:
            return handler(**args)
        except Exception as e:
            return {"error": f"Prompt generation failed: {str(e)}"}

    def handle_request(self, method, params=None):
        """
        JSON-RPC-style request dispatcher.
        Supports: tools/list, tools/call, resources/list, resources/read, prompts/list, prompts/get
        """
        if params is None:
            params = {}

        dispatch = {
            "tools/list": lambda p: self.list_tools(),
            "tools/call": lambda p: self.call_tool(p.get("name", ""), p.get("arguments", {})),
            "resources/list": lambda p: self.list_resources(),
            "resources/read": lambda p: self.read_resource(p.get("uri", ""), p.get("context")),
            "prompts/list": lambda p: self.list_prompts(),
            "prompts/get": lambda p: self.get_prompt(p.get("name", ""), p.get("arguments", {})),
        }

        handler = dispatch.get(method)
        if not handler:
            return {"error": f"Unknown method: {method}"}
        return handler(params)

    # ===================== RESOURCE HANDLERS =====================

    def _get_company_profiles(self, context):
        from .models import Company
        companies = Company.objects.all()
        return [{
            "name": c.name,
            "industry": c.industry,
            "tech_stack": c.tech_stack,
            "interview_notes": c.interview_notes,
        } for c in companies]

    def _get_skill_graph(self, context):
        from .models import SkillNode, SkillEdge
        nodes = [{"name": n.name, "category": n.category, "importance": n.importance_score, "difficulty": n.difficulty_level} for n in SkillNode.objects.all()]
        edges = [{"source": e.source.name, "target": e.target.name, "time": e.weight_time, "difficulty": e.weight_difficulty} for e in SkillEdge.objects.select_related('source', 'target').all()]
        return {"nodes": nodes, "edges": edges}

    def _get_career_knowledge(self, context):
        query = context.get("query", "career technology skills")
        results = self.rag.retrieve(query, top_k=10)
        return [{"key": k, "text": t[:500], "score": round(s, 4)} for k, t, s in results]

    def _get_user_profile(self, context):
        user = context.get("user")
        if not user:
            return {"error": "No user context provided"}
        try:
            profile = user.profile
            personality = ""
            try:
                from .models import AssessmentResult
                assessment = AssessmentResult.objects.get(user=user)
                personality = assessment.result_type
            except Exception:
                pass
            return {
                "username": user.username,
                "resume_text": profile.resume_text[:3000] if profile.resume_text else "",
                "bio": profile.bio,
                "target_roles": profile.roles_list,
                "personality_type": personality,
            }
        except Exception:
            return {"username": user.username if user else "anonymous"}

    def _get_market_trends(self, context):
        from .models import SkillSignal
        signals = SkillSignal.objects.select_related('skill').all()
        trends = {"Rising": [], "Stable": [], "Falling": []}
        for sig in signals:
            trends[sig.demand_trend].append({
                "skill": sig.skill.name,
                "success_rate": sig.success_rate,
                "lift": sig.lift,
            })
        return trends

    def _get_interview_patterns(self, context):
        company_name = context.get("company_name", "")
        query = f"interview preparation {company_name} technical behavioral"
        results = self.rag.retrieve(query, top_k=5)
        return [{"key": k, "content": t[:500]} for k, t, s in results]

    # ===================== PROMPT HANDLERS =====================

    def _build_interviewer_prompt(self, company_name="", job_description="", resume_text="", personality_type=""):
        query = f"interview {company_name} technical behavioral preparation tips culture"
        rag_context = self.rag.build_context(query, max_chars=2000)
        return mcp_prompts.get_interviewer_prompt(company_name, job_description, resume_text, personality_type, rag_context)

    def _build_therapist_prompt(self):
        query = "career stress burnout imposter syndrome anxiety CBT techniques mindfulness wellness coping strategies"
        rag_context = self.rag.build_context(query, max_chars=2000)
        return mcp_prompts.get_therapist_prompt(rag_context)

    def _build_career_advisor_prompt(self, personality_type="", resume_text="", career_answers=""):
        query = f"career path {personality_type} technology skills roadmap industry trends salary growth"
        rag_context = self.rag.build_context(query, max_chars=2000)
        return mcp_prompts.get_career_advisor_prompt(personality_type, resume_text, career_answers, rag_context)

    def _build_report_generator_prompt(self, company_name="", resume_text="", personality_type="", similarity_score=0, job_description=""):
        query = f"company {company_name} culture interview tech stack career compatibility"
        rag_context = self.rag.build_context(query, max_chars=2000)
        return mcp_prompts.get_report_generator_prompt(company_name, resume_text, personality_type, similarity_score, job_description, rag_context)

    def _build_interview_prep_prompt(self, company_name="", resume_text=""):
        query = f"interview preparation {company_name} questions system design behavioral coding"
        rag_context = self.rag.build_context(query, max_chars=2000)
        return mcp_prompts.get_interview_prep_prompt(company_name, resume_text, rag_context)


def get_mcp_server():
    """Factory function to get/create the singleton MCP server."""
    return MCPServer()
