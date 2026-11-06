import os
from typing import Any, Dict, List

from crewai import Agent, Crew, Process, Task
from crewai.agents.agent_builder.base_agent import BaseAgent
from crewai import LLM
from crewai.project import CrewBase, agent, crew, task
from dotenv import load_dotenv

from latest_ai_development.secrets_manager import SecretsManager
from latest_ai_development.tracking import get_tracing_backend
from latest_ai_development.tools.keyretained import KeyretainedTool

load_dotenv()

secrets_manager = SecretsManager()


def build_llm(
    provider: str,
    model_name: str,
    temperature: float,
    top_p: float,
    max_tokens: int,
) -> LLM:
    """
    Factory function to build a CrewAI LLM instance.

    IMPORTANT FOR AI / FUTURE EDITS:
    - PROVIDER is a COMMON env var for the whole app
      (OPENAI / ANTHROPICAI / AZUREAI / MICROSOFTFOUNDRYAI / GEMINIAI / BEDROCKAI / OLLAMA).
    - CLOUD_PROVIDER controls where secrets are loaded from (AWS / AZURE).
    - model_name is allowed to differ PER AGENT (read from env per agent).
    - temperature/top_p/max_tokens are HARD-CODED PER AGENT in this file
      (do NOT read these from env, per requirement).
    - For Azure OpenAI: model_name should typically be the AZURE DEPLOYMENT NAME.
    """

    provider = (provider or "OLLAMA").upper()

    if provider == "OPENAI":
        api_key_secret = os.getenv("API_KEY_SECRET", os.getenv("OPENAI_API_KEY_SECRET"))
        api_key = secrets_manager.get_secret(api_key_secret) if api_key_secret else None

        if not api_key:
            raise ValueError(
                "PROVIDER is set to OPENAI but API_KEY_SECRET/OPENAI_API_KEY_SECRET is not set or secret not found."
            )

        if model_name and model_name.startswith("gpt-5"):
            return LLM(
                model=model_name,
                api_key=api_key,
                max_completion_tokens=max_tokens,
            )

        return LLM(
            model=model_name,
            api_key=api_key,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
        )

    if provider == "ANTHROPICAI":
        api_key_secret = os.getenv("API_KEY_SECRET", os.getenv("ANTHROPICAI_API_KEY_SECRET"))
        api_key = secrets_manager.get_secret(api_key_secret) if api_key_secret else None

        if not api_key:
            raise ValueError(
                "PROVIDER is set to ANTHROPICAI but API_KEY_SECRET/ANTHROPICAI_API_KEY_SECRET is not set or secret not found."
            )

        os.environ["ANTHROPIC_API_KEY"] = api_key
        return LLM(
            model=model_name or "anthropic/claude-sonnet-4-6",
            api_key=api_key,
            max_tokens=max_tokens,
        )

    if provider == "AZUREAI":
        api_key_secret = os.getenv("API_KEY_SECRET", os.getenv("AZUREAI_OPENAI_API_KEY_SECRET"))
        api_key = secrets_manager.get_secret(api_key_secret) if api_key_secret else None

        if not api_key:
            raise ValueError(
                "PROVIDER is set to AZUREAI but API_KEY_SECRET/AZUREAI_OPENAI_API_KEY_SECRET is not set or secret not found."
            )

        azureai_base_url = os.getenv("AZUREAI_OPENAI_URL", "https://ap-azureopenai.openai.azure.com")

        return LLM(
            provider="openai",
            model=model_name or "gpt-4.1-mini",
            base_url=f"{azureai_base_url.rstrip('/')}" \
            f"/openai/v1",
            api_key=api_key,
            default_headers={"api-key": api_key},
            api="responses",
            store=False,
            temperature=temperature,
            max_tokens=max_tokens,
        )

    if provider == "MICROSOFTFOUNDRYAI":
        api_key_secret = os.getenv("API_KEY_SECRET", os.getenv("MICROSOFTFOUNDRYAI_API_KEY_SECRET"))
        api_key = secrets_manager.get_secret(api_key_secret) if api_key_secret else None

        if not api_key:
            raise ValueError(
                "PROVIDER is set to MICROSOFTFOUNDRYAI but API_KEY_SECRET/MICROSOFTFOUNDRYAI_API_KEY_SECRET is not set or secret not found."
            )

        foundry_endpoint = os.getenv("MICROSOFTFOUNDRYAI_ENDPOINT")
        if not foundry_endpoint:
            raise ValueError(
                "PROVIDER is set to MICROSOFTFOUNDRYAI but MICROSOFTFOUNDRYAI_ENDPOINT is not set."
            )

        foundry_model = model_name or "gpt-4.1-mini"

        if foundry_endpoint.lower().endswith("/anthropic"):
            foundry_llm_kwargs = {
                "provider": "anthropic",
                "is_litellm": True,
                "model": f"anthropic/{foundry_model}",
                "base_url": foundry_endpoint,
            }
        elif foundry_endpoint.lower().endswith("/openai"):
            foundry_llm_kwargs = {
                "model": f"azure/{foundry_model}",
                "endpoint": f"{foundry_endpoint}/deployments/{foundry_model}",
            }
        else:
            raise ValueError(
                "MICROSOFTFOUNDRYAI_ENDPOINT must end with /anthropic or /openai."
            )

        return LLM(
            **foundry_llm_kwargs,
            api_key=api_key,
            max_tokens=max_tokens,
        )

    if provider == "GEMINIAI":
        api_key_secret = os.getenv("API_KEY_SECRET", os.getenv("GEMINIAI_API_KEY_SECRET"))
        api_key = secrets_manager.get_secret(api_key_secret) if api_key_secret else None

        if not api_key:
            raise ValueError("PROVIDER is set to GEMINIAI but GEMINIAI_API_KEY_SECRET is not set or secret not found.")

        os.environ["GEMINI_API_KEY"] = api_key
        os.environ["GOOGLE_API_KEY"] = api_key
        return LLM(
            model=model_name or "gemini/gemini-2.5-flash",
            api_key=api_key,
            provider="gemini",
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
        )

    if provider == "BEDROCKAI":
        api_key_secret = os.getenv("API_KEY_SECRET", os.getenv("BEDROCKAI_API_KEY_SECRET"))
        api_key = secrets_manager.get_secret(api_key_secret) if api_key_secret else None
        aws_region_name = os.getenv("AWS_REGION_NAME")

        if not api_key:
            raise ValueError(
                "PROVIDER is set to BEDROCKAI but BEDROCKAI_API_KEY_SECRET is not set or secret not found."
            )

        if not aws_region_name:
            raise ValueError(
                "AWS_REGION_NAME is required when PROVIDER is set to BEDROCKAI"
            )

        os.environ["AWS_BEARER_TOKEN_BEDROCK"] = api_key

        def bedrock_model_route(model_id: str) -> str:
            if model_id.lower().startswith("arn:"):
                return f"bedrock/converse/{model_id}"

            return f"bedrock/{model_id}"

        return LLM(
            model=bedrock_model_route(model_name or "us.anthropic.claude-opus-5"),
            api_key=api_key,
            aws_region_name=aws_region_name,
            max_tokens=max_tokens,
        )

    return LLM(
        model=model_name or "ollama/llama3.2",
        base_url=os.getenv("OLLAMA_BASE_URL"),
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
    )


@CrewBase
class LatestAiDevelopment:
    """LatestAiDevelopment crew"""

    agents: List[BaseAgent]
    tasks: List[Task]

    provider = os.getenv("PROVIDER", "OLLAMA").upper()

    AGENT_LLM_SETTINGS: Dict[str, Dict[str, Any]] = {
        "query_intake_agent": {
            "model": "gpt-6.1-sol",
            "temperature": 0.0,
            "top_p": 0.0,
            "max_tokens": 100000,
        },
        "sql_generation_agent": {
            "model": "gpt-6.1-sol",
            "temperature": 0.0,
            "top_p": 0.0,
            "max_tokens": 100000,
        },
        "connectivity_verification_agent": {
            "model": "gpt-6.1-sol",
            "temperature": 0.0,
            "top_p": 0.0,
            "max_tokens": 100000,
        },
        "query_execution_agent": {
            "model": "gpt-6.1-sol",
            "temperature": 0.0,
            "top_p": 0.0,
            "max_tokens": 100000,
        },
        "response_generation_agent": {
            "model": "gpt-6.1-sol",
            "temperature": 0.0,
            "top_p": 0.0,
            "max_tokens": 100000,
        },
        "audit_logging_agent": {
            "model": "gpt-6.1-sol",
            "temperature": 0.0,
            "top_p": 0.0,
            "max_tokens": 100000,
        },
    }

    def _get_generation_param(
        self,
        cfg: Dict[str, Any],
        key: str,
        env_key: str,
        default: Any,
        cast_type,
    ):
        if key in cfg and cfg[key] is not None:
            return cast_type(cfg[key])

        env_value = os.getenv(env_key)
        if env_value is not None:
            return cast_type(env_value)

        return default

    def _llm_for_agent(self, agent_key: str) -> LLM:
        if agent_key not in self.AGENT_LLM_SETTINGS:
            raise ValueError(f"Unknown agent_key '{agent_key}' in AGENT_LLM_SETTINGS.")

        cfg = self.AGENT_LLM_SETTINGS[agent_key]

        model_name = cfg.get("model")
        temperature = self._get_generation_param(
            cfg, "temperature", "DEFAULT_TEMPERATURE", 0.1, float
        )
        top_p = self._get_generation_param(cfg, "top_p", "DEFAULT_TOP_P", 0.9, float)
        max_tokens = self._get_generation_param(
            cfg, "max_tokens", "DEFAULT_MAX_TOKENS", 1024, int
        )

        return build_llm(
            provider=self.provider,
            model_name=model_name,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
        )

    @agent
    def query_intake_agent(self) -> Agent:
        return Agent(
            config=self.agents_config['query_intake_agent'],
            verbose=True,
            llm=self._llm_for_agent('query_intake_agent'),
        )

    @agent
    def sql_generation_agent(self) -> Agent:
        return Agent(
            config=self.agents_config['sql_generation_agent'],
            verbose=True,
            llm=self._llm_for_agent('sql_generation_agent'),
        )

    @agent
    def connectivity_verification_agent(self) -> Agent:
        return Agent(
            config=self.agents_config['connectivity_verification_agent'],
            verbose=True,
            llm=self._llm_for_agent('connectivity_verification_agent'),
            tools=[KeyretainedTool()],
        )

    @agent
    def query_execution_agent(self) -> Agent:
        return Agent(
            config=self.agents_config['query_execution_agent'],
            verbose=True,
            llm=self._llm_for_agent('query_execution_agent'),
        )

    @agent
    def response_generation_agent(self) -> Agent:
        return Agent(
            config=self.agents_config['response_generation_agent'],
            verbose=True,
            llm=self._llm_for_agent('response_generation_agent'),
        )

    @agent
    def audit_logging_agent(self) -> Agent:
        return Agent(
            config=self.agents_config['audit_logging_agent'],
            verbose=True,
            llm=self._llm_for_agent('audit_logging_agent'),
        )

    @task
    def query_intake_agent_task(self) -> Task:
        return Task(
            config=self.tasks_config['query_intake_agent_task'],
        )

    @task
    def sql_generation_agent_task(self) -> Task:
        return Task(
            config=self.tasks_config['sql_generation_agent_task'],
            context=[self.query_intake_agent_task()],
        )

    @task
    def connectivity_verification_agent_task(self) -> Task:
        return Task(
            config=self.tasks_config['connectivity_verification_agent_task'],
            context=[self.sql_generation_agent_task()],
        )

    @task
    def query_execution_agent_task(self) -> Task:
        return Task(
            config=self.tasks_config['query_execution_agent_task'],
            context=[self.connectivity_verification_agent_task()],
        )

    @task
    def response_generation_agent_task(self) -> Task:
        return Task(
            config=self.tasks_config['response_generation_agent_task'],
            context=[self.query_execution_agent_task()],
        )

    @task
    def audit_logging_agent_task(self) -> Task:
        return Task(
            config=self.tasks_config['audit_logging_agent_task'],
            context=[
                self.query_intake_agent_task(),
                self.sql_generation_agent_task(),
                self.connectivity_verification_agent_task(),
                self.query_execution_agent_task(),
            ],
        )

    @crew
    def crew(self) -> Crew:
        return Crew(
            agents=self.agents,
            tasks=self.tasks,
            process=Process.sequential,
            verbose=True,
            tracing=get_tracing_backend() == "CREWAI",
        )
