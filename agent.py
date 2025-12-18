import os
import json
import asyncio
import logging
from pathlib import Path
from datetime import datetime
from typing import List, Literal, TypedDict, Annotated, Optional

# --- LANGCHAIN & LANGGRAPH ---
from langchain_core.language_models import BaseChatModel
from langchain_core.tools import tool, BaseTool
from langchain_core.messages import BaseMessage, SystemMessage, AIMessage, RemoveMessage, HumanMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import MemorySaver

# --- PROVIDERS ---
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI

# --- CONFIG & UTILS ---
from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict
from dotenv import load_dotenv

# --- LOCAL MODULES ---
# 1. LOGGING (Подключаем ваш модуль)
try:
    from logging_config import setup_logging
    # Инициализируем логгер через ваш конфиг (Rich + File + Filters)
    logger = setup_logging() 
except ImportError:
    # Fallback на случай, если файла нет
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("agent")
    logger.warning("logging_config.py not found, using default logging.")

# 2. FILE TOOLS
try:
    from delete_tools import SafeDeleteFileTool, SafeDeleteDirectoryTool
except ImportError:
    SafeDeleteFileTool = SafeDeleteDirectoryTool = None

# 3. MCP CLIENT
try:
    from langchain_mcp_adapters.client import MultiServerMCPClient
except ImportError:
    MultiServerMCPClient = None


# ==========================================
# 1. КОНФИГУРАЦИЯ (Pydantic)
# ==========================================

class AgentConfig(BaseSettings):
    """
    Конфигурация агента. Читает параметры из файла .env
    """
    model_config = SettingsConfigDict(env_file='.env', env_file_encoding='utf-8', extra='ignore')

    provider: Literal["gemini", "openai"] = "gemini"
    
    # API Keys & Models
    gemini_api_key: Optional[SecretStr] = None
    gemini_model: str = "gemini-1.5-flash"
    
    openai_api_key: Optional[SecretStr] = None
    openai_model: str = "gpt-4o"
    openai_base_url: Optional[str] = None

    # LLM Settings
    temperature: float = 0.5
    max_retries: int = 3
    
    # Agent Logic Settings
    use_long_term_memory: bool = Field(default=False, alias="LONG_TERM_MEMORY")
    summary_threshold: int = Field(default=15, alias="SESSION_SIZE")
    
    # Paths
    mcp_config_path: Path = Path("mcp.json")
    prompt_path: Path = Path("prompt.txt")
    memory_db_path: str = "./memory_db"

    def get_llm(self) -> BaseChatModel:
        """Инициализация LLM в зависимости от провайдера."""
        if self.provider == "gemini":
            if not self.gemini_api_key:
                raise ValueError("GEMINI_API_KEY не найден в .env")
            return ChatGoogleGenerativeAI(
                model=self.gemini_model,
                temperature=self.temperature,
                google_api_key=self.gemini_api_key.get_secret_value(),
                max_retries=self.max_retries,
                convert_system_message_to_human=True
            )
        elif self.provider == "openai":
            if not self.openai_api_key:
                raise ValueError("OPENAI_API_KEY не найден в .env")
            return ChatOpenAI(
                model=self.openai_model,
                temperature=self.temperature,
                api_key=self.openai_api_key.get_secret_value(),
                base_url=self.openai_base_url,
                max_retries=self.max_retries,
                model_kwargs={"stream_options": {"include_usage": True}}
            )
        raise ValueError(f"Unknown provider: {self.provider}")


# ==========================================
# 2. СОСТОЯНИЕ ГРАФА
# ==========================================

class AgentState(TypedDict):
    """
    messages: История сообщений (автоматически объединяется).
    summary: Сжатое содержание предыдущего контекста.
    """
    messages: Annotated[list[BaseMessage], add_messages]
    summary: str


# ==========================================
# 3. WORKFLOW АГЕНТА
# ==========================================

class AgentWorkflow:
    def __init__(self):
        load_dotenv()
        self.config = AgentConfig()
        self.tools: List[BaseTool] = []
        self.llm: Optional[BaseChatModel] = None
        self.llm_with_tools: Optional[BaseChatModel] = None
        self._cached_prompt: Optional[str] = None

    @staticmethod
    def _messages_to_summary_text(messages: List[BaseMessage], *, per_message_limit: int = 800, total_limit: int = 6000) -> str:
        parts: List[str] = []
        total = 0
        for m in messages:
            role = getattr(m, "type", m.__class__.__name__)
            content = getattr(m, "content", "")
            if isinstance(content, list):
                content_str = "".join(
                    (x.get("text", "") if isinstance(x, dict) else str(x))
                    for x in content
                )
            else:
                content_str = str(content)

            content_str = content_str.strip()
            if len(content_str) > per_message_limit:
                content_str = content_str[:per_message_limit] + "..."

            chunk = f"{role}: {content_str}".strip()
            if not chunk:
                continue

            if total + len(chunk) + 1 > total_limit:
                break
            parts.append(chunk)
            total += len(chunk) + 1

        return "\n".join(parts)

    async def initialize_resources(self):
        """Асинхронная инициализация всех ресурсов (LLM, инструменты, память)."""
        logger.info(f"Initializing agent with provider: [bold cyan]{self.config.provider}[/]", extra={"markup": True})
        self.llm = self.config.get_llm()

        # 1. Файловые инструменты (если модуль найден)
        if SafeDeleteFileTool and SafeDeleteDirectoryTool:
            cwd = Path.cwd()
            self.tools.extend([
                SafeDeleteFileTool(root_dir=cwd),
                SafeDeleteDirectoryTool(root_dir=cwd)
            ])
            logger.info("File system tools loaded (Sandbox enabled).")

        # 2. Долговременная память
        if self.config.use_long_term_memory:
            self._init_memory_tools()

        # 3. MCP Tools (если есть конфиг и библиотека)
        if MultiServerMCPClient and self.config.mcp_config_path.exists():
            await self._init_mcp_tools()

        # Привязка инструментов к LLM
        if self.tools:
            self.llm_with_tools = self.llm.bind_tools(self.tools)
        else:
            self.llm_with_tools = self.llm

    def _init_memory_tools(self):
        """Инициализация инструментов памяти (Recall, Remember, Forget)."""
        try:
            from memory_manager import MemoryManager
            # Инициализируем менеджер (Singleton)
            memory = MemoryManager(db_path=self.config.memory_db_path)
            
            @tool
            async def remember_fact(text: str, category: str = "general") -> str:
                """
                Сохраняет важный факт о пользователе, проекте или предпочтениях.
                """
                return await memory.aremember(text, {"type": category})

            @tool
            async def recall_facts(query: str) -> str:
                """
                Ищет информацию в долговременной памяти по смысловому запросу.
                """
                facts = await memory.arecall(query)
                return "\n".join(f"- {f}" for f in facts) if facts else "В памяти ничего не найдено."

            @tool
            async def forget_fact(query: str) -> str:
                """
                Удаляет информацию из памяти. Используйте, если пользователь просит забыть что-то,
                или если информация стала неверной.
                """
                try:
                    count = await memory.adelete_fact_by_query(query)
                    if count > 0:
                        return f"Успешно забыто фактов: {count}"
                    return "Факты для удаления не найдены."
                except Exception as e:
                    return f"Ошибка при удалении: {e}"

            self.tools.extend([remember_fact, recall_facts, forget_fact])
            logger.info("Memory tools loaded (Remember, Recall, Forget).")
        except ImportError:
            logger.warning("MemoryManager module not found. Memory tools disabled.")
        except Exception as e:
            logger.error(f"Error loading memory tools: {e}")

    async def _init_mcp_tools(self):
        """Загрузка инструментов через Model Context Protocol (MCP)."""
        if not self.config.mcp_config_path.exists():
            return

        try:
            raw_cfg = json.loads(self.config.mcp_config_path.read_text("utf-8"))
            
            mcp_cfg = {}
            for name, config in raw_cfg.items():
                # Пропускаем, если выключено
                if not config.get("enabled", True):
                    continue
                
                # Создаем чистую копию для клиента MCP
                clean_config = config.copy()
                
                # УДАЛЯЕМ ключ 'enabled', чтобы клиент MCP не ругался
                clean_config.pop("enabled", None)
                
                # Подставляем пути
                current_args = clean_config.get("args", [])
                clean_config["args"] = [
                    arg.replace("{filesystem_path}", str(Path.cwd())) 
                    for arg in current_args
                ]
                
                mcp_cfg[name] = clean_config
            
            if mcp_cfg:
                client = MultiServerMCPClient(mcp_cfg)
                if hasattr(asyncio, "timeout"):
                    async with asyncio.timeout(60):
                        new_tools = await client.get_tools()
                else:
                    new_tools = await asyncio.wait_for(client.get_tools(), timeout=60)

                self.tools.extend(new_tools)
                logger.info(f"Loaded {len(new_tools)} MCP tools from: {list(mcp_cfg.keys())}")
        except Exception as e:
            logger.error(f"MCP Load Error: {e}")

    def _get_system_prompt(self) -> str:
        """Генерация системного промпта с кэшированием шаблона."""
        if not self._cached_prompt:
            if self.config.prompt_path.exists():
                self._cached_prompt = self.config.prompt_path.read_text("utf-8")
            else:
                self._cached_prompt = "Role: AI Assistant. Be helpful."

        now = datetime.now()
        prompt = self._cached_prompt.replace("{{current_date}}", now.strftime("%Y-%m-%d (%A)"))
        prompt = prompt.replace("{{cwd}}", str(Path.cwd()))
        return prompt

    # ==========================================
    # 4. УЗЛЫ ГРАФА (NODES)
    # ==========================================

    async def _summarize_node(self, state: AgentState):
        """
        Узел сжатия истории. Гарантирует, что история всегда начинается с HumanMessage.
        """
        messages = state["messages"]
        summary = state.get("summary", "")

        if not self.llm:
            return {}

        if len(messages) > self.config.summary_threshold:
            # 1. Сначала определяем, сколько сообщений мы ХОТИМ оставить
            keep_last = 4 
            
            # Если сообщений и так мало, выходим
            if len(messages) <= keep_last:
                return {}

            # 3. Умная корректировка границы:
            # Проверяем первое сообщение, которое ОСТАНЕТСЯ (messages[-keep_last]).
            # Если оно НЕ HumanMessage, нам нужно удалить и его тоже.
            # Мы сдвигаем границу вправо, пока не найдем HumanMessage или пока не кончатся сообщения.
            
            idx_start_keep = len(messages) - keep_last
            
            while idx_start_keep < len(messages):
                msg = messages[idx_start_keep]
                if isinstance(msg, HumanMessage):
                    break # Нашли начало диалога с пользователем, всё ок
                
                # Если это не пользователь (AI или Tool), это сообщение тоже надо сжать/удалить
                idx_start_keep += 1
            
            # Если мы дошли до конца и не нашли HumanMessage, значит удаляем вообще всё
            # (это лучше, чем отправлять битую историю)
            
            # Формируем окончательный список на удаление
            to_summarize = messages[:idx_start_keep]
            
            if not to_summarize:
                return {}

            to_summarize_text = self._messages_to_summary_text(to_summarize)

            prompt = (
                f"Current summary: {summary}\n"
                f"New interactions:\n{to_summarize_text}\n\n"
                "Create a concise updated summary of the conversation, preserving key facts and user requests."
            )
            
            try:
                # Генерируем саммари
                res = await self.llm.ainvoke(prompt)
                
                # Удаляем сообщения
                delete_msgs = [RemoveMessage(id=m.id) for m in to_summarize if m.id]
                
                logger.info(f"Summarized context. Removed {len(delete_msgs)} messages. New history starts with Human.")
                return {"summary": res.content, "messages": delete_msgs}
            except Exception as e:
                logger.error(f"Summarization failed: {e}")
                return {}
        
        return {}
        
    async def _agent_node(self, state: AgentState):
        """
        Главный узел агента.
        """
        if not self.llm_with_tools:
            raise RuntimeError("AgentWorkflow is not initialized. Call initialize_resources() before build_graph()/run.")

        messages = state["messages"]
        summary = state.get("summary", "")
        
        sys_text = self._get_system_prompt()
        if summary:
            sys_text += f"\n\n### Context Summary\n{summary}"
        
        sys_msg = SystemMessage(content=sys_text)
        user_history = [m for m in messages if not isinstance(m, SystemMessage)]
        
        final_messages = [sys_msg] + user_history
        
        response = await self.llm_with_tools.ainvoke(final_messages)
        return {"messages": [response]}

    # ==========================================
    # 5. СБОРКА ГРАФА
    # ==========================================

    def build_graph(self):
        """Компиляция графа LangGraph."""
        workflow = StateGraph(AgentState)

        workflow.add_node("summarize", self._summarize_node)
        workflow.add_node("agent", self._agent_node)
        
        if self.tools:
            workflow.add_node("tools", ToolNode(self.tools))

        workflow.add_edge(START, "summarize")
        workflow.add_edge("summarize", "agent")

        def should_continue(state):
            last_msg = state["messages"][-1]
            tool_calls = getattr(last_msg, "tool_calls", None)
            return "tools" if tool_calls else END

        workflow.add_conditional_edges(
            "agent",
            should_continue,
            ["tools", END] if self.tools else [END]
        )

        if self.tools:
            workflow.add_edge("tools", "agent")

        return workflow.compile(checkpointer=MemorySaver())


# ==========================================
# 6. ТОЧКА ВХОДА (TEST)
# ==========================================

if __name__ == "__main__":
    async def main():
        print("--- Testing Agent Initialization ---")
        try:
            wf = AgentWorkflow()
            await wf.initialize_resources()
            app = wf.build_graph()
            print("✅ Agent Graph built successfully.")
            print(f"🔧 Tools: {[t.name for t in wf.tools]}")
        except Exception as e:
            print(f"❌ Initialization failed: {e}")

    asyncio.run(main())