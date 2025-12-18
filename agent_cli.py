import os
import asyncio
import warnings
import time
import re
from typing import Dict, Tuple
import logging

from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown
from rich.live import Live
from rich.spinner import Spinner
from rich.padding import Padding
from rich.text import Text

from prompt_toolkit import PromptSession
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.styles import Style
from prompt_toolkit.lexers import PygmentsLexer
from pygments.lexers.markup import MarkdownLexer
from prompt_toolkit.history import FileHistory

from langchain_core.messages import HumanMessage, AIMessage, ToolMessage, AIMessageChunk

try:
    from agent import AgentWorkflow, logger
except ImportError:
    import sys
    sys.path.append(".")
    from agent import AgentWorkflow, logger

warnings.filterwarnings("ignore")
console = Console()

_THOUGHT_RE = re.compile(r"<thought>(.*?)</thought>", re.DOTALL)

class TokenTracker:
    def __init__(self):
        self.input = 0
        self.output = 0
    
    def update(self, msg):
        try:
            if hasattr(msg, "usage_metadata") and msg.usage_metadata:
                usage = msg.usage_metadata
                self.input += usage.get("input_tokens", 0)
                self.output += usage.get("output_tokens", 0)
                return

            if hasattr(msg, "response_metadata") and msg.response_metadata:
                meta = msg.response_metadata
                usage = meta.get("token_usage") or meta
                
                if isinstance(usage, dict):
                    p_tokens = usage.get("prompt_tokens", 0)
                    c_tokens = usage.get("completion_tokens", 0)
                    if p_tokens or c_tokens:
                        self.input += p_tokens
                        self.output += c_tokens
        except Exception:
            pass

    def render(self, duration: float) -> str:
        txt = f"⏱ {duration:.1f}s"
        if self.input or self.output:
            txt += f" | In: {self.input} Out: {self.output}"
        return f"[bright_black]{txt}[/]"

def get_key_bindings():
    kb = KeyBindings()
    @kb.add('enter')
    def _(event):
        buf = event.current_buffer
        if not buf.text.strip(): return
        buf.validate_and_handle()
    @kb.add('escape', 'enter')
    def _(event):
        event.current_buffer.insert_text("\n")
    return kb

def parse_thought(text: str) -> Tuple[str, str, bool]:
    match = _THOUGHT_RE.search(text)
    
    if match:
        thought_content = match.group(1).strip()
        clean_text = _THOUGHT_RE.sub('', text).strip()
        return thought_content, clean_text, True
    
    if "<thought>" in text and "</thought>" not in text:
        start = text.find("<thought>") + len("<thought>")
        partial_thought = text[start:].strip()
        return partial_thought, text[:text.find("<thought>")], False

    return "", text, False

async def process_stream(agent_app, user_input: str, thread_id: str):
    config = {"configurable": {"thread_id": thread_id}}
    tracker = TokenTracker()
    start_time = time.time()
    
    accumulated_text = ""
    printed_thoughts = set()
    
    try:
        with Live(Spinner("dots", style="cyan"), refresh_per_second=10, console=console, transient=True) as live:
            
            async for event in agent_app.astream(
                {"messages": [HumanMessage(content=user_input)]},
                config=config,
                stream_mode="messages"
            ):
                msg, metadata = event
                node = metadata.get("langgraph_node")
                tracker.update(msg)

                if node == "agent" and isinstance(msg, (AIMessage, AIMessageChunk)):
                    if msg.tool_calls:
                        for tc in msg.tool_calls:
                            tool_name = tc.get('name')
                            if tool_name:
                                live.update(Spinner("earth", text=f"[bold cyan]Вызов:[/] {tool_name}", style="cyan"))                    
                    
                    if msg.content:
                        chunk = msg.content if isinstance(msg.content, str) else ""
                        if isinstance(msg.content, list):
                            chunk = "".join(x.get("text", "") for x in msg.content if isinstance(x, dict))

                        if isinstance(msg, AIMessageChunk):
                            accumulated_text += chunk
                        else:
                            if not accumulated_text:
                                accumulated_text = chunk
                        
                        thought, clean_text, is_complete = parse_thought(accumulated_text)
                        
                        if thought:
                            live.update(Spinner("dots", text=f"[yellow italic]{thought}...[/]", style="yellow"))
                            
                            if is_complete and thought not in printed_thoughts:
                                live.console.print(Padding(f"➤ [italic yellow]{thought}[/]", (0, 0, 0, 2)))
                                printed_thoughts.add(thought)
                                accumulated_text = clean_text

                        elif clean_text.strip() and "<thought>" not in accumulated_text:
                            pretty_md = re.sub(r'\n{3,}', '\n\n', clean_text)
                            live.update(Padding(Markdown(pretty_md), (1, 1)))

                elif node == "tools" and isinstance(msg, ToolMessage):
                    res = str(msg.content)
                    preview = (res[:200] + "...") if len(res) > 200 else res
                    preview = preview.replace("\n", " ")
                    
                    live.console.print(Padding(f"[dim white]✓ {msg.name}: {preview}[/]", (0, 0, 0, 4)))
                    live.update(Spinner("dots", text="Анализирую...", style="cyan"))

    except KeyboardInterrupt:
        console.print("\n[bold red]🛑 Генерация прервана пользователем[/]")
        return 

    _, final_clean, _ = parse_thought(accumulated_text)
    if final_clean.strip():
        console.print(Padding(Markdown(final_clean), (0, 1, 1, 1)))
    
    console.print(tracker.render(time.time() - start_time), justify="right")

async def main():
    os.system("cls" if os.name == "nt" else "clear")
    console.print(Panel("[bold blue]AI Agent CLI[/]", subtitle="v2.5"))

    # Сохраняем текущий уровень логирования
    # logger импортируется из agent (строка 29 или 33 вашего кода)
    previous_level = logger.getEffectiveLevel()
    
    # Устанавливаем уровень WARNING, чтобы скрыть обычные сообщения (INFO)
    logger.setLevel(logging.WARNING)

    try:
        # Теперь спиннер будет крутиться чисто, без прыжков строк
        with console.status("[bold green]Запуск системы...[/]", spinner="dots"):
            workflow = AgentWorkflow()
            await workflow.initialize_resources()
            agent_app = workflow.build_graph()
        
        console.print("[bold green]Система готова к работе![/]")

    except Exception as e:
        console.print(f"[bold red]Ошибка инициализации:[/] {e}")
        return
    finally:
        # Возвращаем уровень логирования обратно (важно для работы агента)
        logger.setLevel(previous_level)
        
    model = workflow.config.gemini_model if workflow.config.provider == "gemini" else workflow.config.openai_model
    console.print(f"[dim]Model: {model} | Tools: {len(workflow.tools)}[/]")
    console.print("[dim]Type 'exit', 'reset' or just chat.[/]\n")

    session = PromptSession(
        history=FileHistory(".history"),
        style=Style.from_dict({"prompt": "bold cyan"}),
        key_bindings=get_key_bindings(),
        lexer=PygmentsLexer(MarkdownLexer)
    )

    thread_id = "main_session"

    while True:
        try:
            user_input = await session.prompt_async("You > ")
            user_input = user_input.strip()

            if not user_input: continue
            if user_input.lower() in ["exit", "quit"]: break
            if user_input.lower() in ["clear", "reset"]:
                thread_id = f"session_{int(time.time())}"
                console.print("[yellow]♻ New session started[/]")
                continue

            await process_stream(agent_app, user_input, thread_id)
            console.print()

        except KeyboardInterrupt:
            console.print("\n[yellow]Отменено. Введите 'exit' для выхода.[/]")
            continue
        except Exception as e:
            console.print(f"[bold red]Error:[/] {e}")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass