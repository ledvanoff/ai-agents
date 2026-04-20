#!/usr/bin/env python3
"""
CLI бот для взаимодействия с LLM через OpenRouter.
Демонстрирует работу с историей диалога, метриками и красивым выводом.
"""

import os
import sys
from typing import List, Dict, Optional
from dotenv import load_dotenv
from openai import OpenAI
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.markdown import Markdown
from rich import box


# Инициализация Rich консоли для красивого вывода
console = Console()

# Системный промпт - определяет роль и поведение ассистента
# ЗАДАНИЕ: Вставьте сюда ваш системный промпт, который определит поведение бота
# Например: "Ты — профессиональный банковский консультант..."
SYSTEM_PROMPT = """Ты — специалист технической поддержки IT-компании.
Помогай пользователям решать проблемы с программным обеспечением.
Задавай уточняющие вопросы, давай пошаговые инструкции.
Используй простой язык без технического жаргона.
Будь терпелив и эмпатичен к проблемам пользователей."""


class ChatBot:
    """Простой CLI бот для общения с LLM."""

    def __init__(self):
        """Инициализация бота с загрузкой конфигурации."""
        load_dotenv()

        # Стратегия 2 (Задание 4): порог по числу несистемных сообщений и сколько последних оставить.
        self.history_summarize_threshold = int(os.getenv("HISTORY_SUMMARIZE_THRESHOLD", "12"))
        self.history_keep_recent = int(os.getenv("HISTORY_KEEP_RECENT", "6"))

        api_key = os.getenv("OPENROUTER_API_KEY")
        base_url = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
        self.model_name = os.getenv("MODEL_NAME", "openai/gpt-4o-mini")

        if not api_key:
            console.print("[red]❌ Ошибка: OPENROUTER_API_KEY не найден в .env файле![/red]")
            sys.exit(1)

        self.client = OpenAI(
            api_key=api_key,
            base_url=base_url,
        )

        self.conversation_history: List[Dict[str, str]] = []

        if SYSTEM_PROMPT:
            self.conversation_history.append({
                "role": "system",
                "content": SYSTEM_PROMPT
            })

        self.session_metrics = {
            "total_prompt_tokens": 0,
            "total_completion_tokens": 0,
            "total_tokens": 0,
            "messages_count": 0,
        }

    def _split_system_and_rest(self) -> tuple[List[Dict[str, str]], List[Dict[str, str]]]:
        """Системный промпт (если есть) и остальная история."""
        if self.conversation_history and self.conversation_history[0].get("role") == "system":
            return [self.conversation_history[0]], self.conversation_history[1:]
        return [], list(self.conversation_history)

    def _summarize_transcript(self, messages: List[Dict[str, str]]) -> str:
        """Один вызов LLM: сжать переданные реплики в 2–3 предложения."""
        lines: List[str] = []
        for m in messages:
            label = "Пользователь" if m["role"] == "user" else "Ассистент"
            lines.append(f"{label}: {m['content']}")
        blob = "\n".join(lines)
        instruction = (
            "Кратко резюмируй эту переписку в 2–3 предложениях по-русски. "
            "Сохрани факты, цифры и договорённости. Без вступлений — только текст резюме."
        )
        summarizer_messages = [
            {"role": "user", "content": f"{instruction}\n\n---\n{blob}"},
        ]
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=summarizer_messages,
        )
        text = (response.choices[0].message.content or "").strip()
        if response.usage:
            u = response.usage.model_dump()
            self.session_metrics["total_prompt_tokens"] += u.get("prompt_tokens", 0)
            self.session_metrics["total_completion_tokens"] += u.get("completion_tokens", 0)
            self.session_metrics["total_tokens"] += u.get("total_tokens", 0)
            self.session_metrics["messages_count"] += 1
        return text

    def summarize_history(self) -> None:
        """
        Стратегия 2: при длинной истории сжать старые реплики в резюме через LLM,
        сохранить системный промпт и последние history_keep_recent несистемных сообщений.
        """
        system_msgs, others = self._split_system_and_rest()
        if len(others) <= self.history_summarize_threshold:
            return

        # Оставляем «хвост» не длиннее порога, чтобы после сжатия снова не триггерить подряд.
        keep = min(self.history_keep_recent, len(others) - 1)
        keep = max(1, keep)
        to_summarize = others[:-keep]
        tail = others[-keep:]

        if not to_summarize:
            return

        try:
            summary = self._summarize_transcript(to_summarize)
        except Exception as e:
            console.print(f"[yellow]⚠ Не удалось получить резюме ({e}). Обрезаю старые реплики.[/yellow]")
            self.conversation_history = system_msgs + tail
            return

        if not summary:
            self.conversation_history = system_msgs + tail
            return

        compressed: List[Dict[str, str]] = [
            {
                "role": "user",
                "content": "[Сжатый контекст ранее в диалоге]\n" + summary,
            }
        ] + tail
        self.conversation_history = system_msgs + compressed
        console.print("[dim]📎 История сжата: ранние реплики заменены резюме.[/dim]")

    def add_message(self, role: str, content: str):
        """Добавить сообщение в историю диалога."""
        self.conversation_history.append({
            "role": role,
            "content": content
        })

        _, others = self._split_system_and_rest()
        if len(others) > self.history_summarize_threshold:
            self.summarize_history()

    def clear_history(self):
        """Очистить историю диалога."""
        self.conversation_history = []
        if SYSTEM_PROMPT:
            self.conversation_history.append({
                "role": "system",
                "content": SYSTEM_PROMPT
            })
        console.print("[yellow]📝 История диалога очищена[/yellow]\n")

    def display_metrics(self, usage: Optional[dict], finish_reason: Optional[str] = None):
        """Отобразить метрики и метаданные ответа."""
        if not usage:
            return

        prompt_tokens = usage.get("prompt_tokens", 0)
        completion_tokens = usage.get("completion_tokens", 0)
        total_tokens = usage.get("total_tokens", 0)

        self.session_metrics["total_prompt_tokens"] += prompt_tokens
        self.session_metrics["total_completion_tokens"] += completion_tokens
        self.session_metrics["total_tokens"] += total_tokens
        self.session_metrics["messages_count"] += 1

        table = Table(title="📊 Метрики ответа", box=box.ROUNDED, show_header=True)
        table.add_column("Параметр", style="cyan")
        table.add_column("Значение", style="green")

        table.add_row("Модель", self.model_name)
        table.add_row("Prompt токены", str(prompt_tokens))
        table.add_row("Completion токены", str(completion_tokens))
        table.add_row("Всего токены", str(total_tokens))

        if finish_reason:
            table.add_row("Finish reason", finish_reason)

        console.print(table)

        session_table = Table(title="🎯 Статистика сессии", box=box.ROUNDED)
        session_table.add_column("Параметр", style="cyan")
        session_table.add_column("Значение", style="magenta")

        session_table.add_row("Сообщений", str(self.session_metrics["messages_count"]))
        session_table.add_row("Всего токенов", str(self.session_metrics["total_tokens"]))

        console.print(session_table)
        console.print()

    def display_stats(self):
        """Показать статистику сессии."""
        console.print("\n[bold cyan]📈 Статистика текущей сессии:[/bold cyan]")

        stats_table = Table(box=box.DOUBLE)
        stats_table.add_column("Метрика", style="cyan", no_wrap=True)
        stats_table.add_column("Значение", style="green")

        stats_table.add_row("Модель", self.model_name)
        stats_table.add_row("Сообщений в сессии", str(self.session_metrics["messages_count"]))
        stats_table.add_row("Сообщений в истории", str(len(self.conversation_history)))
        stats_table.add_row("Prompt токены", str(self.session_metrics["total_prompt_tokens"]))
        stats_table.add_row("Completion токены", str(self.session_metrics["total_completion_tokens"]))
        stats_table.add_row("Всего токены", str(self.session_metrics["total_tokens"]))

        console.print(stats_table)
        console.print()

    def send_message(self, user_message: str) -> Optional[str]:
        """Отправить сообщение в LLM и получить ответ."""
        self.add_message("user", user_message)

        try:
            with console.status("[bold green]🤔 Думаю...", spinner="dots"):
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=self.conversation_history,
                )

            assistant_message = response.choices[0].message.content
            finish_reason = response.choices[0].finish_reason

            self.add_message("assistant", assistant_message)

            console.print(Panel(
                Markdown(assistant_message),
                title="🤖 Ассистент",
                border_style="blue",
                padding=(1, 2)
            ))

            self.display_metrics(response.usage.model_dump() if response.usage else None, finish_reason)

            return assistant_message

        except Exception as e:
            console.print(f"[red]❌ Ошибка при обращении к LLM: {e}[/red]\n")
            if self.conversation_history and self.conversation_history[-1]["role"] == "user":
                self.conversation_history.pop()
            return None

    def show_welcome(self):
        """Показать приветственное сообщение."""
        welcome_text = """
# 🤖 CLI LLM Бот

Образовательный проект для работы с LLM через OpenRouter API.

**Доступные команды:**
- `/exit` - выход из программы
- `/clear` - очистить историю диалога
- `/stats` - показать статистику сессии
- `/help` - показать эту справку

Начните диалог с вопроса или сообщения!
        """
        console.print(Panel(
            Markdown(welcome_text),
            title="📖 Справка",
            border_style="green",
            padding=(1, 2)
        ))

        if not SYSTEM_PROMPT:
            console.print("[yellow]⚠️  Системный промпт не задан. Отредактируйте SYSTEM_PROMPT в src/bot.py[/yellow]\n")
        else:
            console.print("[green]✓ Системный промпт активен[/green]\n")

    def run(self):
        """Запустить основной цикл бота (REPL)."""
        self.show_welcome()

        try:
            while True:
                try:
                    user_input = console.input("[bold cyan]👤 Вы:[/bold cyan] ").strip()
                except EOFError:
                    break

                if not user_input:
                    continue

                if user_input.startswith("/"):
                    command = user_input.lower()

                    if command == "/exit":
                        console.print("[yellow]👋 До свидания![/yellow]")
                        break

                    elif command == "/clear":
                        self.clear_history()
                        continue

                    elif command == "/stats":
                        self.display_stats()
                        continue

                    elif command == "/help":
                        self.show_welcome()
                        continue

                    else:
                        console.print(f"[red]❌ Неизвестная команда: {user_input}[/red]")
                        console.print("[yellow]Используйте /help для справки[/yellow]\n")
                        continue

                console.print(Panel(
                    user_input,
                    title="👤 Вы",
                    border_style="cyan",
                    padding=(1, 2)
                ))

                self.send_message(user_input)

        except KeyboardInterrupt:
            console.print("\n[yellow]👋 Прервано пользователем. До свидания![/yellow]")

        if self.session_metrics["messages_count"] > 0:
            console.print("\n[bold green]📊 Финальная статистика сессии:[/bold green]")
            self.display_stats()


def main():
    """Точка входа в программу."""
    bot = ChatBot()
    bot.run()


if __name__ == "__main__":
    main()
