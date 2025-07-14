#!/usr/bin/env python3
"""
AI-Powered Knowledge Acquisition Engine for HFT Systems

This module provides rapid learning capabilities by leveraging multiple AI models
to acquire knowledge about HFT concepts, technologies, and implementation details.
"""

import asyncio
import json
import logging
import os
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Any
import yaml

import openai
import anthropic
import google.generativeai as genai
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

console = Console()


class AIModel(Enum):
    """Supported AI models for knowledge acquisition"""
    GPT4 = "gpt-4"
    CLAUDE3 = "claude-3-opus-20240229"
    GEMINI = "gemini-pro"


@dataclass
class LearningPrompt:
    """Structured learning prompt for AI models"""
    topic: str
    depth: str  # "Basic", "Intermediate", "Advanced"
    length: str  # "Brief", "Detailed", "Comprehensive"
    format: str  # "Explanation", "Code", "Examples", "Tutorial"
    context: str = "HFT Trading Systems"
    
    def to_prompt(self) -> str:
        """Convert to formatted prompt string"""
        return f"""
Topic: {self.topic}
Context: {self.context}
Depth: {self.depth}
Length: {self.length}
Format: {self.format}

Please provide a {self.depth.lower()}, {self.length.lower()} {self.format.lower()} 
about {self.topic} in the context of {self.context}.

Requirements:
- Be precise and technical
- Include practical examples
- Focus on implementation details
- Optimize for ultra-low latency where applicable
"""


class AIKnowledgeEngine:
    """AI-powered knowledge acquisition engine"""
    
    def __init__(self):
        self.openai_client = None
        self.anthropic_client = None
        self.gemini_model = None
        self._setup_clients()
        
    def _setup_clients(self):
        """Setup AI model clients"""
        try:
            if os.getenv("OPENAI_API_KEY"):
                openai.api_key = os.getenv("OPENAI_API_KEY")
                self.openai_client = openai
                
            if os.getenv("ANTHROPIC_API_KEY"):
                self.anthropic_client = anthropic.Anthropic(
                    api_key=os.getenv("ANTHROPIC_API_KEY")
                )
                
            if os.getenv("GOOGLE_API_KEY"):
                genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
                self.gemini_model = genai.GenerativeModel('gemini-pro')
                
        except Exception as e:
            logger.warning(f"Failed to setup some AI clients: {e}")
    
    async def query_gpt4(self, prompt: str) -> str:
        """Query GPT-4 for knowledge"""
        if not self.openai_client:
            return "GPT-4 not available"
            
        try:
            response = await asyncio.to_thread(
                self.openai_client.ChatCompletion.create,
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are an expert in high-frequency trading systems and financial technology."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=2000,
                temperature=0.3
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            logger.error(f"GPT-4 query failed: {e}")
            return f"Error: {e}"
    
    async def query_claude(self, prompt: str) -> str:
        """Query Claude for knowledge"""
        if not self.anthropic_client:
            return "Claude not available"
            
        try:
            response = await asyncio.to_thread(
                self.anthropic_client.messages.create,
                model="claude-3-opus-20240229",
                max_tokens=2000,
                messages=[
                    {"role": "user", "content": prompt}
                ]
            )
            return response.content[0].text.strip()
        except Exception as e:
            logger.error(f"Claude query failed: {e}")
            return f"Error: {e}"
    
    async def query_gemini(self, prompt: str) -> str:
        """Query Gemini for knowledge"""
        if not self.gemini_model:
            return "Gemini not available"
            
        try:
            response = await asyncio.to_thread(
                self.gemini_model.generate_content,
                prompt
            )
            return response.text.strip()
        except Exception as e:
            logger.error(f"Gemini query failed: {e}")
            return f"Error: {e}"
    
    async def multi_model_query(self, prompt: LearningPrompt) -> Dict[str, str]:
        """Query multiple AI models for comprehensive knowledge"""
        prompt_text = prompt.to_prompt()
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            task = progress.add_task(f"Querying AI models for: {prompt.topic}", total=3)
            
            # Query all models concurrently
            tasks = []
            if self.openai_client:
                tasks.append(("GPT-4", self.query_gpt4(prompt_text)))
            if self.anthropic_client:
                tasks.append(("Claude", self.query_claude(prompt_text)))
            if self.gemini_model:
                tasks.append(("Gemini", self.query_gemini(prompt_text)))
            
            results = {}
            for name, task_coro in tasks:
                result = await task_coro
                results[name] = result
                progress.advance(task)
        
        return results
    
    def format_results(self, results: Dict[str, str], topic: str) -> str:
        """Format results from multiple AI models"""
        output = []
        output.append(f"# Knowledge Acquisition: {topic}")
        output.append("=" * 50)
        output.append("")
        
        for model, response in results.items():
            output.append(f"## {model} Response")
            output.append("-" * 30)
            output.append(response)
            output.append("")
        
        return "\n".join(output)
    
    async def batch_learning(self, topics: List[LearningPrompt]) -> Dict[str, Dict[str, str]]:
        """Perform batch learning on multiple topics"""
        results = {}
        
        console.print(f"[bold green]Starting batch learning for {len(topics)} topics...[/bold green]")
        
        for prompt in topics:
            console.print(f"[cyan]Learning: {prompt.topic}[/cyan]")
            topic_results = await self.multi_model_query(prompt)
            results[prompt.topic] = topic_results
            
            # Save individual topic results
            filename = f"knowledge_{prompt.topic.lower().replace(' ', '_')}.md"
            with open(filename, 'w') as f:
                f.write(self.format_results(topic_results, prompt.topic))
        
        return results


def load_learning_curriculum() -> List[LearningPrompt]:
    """Load predefined learning curriculum for HFT systems"""
    curriculum = [
        LearningPrompt(
            topic="FPGA latency optimization",
            depth="Intermediate",
            length="Brief",
            format="Examples"
        ),
        LearningPrompt(
            topic="DPDK zero-copy packet processing",
            depth="Advanced",
            length="Detailed",
            format="Code"
        ),
        LearningPrompt(
            topic="Lock-free data structures for HFT",
            depth="Advanced",
            length="Comprehensive",
            format="Tutorial"
        ),
        LearningPrompt(
            topic="FIX protocol optimization",
            depth="Intermediate",
            length="Detailed",
            format="Implementation"
        ),
        LearningPrompt(
            topic="Market microstructure for HFT",
            depth="Advanced",
            length="Comprehensive",
            format="Analysis"
        ),
        LearningPrompt(
            topic="Risk management in microsecond timeframes",
            depth="Advanced",
            length="Detailed",
            format="Framework"
        ),
        LearningPrompt(
            topic="CPU affinity and NUMA optimization",
            depth="Intermediate",
            length="Brief",
            format="Configuration"
        ),
        LearningPrompt(
            topic="Avellaneda-Stoikov market making model",
            depth="Advanced",
            length="Comprehensive",
            format="Mathematical"
        )
    ]
    return curriculum


async def interactive_learning():
    """Interactive learning session"""
    engine = AIKnowledgeEngine()
    
    console.print("[bold blue]HFT Knowledge Acquisition Engine[/bold blue]")
    console.print("[italic]Type 'quit' to exit, 'curriculum' for batch learning[/italic]")
    
    while True:
        try:
            topic = console.input("\n[bold green]Enter topic to learn about: [/bold green]")
            
            if topic.lower() == 'quit':
                break
            elif topic.lower() == 'curriculum':
                curriculum = load_learning_curriculum()
                results = await engine.batch_learning(curriculum)
                console.print("[bold green]Batch learning completed![/bold green]")
                continue
            
            # Create learning prompt
            depth = console.input("[yellow]Depth (Basic/Intermediate/Advanced): [/yellow]") or "Intermediate"
            length = console.input("[yellow]Length (Brief/Detailed/Comprehensive): [/yellow]") or "Brief"
            format_type = console.input("[yellow]Format (Explanation/Code/Examples/Tutorial): [/yellow]") or "Explanation"
            
            prompt = LearningPrompt(
                topic=topic,
                depth=depth,
                length=length,
                format=format_type
            )
            
            # Query AI models
            results = await engine.multi_model_query(prompt)
            
            # Display results
            table = Table(title=f"Knowledge: {topic}")
            table.add_column("AI Model", style="cyan", no_wrap=True)
            table.add_column("Response", style="white")
            
            for model, response in results.items():
                # Truncate long responses for display
                display_response = response[:200] + "..." if len(response) > 200 else response
                table.add_row(model, display_response)
            
            console.print(table)
            
            # Save detailed results
            filename = f"knowledge_{topic.lower().replace(' ', '_')}.md"
            with open(filename, 'w') as f:
                f.write(engine.format_results(results, topic))
            
            console.print(f"[green]Detailed results saved to {filename}[/green]")
            
        except KeyboardInterrupt:
            break
        except Exception as e:
            console.print(f"[red]Error: {e}[/red]")


if __name__ == "__main__":
    asyncio.run(interactive_learning())
