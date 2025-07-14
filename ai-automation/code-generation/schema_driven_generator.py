#!/usr/bin/env python3
"""
Schema-Driven Code Generation for HFT Systems

This module implements automated code generation based on YAML schemas,
leveraging AI models to generate optimized, production-ready code.
"""

import asyncio
import json
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Any
import yaml

import openai
import anthropic
from jinja2 import Template
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn

console = Console()


@dataclass
class SystemSchema:
    """System definition schema for code generation"""
    name: str
    components: List[str]
    tech_stack: Dict[str, List[str]]
    latency_budget_ns: int
    throughput_target: int
    risk_limits: Dict[str, Any]
    deployment_target: str = "production"
    
    @classmethod
    def from_yaml(cls, yaml_path: str) -> 'SystemSchema':
        """Load schema from YAML file"""
        with open(yaml_path, 'r') as f:
            data = yaml.safe_load(f)
        return cls(**data)


class AICodeGenerator:
    """AI-powered code generation engine"""
    
    def __init__(self):
        self.openai_client = None
        self.anthropic_client = None
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
        except Exception as e:
            console.print(f"[red]Failed to setup AI clients: {e}[/red]")
    
    async def generate_networking_module(self, schema: SystemSchema) -> str:
        """Generate optimized networking module"""
        prompt = f"""
Generate optimized C++ networking module code for HFT system with:

System Requirements:
- Name: {schema.name}
- Latency Budget: {schema.latency_budget_ns} nanoseconds
- Throughput Target: {schema.throughput_target} messages/second
- Tech Stack: {schema.tech_stack.get('networking', [])}

Features Required:
1. Solarflare NIC with Onload API integration
2. DPDK zero-copy packet handling
3. FPGA interface for market data parsing
4. Lock-free ring buffers for message queues
5. CPU affinity and NUMA optimization
6. Timestamp precision to nanoseconds

Code Requirements:
- Use C++17/20 features
- Template-based design for flexibility
- Inline functions for performance
- Memory pre-allocation
- Cache-friendly data structures
- Comprehensive error handling
- Performance monitoring hooks

Please provide:
1. Header file (.hpp)
2. Implementation file (.cpp)
3. Integration instructions
4. Performance tuning guidelines
5. Testing framework
"""
        
        if self.openai_client:
            response = await asyncio.to_thread(
                self.openai_client.ChatCompletion.create,
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are an expert C++ developer specializing in ultra-low latency systems for high-frequency trading."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=4000,
                temperature=0.2
            )
            return response.choices[0].message.content.strip()
        
        return "GPT-4 not available for code generation"
    
    async def generate_market_data_handler(self, schema: SystemSchema) -> str:
        """Generate market data processing module"""
        prompt = f"""
Generate ultra-fast market data handler for HFT system:

System: {schema.name}
Latency Target: < {schema.latency_budget_ns}ns
Protocols: {schema.tech_stack.get('protocols', ['FIX', 'ITCH', 'OUCH'])}

Requirements:
1. Multi-protocol support (FIX, ITCH, OUCH, FAST)
2. Lock-free message parsing
3. Order book reconstruction
4. Real-time market data normalization
5. Configurable symbol filtering
6. Memory pool allocation
7. SIMD optimizations where applicable

Implementation details:
- Template-based message parsers
- Compile-time protocol selection
- Zero-allocation message handling
- Cache-line aligned data structures
- Branch prediction optimization
- Prefetching strategies

Provide complete C++ implementation with:
- Class definitions
- Message parsing logic
- Order book management
- Performance monitoring
- Unit tests
- Benchmarking code
"""
        
        if self.anthropic_client:
            response = await asyncio.to_thread(
                self.anthropic_client.messages.create,
                model="claude-3-opus-20240229",
                max_tokens=4000,
                messages=[
                    {"role": "user", "content": prompt}
                ]
            )
            return response.content[0].text.strip()
        
        return "Claude not available for code generation"
    
    async def generate_risk_engine(self, schema: SystemSchema) -> str:
        """Generate real-time risk management engine"""
        prompt = f"""
Generate ultra-low latency risk management engine:

System: {schema.name}
Risk Limits: {json.dumps(schema.risk_limits, indent=2)}
Latency Budget: < 100ns per risk check

Features:
1. Pre-trade risk checks
2. Real-time position tracking
3. P&L calculation
4. Limit monitoring (position, loss, concentration)
5. Circuit breakers
6. Risk metric computation (VaR, Greeks)
7. Compliance reporting

Technical Requirements:
- Lock-free atomic operations
- SIMD-optimized calculations
- Memory-mapped configuration
- Hot/cold path separation
- Configurable risk parameters
- Real-time alerting
- Audit trail

Provide:
1. Risk engine class hierarchy
2. Position manager
3. Limit checker implementations
4. Configuration system
5. Monitoring interfaces
6. Performance tests
"""
        
        if self.openai_client:
            response = await asyncio.to_thread(
                self.openai_client.ChatCompletion.create,
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are an expert in financial risk management systems and real-time computing."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=4000,
                temperature=0.1
            )
            return response.choices[0].message.content.strip()
        
        return "GPT-4 not available for risk engine generation"
    
    async def generate_order_execution_engine(self, schema: SystemSchema) -> str:
        """Generate order execution and routing engine"""
        prompt = f"""
Generate high-performance order execution engine:

System: {schema.name}
Execution Latency Target: < {schema.latency_budget_ns}ns
Throughput: {schema.throughput_target} orders/second

Core Features:
1. Smart order routing (SOR)
2. Order lifecycle management
3. Fill handling and matching
4. Exchange gateway abstraction
5. Order book management
6. Execution algorithms (TWAP, VWAP, Implementation Shortfall)
7. IOC/FOK/GTC order types

Technical Implementation:
- State machine-based order management
- Lock-free order queues
- Atomic order state updates
- Exchange-specific adapters
- Latency measurement hooks
- Order validation pipeline
- Error handling and recovery

Architecture:
- Modular design with interfaces
- Plugin-based exchange connectors
- Configurable execution strategies
- Real-time monitoring
- Comprehensive logging
- Performance metrics

Deliver complete C++ solution with examples.
"""
        
        if self.anthropic_client:
            response = await asyncio.to_thread(
                self.anthropic_client.messages.create,
                model="claude-3-opus-20240229",
                max_tokens=4000,
                messages=[
                    {"role": "user", "content": prompt}
                ]
            )
            return response.content[0].text.strip()
        
        return "Claude not available for execution engine generation"
    
    async def generate_infrastructure_code(self, schema: SystemSchema) -> Dict[str, str]:
        """Generate complete infrastructure code"""
        console.print(f"[cyan]Generating infrastructure code for {schema.name}...[/cyan]")
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            
            task = progress.add_task("Generating components...", total=4)
            
            # Generate all components concurrently
            networking_task = asyncio.create_task(self.generate_networking_module(schema))
            progress.advance(task)
            
            market_data_task = asyncio.create_task(self.generate_market_data_handler(schema))
            progress.advance(task)
            
            risk_task = asyncio.create_task(self.generate_risk_engine(schema))
            progress.advance(task)
            
            execution_task = asyncio.create_task(self.generate_order_execution_engine(schema))
            progress.advance(task)
            
            # Wait for all tasks to complete
            networking_code = await networking_task
            market_data_code = await market_data_task
            risk_code = await risk_task
            execution_code = await execution_task
        
        return {
            "networking": networking_code,
            "market_data": market_data_code,
            "risk_engine": risk_code,
            "order_execution": execution_code
        }
    
    async def generate_python_analytics(self, schema: SystemSchema) -> str:
        """Generate Python analytics and ML components"""
        prompt = f"""
Generate comprehensive Python analytics suite for HFT system:

System: {schema.name}
Integration: Connect to C++ core via Python bindings
ML Stack: {schema.tech_stack.get('ml', ['scikit-learn', 'xgboost', 'tensorflow'])}

Components needed:
1. Real-time performance analytics
2. Strategy backtesting framework
3. Risk analytics and reporting
4. ML feature engineering pipeline
5. Model training and deployment
6. Real-time prediction service
7. Portfolio optimization
8. Market microstructure analysis

Features:
- Async I/O for real-time data
- Efficient NumPy/Pandas operations
- Machine learning model pipeline
- Real-time visualization (Plotly/Dash)
- RESTful API for model serving
- Configuration management
- Logging and monitoring
- Unit testing framework

Provide:
1. Package structure
2. Core analytics classes
3. ML pipeline implementation
4. API server code
5. Configuration files
6. Requirements.txt
7. Docker setup
8. Testing suite
"""
        
        if self.openai_client:
            response = await asyncio.to_thread(
                self.openai_client.ChatCompletion.create,
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are an expert Python developer specializing in financial analytics and machine learning."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=4000,
                temperature=0.2
            )
            return response.choices[0].message.content.strip()
        
        return "GPT-4 not available for Python analytics generation"


class CodeOrganizer:
    """Organizes and saves generated code"""
    
    def __init__(self, base_path: str = "generated"):
        self.base_path = Path(base_path)
        self.base_path.mkdir(exist_ok=True)
    
    def save_cpp_component(self, component_name: str, code: str, schema: SystemSchema):
        """Save C++ component code"""
        component_dir = self.base_path / "cpp" / component_name
        component_dir.mkdir(parents=True, exist_ok=True)
        
        # Split code into header and implementation if possible
        if "```cpp" in code:
            # Extract code blocks
            parts = code.split("```cpp")
            for i, part in enumerate(parts[1:], 1):
                if "```" in part:
                    code_content = part.split("```")[0].strip()
                    if ".hpp" in part or "header" in part.lower():
                        filename = f"{component_name}.hpp"
                    else:
                        filename = f"{component_name}.cpp"
                    
                    with open(component_dir / filename, 'w') as f:
                        f.write(code_content)
        else:
            # Save as single file
            with open(component_dir / f"{component_name}.cpp", 'w') as f:
                f.write(code)
        
        # Generate CMakeLists.txt
        cmake_content = f"""
cmake_minimum_required(VERSION 3.20)
project({component_name})

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_FLAGS "${{CMAKE_CXX_FLAGS}} -O3 -march=native")

add_library({component_name} {component_name}.cpp)
target_include_directories({component_name} PUBLIC .)
"""
        with open(component_dir / "CMakeLists.txt", 'w') as f:
            f.write(cmake_content.strip())
    
    def save_python_component(self, component_name: str, code: str, schema: SystemSchema):
        """Save Python component code"""
        component_dir = self.base_path / "python" / component_name
        component_dir.mkdir(parents=True, exist_ok=True)
        
        # Extract Python code blocks
        if "```python" in code:
            parts = code.split("```python")
            for i, part in enumerate(parts[1:], 1):
                if "```" in part:
                    code_content = part.split("```")[0].strip()
                    filename = f"{component_name}.py"
                    if i > 1:
                        filename = f"{component_name}_{i}.py"
                    
                    with open(component_dir / filename, 'w') as f:
                        f.write(code_content)
        else:
            with open(component_dir / f"{component_name}.py", 'w') as f:
                f.write(code)
    
    def generate_integration_script(self, schema: SystemSchema):
        """Generate integration and build scripts"""
        script_content = f"""#!/bin/bash
# Auto-generated build script for {schema.name}

set -e

echo "Building {schema.name} HFT System..."

# Build C++ components
cd generated/cpp
for component in */; do
    echo "Building $component"
    cd "$component"
    mkdir -p build
    cd build
    cmake ..
    make -j$(nproc)
    cd ../..
done

cd ../..

# Setup Python environment
cd generated/python
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Run tests
python -m pytest tests/

echo "{schema.name} build completed successfully!"
"""
        
        with open(self.base_path / "build.sh", 'w') as f:
            f.write(script_content.strip())
        
        os.chmod(self.base_path / "build.sh", 0o755)


async def main():
    """Main code generation workflow"""
    console.print("[bold blue]AI-Powered HFT Code Generation Engine[/bold blue]")
    
    # Load system schema
    schema_path = input("Enter path to system schema YAML (or press Enter for example): ").strip()
    if not schema_path:
        # Create example schema
        example_schema = {
            "name": "lightning-hft",
            "components": ["networking", "market_data", "risk_engine", "order_execution"],
            "tech_stack": {
                "languages": ["C++", "Python"],
                "networking": ["DPDK", "Solarflare", "FPGA"],
                "middleware": ["ZeroMQ", "Chronicle Queue"],
                "protocols": ["FIX", "ITCH", "OUCH"],
                "ml": ["scikit-learn", "xgboost", "tensorflow"]
            },
            "latency_budget_ns": 500,
            "throughput_target": 1000000,
            "risk_limits": {
                "max_position_usd": 10000000,
                "max_daily_loss": 1000000,
                "max_leverage": 5.0
            }
        }
        
        schema_path = "example_schema.yaml"
        with open(schema_path, 'w') as f:
            yaml.dump(example_schema, f, default_flow_style=False)
        
        console.print(f"[green]Created example schema: {schema_path}[/green]")
    
    # Load schema
    schema = SystemSchema.from_yaml(schema_path)
    console.print(f"[cyan]Loaded schema for system: {schema.name}[/cyan]")
    
    # Initialize code generator
    generator = AICodeGenerator()
    organizer = CodeOrganizer()
    
    # Generate infrastructure code
    infrastructure_code = await generator.generate_infrastructure_code(schema)
    
    # Generate Python analytics
    python_code = await generator.generate_python_analytics(schema)
    
    # Save all generated code
    console.print("[yellow]Organizing and saving generated code...[/yellow]")
    
    for component, code in infrastructure_code.items():
        organizer.save_cpp_component(component, code, schema)
        console.print(f"[green]✓ Saved C++ component: {component}[/green]")
    
    organizer.save_python_component("analytics", python_code, schema)
    console.print("[green]✓ Saved Python analytics[/green]")
    
    # Generate integration scripts
    organizer.generate_integration_script(schema)
    console.print("[green]✓ Generated build scripts[/green]")
    
    console.print(f"[bold green]Code generation completed! Check the 'generated' directory.[/bold green]")
    console.print(f"[cyan]Run './generated/build.sh' to build the system.[/cyan]")


if __name__ == "__main__":
    asyncio.run(main())
