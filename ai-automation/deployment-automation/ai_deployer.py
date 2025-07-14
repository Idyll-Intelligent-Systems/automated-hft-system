#!/usr/bin/env python3
"""
AI-Powered Deployment Automation for HFT Systems

This module provides real-time automated deployment from structured prompts,
generating infrastructure code, deployment scripts, and monitoring configurations.
"""

import asyncio
import json
import os
import subprocess
import yaml
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime

import openai
import anthropic
from jinja2 import Template
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table

console = Console()


@dataclass
class DeploymentConfig:
    """Deployment configuration from prompts"""
    action: str  # "deploy", "update", "rollback", "scale"
    component: str
    environment: str  # "dev", "staging", "production"
    latency_tolerance_ns: int
    servers: List[str]
    resources: Dict[str, Any]
    security: Dict[str, Any]
    monitoring: Dict[str, Any]
    
    @classmethod
    def from_prompt(cls, prompt_text: str) -> 'DeploymentConfig':
        """Parse deployment config from natural language prompt"""
        # In production, this would use NLP to parse the prompt
        # For now, using a simplified approach
        lines = prompt_text.strip().split('\n')
        config = {
            'action': 'deploy',
            'component': 'trading-engine',
            'environment': 'production',
            'latency_tolerance_ns': 1000,
            'servers': ['ny4-server-1'],
            'resources': {'cpu': 16, 'memory': '64GB'},
            'security': {'tls': True, 'encryption': True},
            'monitoring': {'metrics': True, 'alerts': True}
        }
        
        for line in lines:
            if 'Action:' in line:
                config['action'] = line.split(':')[1].strip().lower()
            elif 'Component:' in line:
                config['component'] = line.split(':')[1].strip()
            elif 'Environment:' in line:
                config['environment'] = line.split(':')[1].strip().lower()
            elif 'LatencyTolerance' in line:
                config['latency_tolerance_ns'] = int(line.split(':')[1].strip().replace('ns', ''))
            elif 'Server:' in line or 'Servers:' in line:
                servers = line.split(':')[1].strip().strip('[]').split(',')
                config['servers'] = [s.strip() for s in servers]
        
        return cls(**config)


class AIInfrastructureGenerator:
    """AI-powered infrastructure code generation"""
    
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
    
    async def generate_dockerfile(self, config: DeploymentConfig) -> str:
        """Generate optimized Dockerfile for HFT component"""
        
        prompt = f"""
Generate optimized Dockerfile for HFT component deployment:

Component: {config.component}
Environment: {config.environment}
Latency Target: < {config.latency_tolerance_ns} nanoseconds
Resources: {json.dumps(config.resources)}

Requirements:
1. Ultra-low latency optimizations
2. Minimal image size
3. Security hardening
4. Runtime performance tuning
5. Monitoring integration
6. Health checks
7. Graceful shutdown handling

Optimizations needed:
- Multi-stage build
- Minimal base image (Alpine or distroless)
- CPU affinity settings
- Memory allocation optimization
- Network stack tuning
- Kernel parameter optimization
- Real-time scheduling
- NUMA awareness

Security features:
- Non-root user
- Minimal attack surface
- Secret management
- Network policies
- Resource limits

Provide complete Dockerfile with comments and best practices.
"""
        
        if self.openai_client:
            response = await asyncio.to_thread(
                self.openai_client.ChatCompletion.create,
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are an expert DevOps engineer specializing in containerized HFT systems."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=2000,
                temperature=0.1
            )
            return response.choices[0].message.content.strip()
        
        return "GPT-4 not available for Dockerfile generation"
    
    async def generate_kubernetes_manifests(self, config: DeploymentConfig) -> str:
        """Generate Kubernetes deployment manifests"""
        
        prompt = f"""
Generate complete Kubernetes manifests for HFT component:

Component: {config.component}
Environment: {config.environment}
Servers: {config.servers}
Latency Requirements: < {config.latency_tolerance_ns}ns
Resources: {json.dumps(config.resources)}

Manifests needed:
1. Deployment with ultra-low latency optimizations
2. Service for internal communication
3. ConfigMap for configuration
4. Secret for sensitive data
5. NetworkPolicy for security
6. PodDisruptionBudget for availability
7. HorizontalPodAutoscaler (if applicable)
8. ServiceMonitor for Prometheus

Low-latency optimizations:
- Guaranteed QoS class
- CPU and memory requests/limits
- Node affinity for specific hardware
- Pod anti-affinity for distribution
- Huge pages support
- SR-IOV networking
- Real-time scheduling class
- Dedicated CPU cores

Security considerations:
- Security contexts
- Network policies
- RBAC configuration
- Pod security policies
- Admission controllers

Monitoring integration:
- Prometheus metrics
- Health check endpoints
- Readiness/liveness probes
- Log aggregation labels

Provide YAML manifests with detailed annotations.
"""
        
        if self.anthropic_client:
            response = await asyncio.to_thread(
                self.anthropic_client.messages.create,
                model="claude-3-opus-20240229",
                max_tokens=3000,
                messages=[
                    {"role": "user", "content": prompt}
                ]
            )
            return response.content[0].text.strip()
        
        return "Claude not available for Kubernetes manifest generation"
    
    async def generate_terraform_infrastructure(self, config: DeploymentConfig) -> str:
        """Generate Terraform infrastructure code"""
        
        prompt = f"""
Generate Terraform infrastructure for HFT deployment:

Environment: {config.environment}
Servers: {config.servers}
Component: {config.component}
Latency Target: {config.latency_tolerance_ns}ns

Infrastructure components:
1. VPC with optimized networking
2. Subnets in multiple AZs
3. Security groups with minimal access
4. EC2 instances with SR-IOV
5. Dedicated hosts for isolation
6. EBS volumes with provisioned IOPS
7. Elastic Network Interfaces
8. Route tables and gateways
9. NAT gateways for outbound traffic
10. VPC endpoints for AWS services

Performance optimizations:
- Enhanced networking (SR-IOV)
- Placement groups (cluster)
- Dedicated tenancy
- Optimized instance types (C5n, M5n)
- High-performance storage
- Low-latency networking
- CPU optimization features

Security features:
- Network ACLs
- Security groups
- VPC Flow Logs
- GuardDuty integration
- CloudTrail logging
- KMS encryption
- IAM roles and policies

Monitoring and logging:
- CloudWatch metrics
- CloudWatch Logs
- X-Ray tracing
- VPC Flow Logs
- Load balancer logs

Provide complete Terraform configuration with modules.
"""
        
        if self.openai_client:
            response = await asyncio.to_thread(
                self.openai_client.ChatCompletion.create,
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are an expert cloud architect specializing in high-performance trading infrastructure."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=3000,
                temperature=0.1
            )
            return response.choices[0].message.content.strip()
        
        return "GPT-4 not available for Terraform generation"
    
    async def generate_ansible_playbooks(self, config: DeploymentConfig) -> str:
        """Generate Ansible configuration playbooks"""
        
        prompt = f"""
Generate Ansible playbooks for HFT system configuration:

Component: {config.component}
Environment: {config.environment}
Servers: {config.servers}
Latency Target: {config.latency_tolerance_ns}ns

Playbooks needed:
1. System optimization playbook
2. Application deployment playbook
3. Monitoring setup playbook
4. Security hardening playbook
5. Backup configuration playbook

System optimizations:
- Kernel parameter tuning
- CPU governor settings
- Memory management
- Network stack optimization
- Disk I/O scheduling
- Interrupt handling
- NUMA configuration
- Huge pages setup
- Real-time kernel setup
- CPU isolation and affinity

Application deployment:
- Service installation
- Configuration management
- Environment variables
- Log rotation setup
- Health check configuration
- Graceful restart procedures
- Performance monitoring
- Error handling

Security hardening:
- Firewall configuration
- SSH hardening
- User management
- Sudo configuration
- File permissions
- Audit logging
- Intrusion detection
- Certificate management

Monitoring setup:
- Prometheus node exporter
- Log forwarding
- Metric collection
- Alert configuration
- Dashboard setup

Provide complete Ansible playbooks with error handling.
"""
        
        if self.anthropic_client:
            response = await asyncio.to_thread(
                self.anthropic_client.messages.create,
                model="claude-3-opus-20240229",
                max_tokens=3000,
                messages=[
                    {"role": "user", "content": prompt}
                ]
            )
            return response.content[0].text.strip()
        
        return "Claude not available for Ansible playbook generation"


class DeploymentOrchestrator:
    """Orchestrates the complete deployment process"""
    
    def __init__(self, base_path: str = "deployment-generated"):
        self.base_path = Path(base_path)
        self.base_path.mkdir(exist_ok=True)
        self.generator = AIInfrastructureGenerator()
    
    async def execute_deployment(self, config: DeploymentConfig) -> Dict[str, Any]:
        """Execute complete deployment workflow"""
        
        console.print(f"[cyan]Starting deployment for {config.component}...[/cyan]")
        
        deployment_id = f"{config.component}-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
        deployment_dir = self.base_path / deployment_id
        deployment_dir.mkdir(exist_ok=True)
        
        results = {}
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            
            # Generate infrastructure code
            task = progress.add_task("Generating infrastructure code...", total=4)
            
            # Generate Dockerfile
            dockerfile = await self.generator.generate_dockerfile(config)
            self._save_file(deployment_dir / "Dockerfile", dockerfile)
            results['dockerfile'] = 'generated'
            progress.advance(task)
            
            # Generate Kubernetes manifests
            k8s_manifests = await self.generator.generate_kubernetes_manifests(config)
            self._save_file(deployment_dir / "k8s-manifests.yaml", k8s_manifests)
            results['kubernetes'] = 'generated'
            progress.advance(task)
            
            # Generate Terraform infrastructure
            terraform_code = await self.generator.generate_terraform_infrastructure(config)
            self._save_file(deployment_dir / "main.tf", terraform_code)
            results['terraform'] = 'generated'
            progress.advance(task)
            
            # Generate Ansible playbooks
            ansible_playbooks = await self.generator.generate_ansible_playbooks(config)
            self._save_file(deployment_dir / "site.yml", ansible_playbooks)
            results['ansible'] = 'generated'
            progress.advance(task)
        
        # Generate deployment script
        deploy_script = self._generate_deployment_script(config, deployment_id)
        self._save_file(deployment_dir / "deploy.sh", deploy_script)
        os.chmod(deployment_dir / "deploy.sh", 0o755)
        
        # Generate monitoring configuration
        monitoring_config = self._generate_monitoring_config(config)
        self._save_file(deployment_dir / "monitoring.yaml", monitoring_config)
        
        results.update({
            'deployment_id': deployment_id,
            'deployment_dir': str(deployment_dir),
            'status': 'ready',
            'timestamp': datetime.now().isoformat()
        })
        
        console.print(f"[green]Deployment package generated: {deployment_dir}[/green]")
        return results
    
    def _save_file(self, filepath: Path, content: str):
        """Save content to file"""
        with open(filepath, 'w') as f:
            f.write(content)
    
    def _generate_deployment_script(self, config: DeploymentConfig, deployment_id: str) -> str:
        """Generate deployment execution script"""
        
        script = f"""#!/bin/bash
# Auto-generated deployment script for {config.component}
# Deployment ID: {deployment_id}

set -e

COMPONENT="{config.component}"
ENVIRONMENT="{config.environment}"
DEPLOYMENT_ID="{deployment_id}"

echo "Starting deployment of $COMPONENT to $ENVIRONMENT..."

# Pre-deployment checks
echo "Running pre-deployment checks..."

# Check system resources
if ! command -v docker &> /dev/null; then
    echo "Docker is required but not installed"
    exit 1
fi

if ! command -v kubectl &> /dev/null; then
    echo "kubectl is required but not installed"
    exit 1
fi

# Build Docker image
echo "Building Docker image..."
docker build -t $COMPONENT:$DEPLOYMENT_ID .

# Apply Kubernetes manifests
echo "Deploying to Kubernetes..."
kubectl apply -f k8s-manifests.yaml

# Wait for deployment
echo "Waiting for deployment to be ready..."
kubectl rollout status deployment/$COMPONENT

# Run health checks
echo "Running health checks..."
kubectl get pods -l app=$COMPONENT

# Configure monitoring
echo "Setting up monitoring..."
kubectl apply -f monitoring.yaml

echo "Deployment completed successfully!"
echo "Deployment ID: $DEPLOYMENT_ID"
"""
        
        return script.strip()
    
    def _generate_monitoring_config(self, config: DeploymentConfig) -> str:
        """Generate monitoring configuration"""
        
        monitoring_yaml = f"""
apiVersion: v1
kind: ServiceMonitor
metadata:
  name: {config.component}-monitor
  labels:
    app: {config.component}
spec:
  selector:
    matchLabels:
      app: {config.component}
  endpoints:
  - port: metrics
    interval: 1s
    path: /metrics
---
apiVersion: monitoring.coreos.com/v1
kind: PrometheusRule
metadata:
  name: {config.component}-alerts
spec:
  groups:
  - name: {config.component}
    rules:
    - alert: HighLatency
      expr: trading_latency_p99 > {config.latency_tolerance_ns}
      for: 5s
      labels:
        severity: critical
      annotations:
        summary: "High latency detected"
        description: "99th percentile latency is above threshold"
    
    - alert: HighErrorRate
      expr: rate(trading_errors_total[1m]) > 0.01
      for: 30s
      labels:
        severity: warning
      annotations:
        summary: "High error rate detected"
        description: "Error rate is above 1%"
"""
        
        return monitoring_yaml.strip()


async def interactive_deployment():
    """Interactive deployment interface"""
    
    console.print("[bold blue]AI-Powered HFT Deployment Engine[/bold blue]")
    console.print("[italic]Enter deployment prompts or type 'quit' to exit[/italic]")
    
    orchestrator = DeploymentOrchestrator()
    
    while True:
        try:
            prompt_text = console.input("\n[bold green]Enter deployment prompt: [/bold green]")
            
            if prompt_text.lower() == 'quit':
                break
            
            if not prompt_text.strip():
                # Use example prompt
                prompt_text = """
Action: Deploy
Component: ExecutionEngine
Environment: Production
LatencyTolerance: 300ns
Servers: [NY4-Colo-3, NY4-Colo-4]
"""
                console.print(f"[yellow]Using example prompt:[/yellow]\n{prompt_text}")
            
            # Parse deployment configuration
            config = DeploymentConfig.from_prompt(prompt_text)
            
            # Display parsed configuration
            table = Table(title="Deployment Configuration")
            table.add_column("Parameter", style="cyan")
            table.add_column("Value", style="green")
            
            table.add_row("Action", config.action)
            table.add_row("Component", config.component)
            table.add_row("Environment", config.environment)
            table.add_row("Latency Tolerance", f"{config.latency_tolerance_ns}ns")
            table.add_row("Servers", ", ".join(config.servers))
            
            console.print(table)
            
            # Confirm deployment
            confirm = console.input("[yellow]Proceed with deployment? (y/n): [/yellow]")
            if confirm.lower() != 'y':
                continue
            
            # Execute deployment
            results = await orchestrator.execute_deployment(config)
            
            # Display results
            console.print(f"[green]✓ Deployment package created: {results['deployment_id']}[/green]")
            console.print(f"[cyan]Location: {results['deployment_dir']}[/cyan]")
            console.print(f"[yellow]Run: cd {results['deployment_dir']} && ./deploy.sh[/yellow]")
            
        except KeyboardInterrupt:
            break
        except Exception as e:
            console.print(f"[red]Error: {e}[/red]")


async def batch_deployment():
    """Batch deployment from configuration file"""
    
    config_file = input("Enter path to deployment configuration file: ").strip()
    
    if not config_file or not os.path.exists(config_file):
        console.print("[red]Configuration file not found[/red]")
        return
    
    with open(config_file, 'r') as f:
        deployment_configs = yaml.safe_load(f)
    
    orchestrator = DeploymentOrchestrator()
    
    for config_data in deployment_configs.get('deployments', []):
        config = DeploymentConfig(**config_data)
        console.print(f"[cyan]Processing deployment: {config.component}[/cyan]")
        
        results = await orchestrator.execute_deployment(config)
        console.print(f"[green]✓ Completed: {results['deployment_id']}[/green]")


if __name__ == "__main__":
    console.print("[bold blue]Choose deployment mode:[/bold blue]")
    console.print("1. Interactive deployment")
    console.print("2. Batch deployment")
    
    choice = input("Enter choice (1 or 2): ").strip()
    
    if choice == "1":
        asyncio.run(interactive_deployment())
    elif choice == "2":
        asyncio.run(batch_deployment())
    else:
        console.print("[red]Invalid choice[/red]")
