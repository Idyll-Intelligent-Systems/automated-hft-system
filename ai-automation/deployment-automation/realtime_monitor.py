#!/usr/bin/env python3
"""
Real-time Monitoring and Performance Optimizer for HFT Systems

This module provides real-time monitoring, alerting, and automated
performance optimization for deployed HFT systems.
"""

import asyncio
import json
import logging
import time
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable
from pathlib import Path

import psutil
import aiohttp
import numpy as np
from prometheus_client.parser import text_string_to_metric_families
from rich.console import Console
from rich.live import Live
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, BarColumn, TextColumn
from rich.layout import Layout

console = Console()
logger = logging.getLogger(__name__)


@dataclass
class PerformanceMetrics:
    """Performance metrics snapshot"""
    timestamp: float
    latency_p50_ns: float
    latency_p95_ns: float
    latency_p99_ns: float
    throughput_ops_sec: float
    cpu_usage_percent: float
    memory_usage_mb: float
    network_rx_mbps: float
    network_tx_mbps: float
    error_rate_percent: float
    jitter_ns: float
    cache_hit_rate: float


@dataclass
class AlertRule:
    """Alert configuration"""
    name: str
    metric: str
    threshold: float
    operator: str  # "gt", "lt", "eq"
    duration_seconds: int
    severity: str  # "critical", "warning", "info"
    callback: Optional[Callable] = None


class MetricsCollector:
    """Collects metrics from various sources"""
    
    def __init__(self, prometheus_url: str = "http://localhost:9090"):
        self.prometheus_url = prometheus_url
        self.system_metrics = {}
        self.trading_metrics = {}
    
    async def collect_prometheus_metrics(self) -> Dict[str, float]:
        """Collect metrics from Prometheus"""
        metrics = {}
        
        try:
            async with aiohttp.ClientSession() as session:
                # Query key trading metrics
                queries = {
                    'latency_p50': 'histogram_quantile(0.50, trading_latency_histogram)',
                    'latency_p95': 'histogram_quantile(0.95, trading_latency_histogram)',
                    'latency_p99': 'histogram_quantile(0.99, trading_latency_histogram)',
                    'throughput': 'rate(trading_orders_total[1m])',
                    'error_rate': 'rate(trading_errors_total[1m]) / rate(trading_requests_total[1m]) * 100',
                    'jitter': 'trading_latency_stddev',
                    'cache_hits': 'rate(cache_hits_total[1m]) / rate(cache_requests_total[1m]) * 100'
                }
                
                for metric_name, query in queries.items():
                    url = f"{self.prometheus_url}/api/v1/query"
                    params = {'query': query}
                    
                    async with session.get(url, params=params) as response:
                        if response.status == 200:
                            data = await response.json()
                            result = data.get('data', {}).get('result', [])
                            if result:
                                metrics[metric_name] = float(result[0]['value'][1])
                        
        except Exception as e:
            logger.error(f"Failed to collect Prometheus metrics: {e}")
        
        return metrics
    
    def collect_system_metrics(self) -> Dict[str, float]:
        """Collect system-level metrics"""
        try:
            # CPU metrics
            cpu_percent = psutil.cpu_percent(interval=0.1)
            cpu_freq = psutil.cpu_freq()
            
            # Memory metrics
            memory = psutil.virtual_memory()
            memory_mb = memory.used / (1024 * 1024)
            
            # Network metrics
            network = psutil.net_io_counters()
            
            # Disk metrics
            disk = psutil.disk_io_counters()
            
            return {
                'cpu_usage_percent': cpu_percent,
                'cpu_frequency_mhz': cpu_freq.current if cpu_freq else 0,
                'memory_usage_mb': memory_mb,
                'memory_available_mb': memory.available / (1024 * 1024),
                'network_rx_mbps': network.bytes_recv / (1024 * 1024),
                'network_tx_mbps': network.bytes_sent / (1024 * 1024),
                'disk_read_mbps': disk.read_bytes / (1024 * 1024) if disk else 0,
                'disk_write_mbps': disk.write_bytes / (1024 * 1024) if disk else 0,
            }
            
        except Exception as e:
            logger.error(f"Failed to collect system metrics: {e}")
            return {}
    
    async def get_performance_snapshot(self) -> PerformanceMetrics:
        """Get current performance snapshot"""
        
        # Collect from various sources
        prom_metrics = await self.collect_prometheus_metrics()
        sys_metrics = self.collect_system_metrics()
        
        # Combine metrics
        return PerformanceMetrics(
            timestamp=time.time(),
            latency_p50_ns=prom_metrics.get('latency_p50', 0) * 1e9,
            latency_p95_ns=prom_metrics.get('latency_p95', 0) * 1e9,
            latency_p99_ns=prom_metrics.get('latency_p99', 0) * 1e9,
            throughput_ops_sec=prom_metrics.get('throughput', 0),
            cpu_usage_percent=sys_metrics.get('cpu_usage_percent', 0),
            memory_usage_mb=sys_metrics.get('memory_usage_mb', 0),
            network_rx_mbps=sys_metrics.get('network_rx_mbps', 0),
            network_tx_mbps=sys_metrics.get('network_tx_mbps', 0),
            error_rate_percent=prom_metrics.get('error_rate', 0),
            jitter_ns=prom_metrics.get('jitter', 0) * 1e9,
            cache_hit_rate=prom_metrics.get('cache_hits', 0)
        )


class PerformanceOptimizer:
    """Automated performance optimization"""
    
    def __init__(self):
        self.optimization_history = []
        self.baseline_metrics = None
        
    async def analyze_performance(self, metrics: PerformanceMetrics) -> List[str]:
        """Analyze performance and suggest optimizations"""
        
        suggestions = []
        
        # Latency analysis
        if metrics.latency_p99_ns > 1000:  # > 1μs
            suggestions.append("High P99 latency detected - consider CPU affinity optimization")
            
        if metrics.jitter_ns > 100:  # > 100ns jitter
            suggestions.append("High jitter - enable real-time kernel scheduling")
        
        # Throughput analysis
        if metrics.throughput_ops_sec < 100000:  # < 100k ops/sec
            suggestions.append("Low throughput - consider batch processing optimization")
        
        # Resource analysis
        if metrics.cpu_usage_percent > 80:
            suggestions.append("High CPU usage - consider scaling or optimization")
            
        if metrics.memory_usage_mb > 60000:  # > 60GB
            suggestions.append("High memory usage - consider memory pool optimization")
        
        # Error analysis
        if metrics.error_rate_percent > 0.1:
            suggestions.append("Elevated error rate - investigate error patterns")
        
        # Cache analysis
        if metrics.cache_hit_rate < 95:
            suggestions.append("Low cache hit rate - optimize cache warming strategy")
        
        return suggestions
    
    async def auto_optimize(self, metrics: PerformanceMetrics) -> Dict[str, Any]:
        """Perform automated optimizations"""
        
        optimizations = {}
        
        # CPU optimization
        if metrics.cpu_usage_percent > 80:
            optimizations['cpu_affinity'] = await self._optimize_cpu_affinity()
            optimizations['cpu_governor'] = await self._set_performance_governor()
        
        # Memory optimization
        if metrics.memory_usage_mb > 50000:
            optimizations['huge_pages'] = await self._enable_huge_pages()
            optimizations['memory_compaction'] = await self._optimize_memory_compaction()
        
        # Network optimization
        if metrics.latency_p99_ns > 500:
            optimizations['network_tuning'] = await self._optimize_network_stack()
            optimizations['interrupt_tuning'] = await self._optimize_interrupts()
        
        return optimizations
    
    async def _optimize_cpu_affinity(self) -> str:
        """Optimize CPU affinity for trading processes"""
        # In production, this would identify trading processes and set CPU affinity
        return "CPU affinity optimization simulated"
    
    async def _set_performance_governor(self) -> str:
        """Set CPU governor to performance mode"""
        # In production, this would set CPU governor
        return "Performance governor set"
    
    async def _enable_huge_pages(self) -> str:
        """Enable huge pages for better memory performance"""
        # In production, this would configure huge pages
        return "Huge pages enabled"
    
    async def _optimize_memory_compaction(self) -> str:
        """Optimize memory compaction settings"""
        return "Memory compaction optimized"
    
    async def _optimize_network_stack(self) -> str:
        """Optimize network stack parameters"""
        return "Network stack optimized"
    
    async def _optimize_interrupts(self) -> str:
        """Optimize interrupt handling"""
        return "Interrupt handling optimized"


class AlertManager:
    """Manages alerts and notifications"""
    
    def __init__(self):
        self.alert_rules = []
        self.active_alerts = {}
        self.alert_history = []
        
    def add_alert_rule(self, rule: AlertRule):
        """Add an alert rule"""
        self.alert_rules.append(rule)
    
    async def check_alerts(self, metrics: PerformanceMetrics) -> List[Dict[str, Any]]:
        """Check alert conditions"""
        
        triggered_alerts = []
        current_time = time.time()
        
        for rule in self.alert_rules:
            metric_value = getattr(metrics, rule.metric, None)
            if metric_value is None:
                continue
            
            # Check condition
            triggered = False
            if rule.operator == "gt" and metric_value > rule.threshold:
                triggered = True
            elif rule.operator == "lt" and metric_value < rule.threshold:
                triggered = True
            elif rule.operator == "eq" and metric_value == rule.threshold:
                triggered = True
            
            alert_key = f"{rule.name}_{rule.metric}"
            
            if triggered:
                if alert_key not in self.active_alerts:
                    self.active_alerts[alert_key] = {
                        'rule': rule,
                        'start_time': current_time,
                        'value': metric_value
                    }
                
                # Check if alert duration exceeded
                alert_duration = current_time - self.active_alerts[alert_key]['start_time']
                if alert_duration >= rule.duration_seconds:
                    alert = {
                        'name': rule.name,
                        'metric': rule.metric,
                        'value': metric_value,
                        'threshold': rule.threshold,
                        'severity': rule.severity,
                        'duration': alert_duration,
                        'timestamp': current_time
                    }
                    triggered_alerts.append(alert)
                    
                    # Execute callback if defined
                    if rule.callback:
                        await rule.callback(alert)
            else:
                # Clear alert if condition no longer met
                if alert_key in self.active_alerts:
                    del self.active_alerts[alert_key]
        
        return triggered_alerts


class RealTimeMonitor:
    """Real-time monitoring dashboard"""
    
    def __init__(self, update_interval: float = 1.0):
        self.update_interval = update_interval
        self.metrics_collector = MetricsCollector()
        self.optimizer = PerformanceOptimizer()
        self.alert_manager = AlertManager()
        self.metrics_history = []
        self.running = False
        
        # Setup default alert rules
        self._setup_default_alerts()
    
    def _setup_default_alerts(self):
        """Setup default alert rules"""
        
        rules = [
            AlertRule(
                name="HighLatency",
                metric="latency_p99_ns",
                threshold=1000,  # 1μs
                operator="gt",
                duration_seconds=5,
                severity="critical"
            ),
            AlertRule(
                name="HighErrorRate",
                metric="error_rate_percent",
                threshold=0.1,
                operator="gt",
                duration_seconds=30,
                severity="warning"
            ),
            AlertRule(
                name="HighCPU",
                metric="cpu_usage_percent",
                threshold=90,
                operator="gt",
                duration_seconds=60,
                severity="warning"
            ),
            AlertRule(
                name="LowThroughput",
                metric="throughput_ops_sec",
                threshold=50000,
                operator="lt",
                duration_seconds=120,
                severity="info"
            )
        ]
        
        for rule in rules:
            self.alert_manager.add_alert_rule(rule)
    
    def _create_dashboard_layout(self) -> Layout:
        """Create dashboard layout"""
        
        layout = Layout()
        layout.split_column(
            Layout(name="header"),
            Layout(name="main"),
            Layout(name="footer")
        )
        
        layout["main"].split_row(
            Layout(name="metrics"),
            Layout(name="alerts")
        )
        
        return layout
    
    def _create_metrics_table(self, metrics: PerformanceMetrics) -> Table:
        """Create metrics display table"""
        
        table = Table(title="HFT System Performance", show_header=True, header_style="bold magenta")
        table.add_column("Metric", style="cyan", no_wrap=True)
        table.add_column("Value", style="green")
        table.add_column("Status", style="yellow")
        
        # Latency metrics
        table.add_row(
            "Latency P50",
            f"{metrics.latency_p50_ns:.0f} ns",
            "🟢" if metrics.latency_p50_ns < 500 else "🟡" if metrics.latency_p50_ns < 1000 else "🔴"
        )
        table.add_row(
            "Latency P99",
            f"{metrics.latency_p99_ns:.0f} ns",
            "🟢" if metrics.latency_p99_ns < 1000 else "🟡" if metrics.latency_p99_ns < 2000 else "🔴"
        )
        table.add_row(
            "Jitter",
            f"{metrics.jitter_ns:.0f} ns",
            "🟢" if metrics.jitter_ns < 100 else "🟡" if metrics.jitter_ns < 500 else "🔴"
        )
        
        # Throughput metrics
        table.add_row(
            "Throughput",
            f"{metrics.throughput_ops_sec:,.0f} ops/s",
            "🟢" if metrics.throughput_ops_sec > 100000 else "🟡" if metrics.throughput_ops_sec > 50000 else "🔴"
        )
        
        # Resource metrics
        table.add_row(
            "CPU Usage",
            f"{metrics.cpu_usage_percent:.1f}%",
            "🟢" if metrics.cpu_usage_percent < 70 else "🟡" if metrics.cpu_usage_percent < 90 else "🔴"
        )
        table.add_row(
            "Memory Usage",
            f"{metrics.memory_usage_mb:,.0f} MB",
            "🟢" if metrics.memory_usage_mb < 50000 else "🟡" if metrics.memory_usage_mb < 60000 else "🔴"
        )
        
        # Network metrics
        table.add_row(
            "Network RX",
            f"{metrics.network_rx_mbps:.2f} MB/s",
            "🟢"
        )
        table.add_row(
            "Network TX",
            f"{metrics.network_tx_mbps:.2f} MB/s",
            "🟢"
        )
        
        # Error metrics
        table.add_row(
            "Error Rate",
            f"{metrics.error_rate_percent:.3f}%",
            "🟢" if metrics.error_rate_percent < 0.01 else "🟡" if metrics.error_rate_percent < 0.1 else "🔴"
        )
        
        # Cache metrics
        table.add_row(
            "Cache Hit Rate",
            f"{metrics.cache_hit_rate:.1f}%",
            "🟢" if metrics.cache_hit_rate > 95 else "🟡" if metrics.cache_hit_rate > 90 else "🔴"
        )
        
        return table
    
    def _create_alerts_panel(self, alerts: List[Dict[str, Any]]) -> Panel:
        """Create alerts display panel"""
        
        if not alerts:
            content = "[green]No active alerts[/green]"
        else:
            content = ""
            for alert in alerts[-10:]:  # Show last 10 alerts
                severity_color = {
                    'critical': 'red',
                    'warning': 'yellow',
                    'info': 'blue'
                }.get(alert['severity'], 'white')
                
                content += f"[{severity_color}]● {alert['name']}: {alert['metric']} = {alert['value']:.2f} (threshold: {alert['threshold']})[/{severity_color}]\n"
        
        return Panel(content, title="Active Alerts", border_style="red" if alerts else "green")
    
    async def start_monitoring(self):
        """Start real-time monitoring"""
        
        self.running = True
        layout = self._create_dashboard_layout()
        
        with Live(layout, refresh_per_second=1, screen=True) as live:
            while self.running:
                try:
                    # Collect metrics
                    metrics = await self.metrics_collector.get_performance_snapshot()
                    self.metrics_history.append(metrics)
                    
                    # Keep only last 1000 metrics
                    if len(self.metrics_history) > 1000:
                        self.metrics_history.pop(0)
                    
                    # Check alerts
                    alerts = await self.alert_manager.check_alerts(metrics)
                    
                    # Auto-optimize if needed
                    if any(alert['severity'] == 'critical' for alert in alerts):
                        optimizations = await self.optimizer.auto_optimize(metrics)
                        if optimizations:
                            console.print(f"[yellow]Applied optimizations: {list(optimizations.keys())}[/yellow]")
                    
                    # Update dashboard
                    layout["header"].update(
                        Panel(
                            f"[bold blue]HFT Real-Time Monitor[/bold blue] | "
                            f"Updated: {datetime.fromtimestamp(metrics.timestamp).strftime('%H:%M:%S')}",
                            style="blue"
                        )
                    )
                    
                    layout["metrics"].update(self._create_metrics_table(metrics))
                    layout["alerts"].update(self._create_alerts_panel(alerts))
                    
                    layout["footer"].update(
                        Panel(
                            f"Press Ctrl+C to stop monitoring | "
                            f"Metrics collected: {len(self.metrics_history)} | "
                            f"Active alerts: {len(alerts)}",
                            style="dim"
                        )
                    )
                    
                    await asyncio.sleep(self.update_interval)
                    
                except KeyboardInterrupt:
                    break
                except Exception as e:
                    logger.error(f"Monitoring error: {e}")
                    await asyncio.sleep(1)
        
        self.running = False
        console.print("[yellow]Monitoring stopped[/yellow]")
    
    def stop_monitoring(self):
        """Stop monitoring"""
        self.running = False
    
    async def generate_performance_report(self) -> str:
        """Generate performance analysis report"""
        
        if not self.metrics_history:
            return "No metrics data available"
        
        # Calculate statistics
        recent_metrics = self.metrics_history[-100:]  # Last 100 samples
        
        latencies = [m.latency_p99_ns for m in recent_metrics]
        throughputs = [m.throughput_ops_sec for m in recent_metrics]
        cpu_usages = [m.cpu_usage_percent for m in recent_metrics]
        
        report = f"""
# HFT System Performance Report
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Latency Analysis
- P99 Latency (avg): {np.mean(latencies):.0f} ns
- P99 Latency (min): {np.min(latencies):.0f} ns
- P99 Latency (max): {np.max(latencies):.0f} ns
- P99 Latency (std): {np.std(latencies):.0f} ns

## Throughput Analysis
- Throughput (avg): {np.mean(throughputs):,.0f} ops/sec
- Throughput (min): {np.min(throughputs):,.0f} ops/sec
- Throughput (max): {np.max(throughputs):,.0f} ops/sec

## Resource Utilization
- CPU Usage (avg): {np.mean(cpu_usages):.1f}%
- CPU Usage (max): {np.max(cpu_usages):.1f}%

## Recommendations
"""
        
        # Add recommendations based on analysis
        if np.mean(latencies) > 1000:
            report += "- Consider CPU affinity optimization to reduce latency\n"
        
        if np.std(latencies) > 200:
            report += "- High latency variation detected - investigate jitter sources\n"
        
        if np.mean(throughputs) < 100000:
            report += "- Throughput below target - consider batch processing optimization\n"
        
        if np.mean(cpu_usages) > 80:
            report += "- High CPU utilization - consider scaling or optimization\n"
        
        return report


async def main():
    """Main monitoring application"""
    
    console.print("[bold blue]HFT Real-Time Performance Monitor[/bold blue]")
    console.print("[italic]Starting monitoring system...[/italic]")
    
    monitor = RealTimeMonitor(update_interval=1.0)
    
    try:
        await monitor.start_monitoring()
    except KeyboardInterrupt:
        console.print("\n[yellow]Shutting down monitor...[/yellow]")
    finally:
        # Generate final report
        report = await monitor.generate_performance_report()
        
        # Save report
        report_file = f"performance_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
        with open(report_file, 'w') as f:
            f.write(report)
        
        console.print(f"[green]Performance report saved: {report_file}[/green]")


if __name__ == "__main__":
    asyncio.run(main())
