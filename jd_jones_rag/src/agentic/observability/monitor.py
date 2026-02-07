"""
Agent Monitor
Monitors agent performance, alerts on issues, and tracks metrics.
"""

from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import statistics
import logging

logger = logging.getLogger(__name__)


class AlertLevel(str, Enum):
    """Alert severity levels."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class Alert:
    """A monitoring alert."""
    alert_id: str
    level: AlertLevel
    title: str
    message: str
    metric_name: str
    current_value: float
    threshold: float
    timestamp: datetime = field(default_factory=datetime.now)
    acknowledged: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "alert_id": self.alert_id,
            "level": self.level.value,
            "title": self.title,
            "message": self.message,
            "metric_name": self.metric_name,
            "current_value": self.current_value,
            "threshold": self.threshold,
            "timestamp": self.timestamp.isoformat(),
            "acknowledged": self.acknowledged
        }


@dataclass
class MetricPoint:
    """A single metric data point."""
    value: float
    timestamp: datetime


class AgentMonitor:
    """
    Monitors agent performance and health.
    
    Tracks:
    - Response latency
    - Error rates
    - Token usage
    - Tool success rates
    - Cache hit rates
    - User satisfaction (if feedback available)
    """
    
    # Default thresholds
    DEFAULT_THRESHOLDS = {
        "latency_p95_ms": 5000,      # 5 seconds
        "error_rate_percent": 5,      # 5%
        "tokens_per_request": 10000,  # Token budget
        "tool_failure_rate": 10,      # 10%
        "cache_miss_rate": 80,        # 80% miss rate is concerning
    }
    
    def __init__(
        self,
        retention_hours: int = 24,
        thresholds: Optional[Dict[str, float]] = None
    ):
        """
        Initialize monitor.
        
        Args:
            retention_hours: Hours to retain metric data
            thresholds: Custom alert thresholds
        """
        self.retention_hours = retention_hours
        self.thresholds = {**self.DEFAULT_THRESHOLDS, **(thresholds or {})}
        
        # Metric storage
        self.metrics: Dict[str, List[MetricPoint]] = {
            "latency_ms": [],
            "tokens_used": [],
            "tool_calls": [],
            "tool_failures": [],
            "retrievals": [],
            "cache_hits": [],
            "cache_misses": [],
            "errors": [],
            "requests": []
        }
        
        # Alerts
        self.alerts: List[Alert] = []
        self.alert_handlers: List[Callable[[Alert], None]] = []
    
    def record_request(
        self,
        latency_ms: float,
        tokens_used: int,
        tool_calls: int,
        tool_failures: int,
        retrievals: int,
        cache_hit: bool,
        error: bool = False
    ):
        """
        Record metrics for a request.
        
        Args:
            latency_ms: Request latency
            tokens_used: Tokens consumed
            tool_calls: Number of tool calls
            tool_failures: Failed tool calls
            retrievals: Number of retrievals
            cache_hit: Whether cache was hit
            error: Whether request had errors
        """
        now = datetime.now()
        
        self._add_metric("latency_ms", latency_ms)
        self._add_metric("tokens_used", tokens_used)
        self._add_metric("tool_calls", tool_calls)
        self._add_metric("tool_failures", tool_failures)
        self._add_metric("retrievals", retrievals)
        
        if cache_hit:
            self._add_metric("cache_hits", 1)
        else:
            self._add_metric("cache_misses", 1)
        
        if error:
            self._add_metric("errors", 1)
        
        self._add_metric("requests", 1)
        
        # Check thresholds
        self._check_alerts()
        
        # Cleanup old data
        self._cleanup_old_metrics()
    
    def _add_metric(self, name: str, value: float):
        """Add a metric data point."""
        if name not in self.metrics:
            self.metrics[name] = []
        
        self.metrics[name].append(MetricPoint(
            value=value,
            timestamp=datetime.now()
        ))
    
    def _check_alerts(self):
        """Check metrics against thresholds and create alerts."""
        import uuid
        
        # Check latency P95
        latency_p95 = self._calculate_percentile("latency_ms", 95)
        if latency_p95 and latency_p95 > self.thresholds["latency_p95_ms"]:
            self._create_alert(
                level=AlertLevel.WARNING,
                title="High Latency",
                message=f"P95 latency is {latency_p95:.0f}ms (threshold: {self.thresholds['latency_p95_ms']}ms)",
                metric_name="latency_p95_ms",
                current_value=latency_p95,
                threshold=self.thresholds["latency_p95_ms"]
            )
        
        # Check error rate
        error_rate = self._calculate_rate("errors", "requests")
        if error_rate and error_rate > self.thresholds["error_rate_percent"]:
            self._create_alert(
                level=AlertLevel.ERROR,
                title="High Error Rate",
                message=f"Error rate is {error_rate:.1f}% (threshold: {self.thresholds['error_rate_percent']}%)",
                metric_name="error_rate_percent",
                current_value=error_rate,
                threshold=self.thresholds["error_rate_percent"]
            )
        
        # Check tool failure rate
        tool_failure_rate = self._calculate_rate("tool_failures", "tool_calls")
        if tool_failure_rate and tool_failure_rate > self.thresholds["tool_failure_rate"]:
            self._create_alert(
                level=AlertLevel.WARNING,
                title="High Tool Failure Rate",
                message=f"Tool failure rate is {tool_failure_rate:.1f}%",
                metric_name="tool_failure_rate",
                current_value=tool_failure_rate,
                threshold=self.thresholds["tool_failure_rate"]
            )
    
    def _create_alert(
        self,
        level: AlertLevel,
        title: str,
        message: str,
        metric_name: str,
        current_value: float,
        threshold: float
    ):
        """Create and dispatch an alert."""
        import uuid
        
        # Check for duplicate recent alerts
        recent_cutoff = datetime.now() - timedelta(minutes=5)
        for alert in self.alerts:
            if (alert.metric_name == metric_name and 
                alert.timestamp > recent_cutoff and
                not alert.acknowledged):
                return  # Don't create duplicate
        
        alert = Alert(
            alert_id=f"alert_{uuid.uuid4().hex[:8]}",
            level=level,
            title=title,
            message=message,
            metric_name=metric_name,
            current_value=current_value,
            threshold=threshold
        )
        
        self.alerts.append(alert)
        
        # Dispatch to handlers
        for handler in self.alert_handlers:
            try:
                handler(alert)
            except Exception as e:
                logger.error(f"Alert handler error: {e}")
    
    def _calculate_percentile(self, metric_name: str, percentile: int) -> Optional[float]:
        """Calculate a percentile for a metric."""
        if metric_name not in self.metrics:
            return None
        
        values = [p.value for p in self.metrics[metric_name]]
        if not values:
            return None
        
        sorted_values = sorted(values)
        idx = int(len(sorted_values) * percentile / 100)
        return sorted_values[min(idx, len(sorted_values) - 1)]
    
    def _calculate_rate(self, numerator: str, denominator: str) -> Optional[float]:
        """Calculate a rate as percentage."""
        if numerator not in self.metrics or denominator not in self.metrics:
            return None
        
        num_sum = sum(p.value for p in self.metrics[numerator])
        den_sum = sum(p.value for p in self.metrics[denominator])
        
        if den_sum == 0:
            return None
        
        return (num_sum / den_sum) * 100
    
    def _cleanup_old_metrics(self):
        """Remove metrics older than retention period."""
        cutoff = datetime.now() - timedelta(hours=self.retention_hours)
        
        for name in self.metrics:
            self.metrics[name] = [
                p for p in self.metrics[name]
                if p.timestamp > cutoff
            ]
    
    def get_summary(self, hours: int = 1) -> Dict[str, Any]:
        """
        Get metrics summary for the specified period.
        
        Args:
            hours: Number of hours to summarize
            
        Returns:
            Summary statistics
        """
        cutoff = datetime.now() - timedelta(hours=hours)
        
        def filter_recent(metric_name: str) -> List[float]:
            if metric_name not in self.metrics:
                return []
            return [p.value for p in self.metrics[metric_name] if p.timestamp > cutoff]
        
        latencies = filter_recent("latency_ms")
        tokens = filter_recent("tokens_used")
        requests = filter_recent("requests")
        errors = filter_recent("errors")
        cache_hits = filter_recent("cache_hits")
        cache_misses = filter_recent("cache_misses")
        
        total_requests = len(requests)
        total_errors = sum(errors)
        total_cache_hits = sum(cache_hits)
        total_cache_misses = sum(cache_misses)
        
        return {
            "period_hours": hours,
            "total_requests": total_requests,
            "latency": {
                "avg_ms": statistics.mean(latencies) if latencies else 0,
                "p50_ms": statistics.median(latencies) if latencies else 0,
                "p95_ms": self._calculate_percentile("latency_ms", 95) or 0,
                "p99_ms": self._calculate_percentile("latency_ms", 99) or 0
            },
            "tokens": {
                "total": sum(tokens),
                "avg_per_request": sum(tokens) / total_requests if total_requests else 0
            },
            "errors": {
                "total": total_errors,
                "rate_percent": (total_errors / total_requests * 100) if total_requests else 0
            },
            "cache": {
                "hits": total_cache_hits,
                "misses": total_cache_misses,
                "hit_rate_percent": (
                    total_cache_hits / (total_cache_hits + total_cache_misses) * 100
                    if (total_cache_hits + total_cache_misses) > 0 else 0
                )
            },
            "tools": {
                "total_calls": sum(filter_recent("tool_calls")),
                "failures": sum(filter_recent("tool_failures")),
                "failure_rate_percent": self._calculate_rate("tool_failures", "tool_calls") or 0
            }
        }
    
    def get_alerts(
        self,
        level: Optional[AlertLevel] = None,
        unacknowledged_only: bool = False
    ) -> List[Alert]:
        """Get alerts with optional filtering."""
        alerts = self.alerts
        
        if level:
            alerts = [a for a in alerts if a.level == level]
        
        if unacknowledged_only:
            alerts = [a for a in alerts if not a.acknowledged]
        
        return sorted(alerts, key=lambda a: a.timestamp, reverse=True)
    
    def acknowledge_alert(self, alert_id: str) -> bool:
        """Acknowledge an alert."""
        for alert in self.alerts:
            if alert.alert_id == alert_id:
                alert.acknowledged = True
                return True
        return False
    
    def add_alert_handler(self, handler: Callable[[Alert], None]):
        """Add a handler function for alerts."""
        self.alert_handlers.append(handler)
    
    def set_threshold(self, metric_name: str, value: float):
        """Set a custom threshold."""
        self.thresholds[metric_name] = value
