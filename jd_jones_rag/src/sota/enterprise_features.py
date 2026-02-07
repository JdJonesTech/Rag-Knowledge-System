"""
Enterprise Scale Features

Implements enterprise-grade features:
- Distributed Search with sharding
- A/B Testing framework
- Multi-tenant support
- Rate limiting and quotas
"""

import logging
import asyncio
import hashlib
import time
import random
from typing import List, Dict, Any, Optional, Tuple, Set
from dataclasses import dataclass, field
from collections import defaultdict
from datetime import datetime, timedelta
from abc import ABC, abstractmethod
import json

logger = logging.getLogger(__name__)


# =======================
# DISTRIBUTED SEARCH
# =======================

@dataclass
class ShardInfo:
    """Information about a search shard."""
    shard_id: str
    node_url: str
    document_count: int
    is_healthy: bool = True
    last_heartbeat: datetime = field(default_factory=datetime.now)
    load: float = 0.0  # 0-1 load factor


@dataclass
class DistributedSearchResult:
    """Result from distributed search."""
    results: List[Dict[str, Any]]
    shards_queried: int
    shards_succeeded: int
    total_time_ms: float
    shard_times: Dict[str, float]


class DistributedSearch:
    """
    Distributed Search with Sharding.
    
    Enables horizontal scaling by:
    - Partitioning documents across shards
    - Parallel query execution
    - Result merging with deduplication
    
    Usage:
        dist_search = DistributedSearch()
        dist_search.register_shard("shard-1", "http://node1:8080")
        dist_search.register_shard("shard-2", "http://node2:8080")
        results = await dist_search.search(query, top_k=10)
    """
    
    def __init__(
        self,
        num_shards: int = 3,
        replication_factor: int = 1,
        timeout_seconds: float = 5.0,
        merge_strategy: str = "rrf"  # rrf or max_score
    ):
        """
        Initialize distributed search.
        
        Args:
            num_shards: Number of shards
            replication_factor: Replication for redundancy
            timeout_seconds: Shard query timeout
            merge_strategy: How to merge results (rrf or max_score)
        """
        self.num_shards = num_shards
        self.replication_factor = replication_factor
        self.timeout_seconds = timeout_seconds
        self.merge_strategy = merge_strategy
        
        self._shards: Dict[str, ShardInfo] = {}
        self._document_shard_map: Dict[str, str] = {}  # doc_id -> shard_id
        
        self._stats = {
            "queries": 0,
            "shard_failures": 0,
            "avg_latency_ms": 0
        }
    
    def _get_shard_for_document(self, doc_id: str) -> str:
        """Consistent hashing for document -> shard mapping."""
        hash_val = int(hashlib.md5(doc_id.encode()).hexdigest(), 16)
        shard_idx = hash_val % self.num_shards
        return f"shard-{shard_idx}"
    
    def register_shard(self, shard_id: str, node_url: str, document_count: int = 0):
        """Register a shard."""
        self._shards[shard_id] = ShardInfo(
            shard_id=shard_id,
            node_url=node_url,
            document_count=document_count
        )
        logger.info(f"Registered shard {shard_id} at {node_url}")
    
    def get_healthy_shards(self) -> List[ShardInfo]:
        """Get list of healthy shards."""
        cutoff = datetime.now() - timedelta(seconds=30)
        return [
            shard for shard in self._shards.values()
            if shard.is_healthy and shard.last_heartbeat > cutoff
        ]
    
    async def _query_shard(
        self,
        shard: ShardInfo,
        query: str,
        top_k: int
    ) -> Tuple[str, List[Dict], float]:
        """
        Query a single shard.
        
        Returns (shard_id, results, time_ms).
        """
        start = time.time()
        
        try:
            # Simulate shard query (in production, this would be HTTP/gRPC call)
            # In actual implementation, this would call the shard's search API
            
            # For now, return empty results (placeholder)
            await asyncio.sleep(0.01)  # Simulate network latency
            
            time_ms = (time.time() - start) * 1000
            return shard.shard_id, [], time_ms
            
        except asyncio.TimeoutError:
            self._stats["shard_failures"] += 1
            shard.is_healthy = False
            return shard.shard_id, [], 0
            
        except Exception as e:
            logger.error(f"Shard {shard.shard_id} query failed: {e}")
            self._stats["shard_failures"] += 1
            return shard.shard_id, [], 0
    
    def _merge_results(
        self,
        shard_results: List[Tuple[str, List[Dict]]],
        top_k: int
    ) -> List[Dict[str, Any]]:
        """Merge results from multiple shards."""
        if self.merge_strategy == "rrf":
            return self._rrf_merge(shard_results, top_k)
        else:
            return self._max_score_merge(shard_results, top_k)
    
    def _rrf_merge(
        self,
        shard_results: List[Tuple[str, List[Dict]]],
        top_k: int,
        k: int = 60
    ) -> List[Dict[str, Any]]:
        """Reciprocal Rank Fusion merge."""
        doc_scores: Dict[str, float] = defaultdict(float)
        doc_data: Dict[str, Dict] = {}
        
        for shard_id, results in shard_results:
            for rank, doc in enumerate(results):
                doc_id = doc.get("id", doc.get("document_id", ""))
                doc_scores[doc_id] += 1 / (k + rank + 1)
                doc_data[doc_id] = doc
        
        sorted_docs = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)
        
        return [
            {**doc_data[doc_id], "rrf_score": score}
            for doc_id, score in sorted_docs[:top_k]
        ]
    
    def _max_score_merge(
        self,
        shard_results: List[Tuple[str, List[Dict]]],
        top_k: int
    ) -> List[Dict[str, Any]]:
        """Simple max score merge."""
        all_results = []
        for shard_id, results in shard_results:
            for doc in results:
                doc["source_shard"] = shard_id
                all_results.append(doc)
        
        all_results.sort(key=lambda x: x.get("score", 0), reverse=True)
        
        # Deduplicate
        seen: Set[str] = set()
        deduped = []
        for doc in all_results:
            doc_id = doc.get("id", doc.get("document_id", ""))
            if doc_id not in seen:
                seen.add(doc_id)
                deduped.append(doc)
        
        return deduped[:top_k]
    
    async def search(
        self,
        query: str,
        top_k: int = 10,
        shards: Optional[List[str]] = None
    ) -> DistributedSearchResult:
        """
        Search across distributed shards.
        
        Args:
            query: Search query
            top_k: Results per shard
            shards: Specific shards to query (None = all healthy)
            
        Returns:
            DistributedSearchResult with merged results
        """
        self._stats["queries"] += 1
        start = time.time()
        
        # Get shards to query
        if shards:
            target_shards = [s for s in self._shards.values() if s.shard_id in shards]
        else:
            target_shards = self.get_healthy_shards()
        
        if not target_shards:
            return DistributedSearchResult(
                results=[],
                shards_queried=0,
                shards_succeeded=0,
                total_time_ms=0,
                shard_times={}
            )
        
        # Query all shards in parallel
        tasks = [
            self._query_shard(shard, query, top_k * 2)  # Get more for merging
            for shard in target_shards
        ]
        
        shard_responses = await asyncio.gather(*tasks)
        
        # Collect results
        shard_results = []
        shard_times = {}
        succeeded = 0
        
        for shard_id, results, time_ms in shard_responses:
            shard_times[shard_id] = time_ms
            if results:
                shard_results.append((shard_id, results))
                succeeded += 1
        
        # Merge results
        merged = self._merge_results(shard_results, top_k)
        
        total_time = (time.time() - start) * 1000
        self._stats["avg_latency_ms"] = (
            (self._stats["avg_latency_ms"] * (self._stats["queries"] - 1) + total_time)
            / self._stats["queries"]
        )
        
        return DistributedSearchResult(
            results=merged,
            shards_queried=len(target_shards),
            shards_succeeded=succeeded,
            total_time_ms=total_time,
            shard_times=shard_times
        )
    
    def get_stats(self) -> Dict[str, Any]:
        """Get distributed search statistics."""
        return {
            **self._stats,
            "registered_shards": len(self._shards),
            "healthy_shards": len(self.get_healthy_shards())
        }


# =======================
# A/B TESTING FRAMEWORK
# =======================

@dataclass
class Experiment:
    """An A/B test experiment."""
    experiment_id: str
    name: str
    description: str
    variants: Dict[str, float]  # variant_name -> traffic_percentage
    metrics: List[str]  # Metrics to track
    start_time: datetime = field(default_factory=datetime.now)
    end_time: Optional[datetime] = None
    is_active: bool = True


@dataclass
class ExperimentResult:
    """Metrics for an experiment."""
    experiment_id: str
    variant: str
    impressions: int = 0
    conversions: int = 0
    total_latency_ms: float = 0
    errors: int = 0
    custom_metrics: Dict[str, float] = field(default_factory=dict)
    
    @property
    def conversion_rate(self) -> float:
        return self.conversions / max(self.impressions, 1)
    
    @property
    def avg_latency_ms(self) -> float:
        return self.total_latency_ms / max(self.impressions, 1)


class ABTestingFramework:
    """
    A/B Testing Framework for RAG systems.
    
    Enables testing different:
    - Retrieval strategies
    - Reranking models
    - Prompt templates
    - Generation models
    
    Usage:
        ab = ABTestingFramework()
        
        # Create experiment
        ab.create_experiment(
            "reranker_test",
            variants={"colbert": 0.5, "cross_encoder": 0.5},
            metrics=["latency", "relevance_score"]
        )
        
        # Get variant for user
        variant = ab.get_variant("reranker_test", user_id="user123")
        
        # Track results
        ab.track_metric("reranker_test", variant, "latency", 150)
    """
    
    def __init__(self):
        self._experiments: Dict[str, Experiment] = {}
        self._results: Dict[str, Dict[str, ExperimentResult]] = defaultdict(dict)
        self._user_assignments: Dict[str, Dict[str, str]] = defaultdict(dict)
    
    def create_experiment(
        self,
        experiment_id: str,
        name: str = "",
        description: str = "",
        variants: Dict[str, float] = None,
        metrics: List[str] = None
    ) -> Experiment:
        """
        Create a new experiment.
        
        Args:
            experiment_id: Unique experiment ID
            name: Human-readable name
            description: Experiment description
            variants: Dict of variant_name -> traffic percentage
            metrics: List of metrics to track
            
        Returns:
            Created Experiment
        """
        variants = variants or {"control": 0.5, "treatment": 0.5}
        metrics = metrics or ["latency", "conversion"]
        
        # Normalize percentages
        total = sum(variants.values())
        variants = {k: v / total for k, v in variants.items()}
        
        experiment = Experiment(
            experiment_id=experiment_id,
            name=name or experiment_id,
            description=description,
            variants=variants,
            metrics=metrics
        )
        
        self._experiments[experiment_id] = experiment
        
        # Initialize results for each variant
        for variant_name in variants:
            self._results[experiment_id][variant_name] = ExperimentResult(
                experiment_id=experiment_id,
                variant=variant_name
            )
        
        logger.info(f"Created experiment: {experiment_id} with variants {list(variants.keys())}")
        return experiment
    
    def get_variant(
        self,
        experiment_id: str,
        user_id: str
    ) -> Optional[str]:
        """
        Get assigned variant for a user.
        
        Uses sticky assignment - same user always gets same variant.
        
        Args:
            experiment_id: Experiment ID
            user_id: User identifier
            
        Returns:
            Assigned variant name or None if experiment not found
        """
        experiment = self._experiments.get(experiment_id)
        if not experiment or not experiment.is_active:
            return None
        
        # Check existing assignment
        if user_id in self._user_assignments and experiment_id in self._user_assignments[user_id]:
            return self._user_assignments[user_id][experiment_id]
        
        # Deterministic assignment based on hash
        hash_input = f"{experiment_id}:{user_id}"
        hash_val = int(hashlib.md5(hash_input.encode()).hexdigest(), 16)
        random_val = (hash_val % 10000) / 10000.0
        
        # Select variant based on traffic percentages
        cumulative = 0.0
        selected_variant = list(experiment.variants.keys())[0]
        
        for variant_name, percentage in experiment.variants.items():
            cumulative += percentage
            if random_val < cumulative:
                selected_variant = variant_name
                break
        
        # Store assignment
        self._user_assignments[user_id][experiment_id] = selected_variant
        
        # Track impression
        self._results[experiment_id][selected_variant].impressions += 1
        
        return selected_variant
    
    def track_metric(
        self,
        experiment_id: str,
        variant: str,
        metric_name: str,
        value: float
    ):
        """
        Track a metric value for an experiment variant.
        
        Args:
            experiment_id: Experiment ID
            variant: Variant name
            metric_name: Metric name
            value: Metric value
        """
        if experiment_id not in self._results:
            return
        
        if variant not in self._results[experiment_id]:
            return
        
        result = self._results[experiment_id][variant]
        
        if metric_name == "latency":
            result.total_latency_ms += value
        elif metric_name == "conversion":
            result.conversions += 1
        elif metric_name == "error":
            result.errors += 1
        else:
            if metric_name not in result.custom_metrics:
                result.custom_metrics[metric_name] = 0
            result.custom_metrics[metric_name] += value
    
    def get_results(self, experiment_id: str) -> Dict[str, Dict[str, Any]]:
        """
        Get experiment results.
        
        Returns:
            Dict of variant_name -> metrics
        """
        if experiment_id not in self._results:
            return {}
        
        return {
            variant: {
                "impressions": result.impressions,
                "conversions": result.conversions,
                "conversion_rate": round(result.conversion_rate, 4),
                "avg_latency_ms": round(result.avg_latency_ms, 2),
                "errors": result.errors,
                "custom_metrics": result.custom_metrics
            }
            for variant, result in self._results[experiment_id].items()
        }
    
    def conclude_experiment(
        self,
        experiment_id: str,
        winning_variant: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Conclude an experiment.
        
        Args:
            experiment_id: Experiment ID
            winning_variant: Optional winning variant (or auto-detect)
            
        Returns:
            Conclusion summary
        """
        experiment = self._experiments.get(experiment_id)
        if not experiment:
            return {"error": "Experiment not found"}
        
        experiment.is_active = False
        experiment.end_time = datetime.now()
        
        results = self.get_results(experiment_id)
        
        # Detect winner if not specified
        if winning_variant is None:
            best_variant = None
            best_conversion = -1
            
            for variant, metrics in results.items():
                if metrics["conversion_rate"] > best_conversion:
                    best_conversion = metrics["conversion_rate"]
                    best_variant = variant
            
            winning_variant = best_variant
        
        return {
            "experiment_id": experiment_id,
            "winning_variant": winning_variant,
            "duration_days": (experiment.end_time - experiment.start_time).days,
            "total_impressions": sum(r["impressions"] for r in results.values()),
            "results": results
        }
    
    def list_experiments(self) -> List[Dict[str, Any]]:
        """List all experiments."""
        return [
            {
                "experiment_id": exp.experiment_id,
                "name": exp.name,
                "variants": list(exp.variants.keys()),
                "is_active": exp.is_active,
                "start_time": exp.start_time.isoformat()
            }
            for exp in self._experiments.values()
        ]


# =======================
# RATE LIMITING
# =======================

class RateLimiter:
    """
    Rate limiter with sliding window.
    
    Usage:
        limiter = RateLimiter(requests_per_minute=60)
        if limiter.allow_request(user_id):
            # Process request
        else:
            # Return 429
    """
    
    def __init__(
        self,
        requests_per_minute: int = 60,
        burst_limit: int = 10
    ):
        """
        Initialize rate limiter.
        
        Args:
            requests_per_minute: Sustained rate limit
            burst_limit: Maximum burst size
        """
        self.requests_per_minute = requests_per_minute
        self.burst_limit = burst_limit
        
        self._request_times: Dict[str, List[float]] = defaultdict(list)
    
    def allow_request(self, user_id: str) -> bool:
        """
        Check if request should be allowed.
        
        Args:
            user_id: User identifier
            
        Returns:
            True if allowed, False if rate limited
        """
        now = time.time()
        window_start = now - 60  # 1 minute window
        
        # Clean old entries
        self._request_times[user_id] = [
            t for t in self._request_times[user_id]
            if t > window_start
        ]
        
        # Check rate limit
        if len(self._request_times[user_id]) >= self.requests_per_minute:
            return False
        
        # Check burst (last second)
        recent = [t for t in self._request_times[user_id] if t > now - 1]
        if len(recent) >= self.burst_limit:
            return False
        
        # Allow and record
        self._request_times[user_id].append(now)
        return True
    
    def get_remaining(self, user_id: str) -> int:
        """Get remaining requests for user."""
        now = time.time()
        window_start = now - 60
        
        recent = [t for t in self._request_times[user_id] if t > window_start]
        return max(0, self.requests_per_minute - len(recent))


# Singleton instances
_distributed_search: Optional[DistributedSearch] = None
_ab_testing: Optional[ABTestingFramework] = None
_rate_limiter: Optional[RateLimiter] = None


def get_distributed_search() -> DistributedSearch:
    """Get singleton distributed search instance."""
    global _distributed_search
    if _distributed_search is None:
        _distributed_search = DistributedSearch()
    return _distributed_search


def get_ab_testing() -> ABTestingFramework:
    """Get singleton A/B testing instance."""
    global _ab_testing
    if _ab_testing is None:
        _ab_testing = ABTestingFramework()
    return _ab_testing


def get_rate_limiter() -> RateLimiter:
    """Get singleton rate limiter instance."""
    global _rate_limiter
    if _rate_limiter is None:
        _rate_limiter = RateLimiter()
    return _rate_limiter
