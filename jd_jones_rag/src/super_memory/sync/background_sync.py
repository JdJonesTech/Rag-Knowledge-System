"""
Background Sync Module
Celery tasks for background memory synchronization.
"""

from celery import Celery
from typing import Optional, Dict, Any, List
import asyncio
from datetime import datetime, timedelta

from src.config.settings import settings


# Initialize Celery app
celery_app = Celery(
    'jd_jones_rag',
    broker=settings.celery_broker_url,
    backend=settings.celery_result_backend
)

# Celery configuration
celery_app.conf.update(
    task_serializer='json',
    accept_content=['json'],
    result_serializer='json',
    timezone='UTC',
    enable_utc=True,
    task_track_started=True,
    task_time_limit=300,  # 5 minutes
    worker_prefetch_multiplier=1,
    
    # Beat schedule for periodic tasks
    beat_schedule={
        'sync-memories-every-hour': {
            'task': 'src.super_memory.sync.background_sync.sync_all_users',
            'schedule': timedelta(hours=1),
        },
        'cleanup-old-sync-logs-daily': {
            'task': 'src.super_memory.sync.background_sync.cleanup_old_logs',
            'schedule': timedelta(days=1),
        },
        'refresh-memory-embeddings-weekly': {
            'task': 'src.super_memory.sync.background_sync.refresh_embeddings',
            'schedule': timedelta(weeks=1),
        },
    },
)


def run_async(coro):
    """Helper to run async functions in sync context."""
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


@celery_app.task(bind=True, max_retries=3)
def sync_user_memories(self, user_id: str, provider: str = None) -> Dict[str, Any]:
    """
    Sync memories for a specific user.
    
    Args:
        user_id: User to sync
        provider: Optional specific provider to sync with
        
    Returns:
        Sync result summary
    """
    from src.super_memory.sync.sync_manager import SyncManager
    
    try:
        sync_manager = SyncManager()
        
        if provider:
            result = run_async(sync_manager.sync_with_provider(user_id, provider))
        else:
            result = run_async(sync_manager.sync_all_providers(user_id))
        
        return {
            "status": "success",
            "user_id": user_id,
            "provider": provider or "all",
            "result": result
        }
        
    except Exception as e:
        self.retry(exc=e, countdown=60 * (self.request.retries + 1))


@celery_app.task(bind=True)
def sync_all_users(self) -> Dict[str, Any]:
    """
    Sync memories for all users with auto-sync enabled.
    
    Returns:
        Summary of all sync operations
    """
    from src.super_memory.memory_manager import MemoryManager
    
    try:
        memory_manager = MemoryManager()
        
        # Get all users with auto-sync enabled
        users = run_async(memory_manager.get_auto_sync_users())
        
        results = []
        for user_id in users:
            # Queue individual sync tasks
            task = sync_user_memories.delay(user_id)
            results.append({
                "user_id": user_id,
                "task_id": task.id
            })
        
        return {
            "status": "success",
            "users_queued": len(results),
            "tasks": results
        }
        
    except Exception as e:
        return {
            "status": "error",
            "error": str(e)
        }


@celery_app.task(bind=True)
def cleanup_old_logs(self, days_to_keep: int = 30) -> Dict[str, Any]:
    """
    Clean up old sync logs.
    
    Args:
        days_to_keep: Number of days of logs to keep
        
    Returns:
        Cleanup summary
    """
    from src.super_memory.memory_manager import MemoryManager
    
    try:
        memory_manager = MemoryManager()
        
        cutoff_date = datetime.utcnow() - timedelta(days=days_to_keep)
        deleted_count = run_async(memory_manager.cleanup_old_sync_logs(cutoff_date))
        
        return {
            "status": "success",
            "deleted_count": deleted_count,
            "cutoff_date": cutoff_date.isoformat()
        }
        
    except Exception as e:
        return {
            "status": "error",
            "error": str(e)
        }


@celery_app.task(bind=True)
def refresh_embeddings(self, batch_size: int = 100) -> Dict[str, Any]:
    """
    Refresh memory embeddings for all users.
    Useful when embedding model is updated.
    
    Args:
        batch_size: Number of memories to process at a time
        
    Returns:
        Refresh summary
    """
    from src.super_memory.memory_manager import MemoryManager
    
    try:
        memory_manager = MemoryManager()
        
        refreshed_count = run_async(
            memory_manager.refresh_all_embeddings(batch_size=batch_size)
        )
        
        return {
            "status": "success",
            "refreshed_count": refreshed_count
        }
        
    except Exception as e:
        return {
            "status": "error",
            "error": str(e)
        }


@celery_app.task(bind=True, max_retries=3)
def process_conversation_summary(
    self,
    user_id: str,
    session_id: str,
    messages: List[Dict[str, str]]
) -> Dict[str, Any]:
    """
    Process and store a conversation summary.
    
    Args:
        user_id: User ID
        session_id: Session ID
        messages: Conversation messages
        
    Returns:
        Processing result
    """
    from src.super_memory.memory_manager import MemoryManager
    
    try:
        memory_manager = MemoryManager()
        
        summary = run_async(
            memory_manager.summarize_and_store_conversation(
                user_id=user_id,
                session_id=session_id,
                messages=messages
            )
        )
        
        return {
            "status": "success",
            "user_id": user_id,
            "session_id": session_id,
            "summary_id": summary.get("context_id")
        }
        
    except Exception as e:
        self.retry(exc=e, countdown=30)


@celery_app.task(bind=True)
def extract_memories_from_conversation(
    self,
    user_id: str,
    session_id: str,
    messages: List[Dict[str, str]]
) -> Dict[str, Any]:
    """
    Extract and store memories from a conversation.
    
    Args:
        user_id: User ID
        session_id: Session ID
        messages: Conversation messages
        
    Returns:
        Extraction result
    """
    from src.super_memory.memory_learner import MemoryLearner
    
    try:
        learner = MemoryLearner()
        
        memories = run_async(
            learner.extract_memories_from_conversation(
                user_id=user_id,
                messages=messages
            )
        )
        
        return {
            "status": "success",
            "user_id": user_id,
            "session_id": session_id,
            "memories_extracted": len(memories)
        }
        
    except Exception as e:
        return {
            "status": "error",
            "error": str(e)
        }


@celery_app.task(bind=True)
def resolve_conflicts(self, user_id: str) -> Dict[str, Any]:
    """
    Resolve pending memory conflicts for a user.
    
    Args:
        user_id: User ID
        
    Returns:
        Resolution result
    """
    from src.super_memory.sync.conflict_resolver import ConflictResolver
    
    try:
        resolver = ConflictResolver()
        
        resolved = run_async(resolver.resolve_all_pending(user_id))
        
        return {
            "status": "success",
            "user_id": user_id,
            "conflicts_resolved": resolved
        }
        
    except Exception as e:
        return {
            "status": "error",
            "error": str(e)
        }


# Export celery app for command line
app = celery_app
