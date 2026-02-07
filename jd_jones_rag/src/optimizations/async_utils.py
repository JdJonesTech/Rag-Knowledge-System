"""
Async Utilities
Parallelization and async optimization utilities.

Features:
1. Parallel tool execution
2. Concurrency-limited gather
3. Timeout decorators
4. Rate limiting
5. Retry with exponential backoff
"""

import asyncio
import functools
import logging
import time
from typing import List, Any, Callable, TypeVar, Optional, Coroutine, Dict
from dataclasses import dataclass
from contextlib import asynccontextmanager
from collections import deque

logger = logging.getLogger(__name__)

T = TypeVar('T')


async def parallel_execute(
    functions: List[Callable[[], Coroutine[Any, Any, T]]],
    timeout: Optional[float] = None,
    return_exceptions: bool = False
) -> List[T]:
    """
    Execute multiple async functions in parallel.
    
    Args:
        functions: List of async callables (no arguments)
        timeout: Optional timeout in seconds
        return_exceptions: If True, exceptions are returned as results
        
    Returns:
        List of results in same order as functions
        
    Example:
        results = await parallel_execute([
            lambda: tool1.execute(query),
            lambda: tool2.execute(query),
            lambda: tool3.execute(query)
        ])
    """
    # Create tasks
    tasks = [asyncio.create_task(fn()) for fn in functions]
    
    try:
        if timeout:
            results = await asyncio.wait_for(
                asyncio.gather(*tasks, return_exceptions=return_exceptions),
                timeout=timeout
            )
        else:
            results = await asyncio.gather(*tasks, return_exceptions=return_exceptions)
        
        return results
        
    except asyncio.TimeoutError:
        # Cancel remaining tasks
        for task in tasks:
            if not task.done():
                task.cancel()
        raise


async def gather_with_concurrency(
    coroutines: List[Coroutine[Any, Any, T]],
    n: int = 10,
    return_exceptions: bool = False
) -> List[T]:
    """
    Execute coroutines with limited concurrency.
    
    Args:
        coroutines: List of coroutines to execute
        n: Maximum concurrent coroutines
        return_exceptions: If True, exceptions are returned as results
        
    Returns:
        List of results in same order as coroutines
    """
    semaphore = asyncio.Semaphore(n)
    
    async def bounded_coro(index: int, coro: Coroutine) -> tuple:
        async with semaphore:
            try:
                result = await coro
                return (index, result)
            except Exception as e:
                if return_exceptions:
                    return (index, e)
                raise
    
    # Create bounded tasks
    tasks = [
        asyncio.create_task(bounded_coro(i, coro))
        for i, coro in enumerate(coroutines)
    ]
    
    # Wait for all
    completed = await asyncio.gather(*tasks, return_exceptions=return_exceptions)
    
    # Sort by original index
    sorted_results = sorted(completed, key=lambda x: x[0])
    return [r[1] for r in sorted_results]


def with_timeout(seconds: float):
    """
    Decorator to add timeout to async functions.
    
    Args:
        seconds: Timeout in seconds
        
    Example:
        @with_timeout(5.0)
        async def slow_function():
            await asyncio.sleep(10)
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        async def wrapper(*args, **kwargs) -> Any:
            return await asyncio.wait_for(
                func(*args, **kwargs),
                timeout=seconds
            )
        return wrapper
    return decorator


def with_retry(
    max_attempts: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 60.0,
    exponential_base: float = 2.0,
    exceptions: tuple = (Exception,)
):
    """
    Decorator for retry with exponential backoff.
    
    Args:
        max_attempts: Maximum retry attempts
        base_delay: Initial delay in seconds
        max_delay: Maximum delay in seconds
        exponential_base: Base for exponential backoff
        exceptions: Tuple of exceptions to catch
        
    Example:
        @with_retry(max_attempts=3, base_delay=1.0)
        async def unreliable_api_call():
            ...
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        async def wrapper(*args, **kwargs) -> Any:
            last_exception = None
            
            for attempt in range(max_attempts):
                try:
                    return await func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    
                    if attempt < max_attempts - 1:
                        delay = min(
                            base_delay * (exponential_base ** attempt),
                            max_delay
                        )
                        logger.warning(
                            f"Attempt {attempt + 1} failed: {e}. "
                            f"Retrying in {delay:.1f}s..."
                        )
                        await asyncio.sleep(delay)
            
            raise last_exception
        return wrapper
    return decorator


class RateLimiter:
    """
    Token bucket rate limiter for async operations.
    
    Features:
    - Configurable rate and burst
    - Async-safe
    - Sliding window
    """
    
    def __init__(
        self,
        rate: float,
        burst: int = 10,
        period: float = 1.0
    ):
        """
        Initialize rate limiter.
        
        Args:
            rate: Requests per period
            burst: Maximum burst size
            period: Time period in seconds
        """
        self.rate = rate
        self.burst = burst
        self.period = period
        
        self._tokens = burst
        self._last_update = time.monotonic()
        self._lock = asyncio.Lock()
    
    async def acquire(self, tokens: int = 1) -> bool:
        """
        Acquire tokens, waiting if necessary.
        
        Args:
            tokens: Number of tokens to acquire
            
        Returns:
            True if tokens acquired
        """
        async with self._lock:
            now = time.monotonic()
            
            # Add tokens based on elapsed time
            elapsed = now - self._last_update
            self._tokens = min(
                self.burst,
                self._tokens + elapsed * (self.rate / self.period)
            )
            self._last_update = now
            
            if self._tokens >= tokens:
                self._tokens -= tokens
                return True
            
            # Wait for tokens
            wait_time = (tokens - self._tokens) * (self.period / self.rate)
            await asyncio.sleep(wait_time)
            
            self._tokens = 0
            return True
    
    @asynccontextmanager
    async def limited(self):
        """Context manager for rate limiting."""
        await self.acquire()
        try:
            yield
        finally:
            pass


class AsyncTaskQueue:
    """
    Async task queue with priority support.
    
    Features:
    - Priority-based execution
    - Configurable workers
    - Task status tracking
    """
    
    @dataclass
    class Task:
        id: str
        priority: int
        coro: Coroutine
        created_at: float = None
        
        def __post_init__(self):
            if self.created_at is None:
                self.created_at = time.time()
        
        def __lt__(self, other):
            return self.priority < other.priority
    
    def __init__(self, workers: int = 4):
        """
        Initialize task queue.
        
        Args:
            workers: Number of worker coroutines
        """
        self.workers = workers
        self._queue: asyncio.PriorityQueue = asyncio.PriorityQueue()
        self._results: Dict[str, Any] = {}
        self._running = False
        self._worker_tasks: List[asyncio.Task] = []
    
    async def start(self):
        """Start worker coroutines."""
        if not self._running:
            self._running = True
            self._worker_tasks = [
                asyncio.create_task(self._worker(i))
                for i in range(self.workers)
            ]
            logger.info(f"Task queue started with {self.workers} workers")
    
    async def stop(self):
        """Stop worker coroutines."""
        self._running = False
        for task in self._worker_tasks:
            task.cancel()
        await asyncio.gather(*self._worker_tasks, return_exceptions=True)
        logger.info("Task queue stopped")
    
    async def submit(
        self,
        task_id: str,
        coro: Coroutine,
        priority: int = 5
    ):
        """
        Submit a task to the queue.
        
        Args:
            task_id: Unique task identifier
            coro: Coroutine to execute
            priority: Priority (lower = higher priority)
        """
        task = self.Task(id=task_id, priority=priority, coro=coro)
        await self._queue.put((priority, task))
    
    async def get_result(self, task_id: str, timeout: float = None) -> Any:
        """
        Get result for a task.
        
        Args:
            task_id: Task identifier
            timeout: Optional timeout in seconds
            
        Returns:
            Task result
        """
        start = time.time()
        while True:
            if task_id in self._results:
                return self._results.pop(task_id)
            
            if timeout and (time.time() - start) > timeout:
                raise asyncio.TimeoutError(f"Task {task_id} timed out")
            
            await asyncio.sleep(0.01)
    
    async def _worker(self, worker_id: int):
        """Worker coroutine."""
        while self._running:
            try:
                _, task = await asyncio.wait_for(
                    self._queue.get(),
                    timeout=1.0
                )
                
                try:
                    result = await task.coro
                    self._results[task.id] = result
                except Exception as e:
                    self._results[task.id] = e
                    
            except asyncio.TimeoutError:
                continue
            except asyncio.CancelledError:
                break


async def run_in_executor(fn: Callable, *args, **kwargs) -> Any:
    """
    Run a sync function in the thread pool executor.
    
    Args:
        fn: Sync function to run
        *args, **kwargs: Arguments for function
        
    Returns:
        Function result
    """
    loop = asyncio.get_event_loop()
    
    if kwargs:
        # Partial application for kwargs support
        fn = functools.partial(fn, **kwargs)
    
    return await loop.run_in_executor(None, fn, *args)


def create_batch_scheduler(
    batch_fn: Callable[[List[Any]], List[Any]],
    batch_size: int = 32,
    max_wait_ms: int = 50
) -> Callable[[Any], Coroutine[Any, Any, Any]]:
    """
    Create a batch scheduler for automatic request batching.
    
    Args:
        batch_fn: Function to process a batch
        batch_size: Maximum batch size
        max_wait_ms: Maximum wait time before processing
        
    Returns:
        Async function that accepts single items and batches them
    """
    queue: deque = deque()
    lock = asyncio.Lock()
    processing = False
    
    async def schedule(item: Any) -> Any:
        nonlocal processing
        
        future = asyncio.get_event_loop().create_future()
        
        async with lock:
            queue.append((item, future))
            
            if not processing:
                processing = True
                asyncio.create_task(process_batch())
        
        return await future
    
    async def process_batch():
        nonlocal processing
        
        await asyncio.sleep(max_wait_ms / 1000)
        
        async with lock:
            if not queue:
                processing = False
                return
            
            # Collect batch
            batch = []
            futures = []
            while queue and len(batch) < batch_size:
                item, future = queue.popleft()
                batch.append(item)
                futures.append(future)
        
        try:
            # Process batch
            if asyncio.iscoroutinefunction(batch_fn):
                results = await batch_fn(batch)
            else:
                results = await run_in_executor(batch_fn, batch)
            
            # Set results
            for future, result in zip(futures, results):
                if not future.done():
                    future.set_result(result)
                    
        except Exception as e:
            for future in futures:
                if not future.done():
                    future.set_exception(e)
        
        finally:
            async with lock:
                if queue:
                    asyncio.create_task(process_batch())
                else:
                    processing = False
    
    return schedule
