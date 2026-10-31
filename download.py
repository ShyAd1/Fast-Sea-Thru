"""Parallel COCO image downloader with adaptive worker scaling and retry logic.

This script efficiently downloads images from a COCO-format dataset using parallel
workers that scale dynamically based on server performance and failures.
"""

import argparse
import asyncio
import json
import logging
import signal
import sys
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Deque, List, Optional, Set

import httpx
from coco_lib.objectdetection import ObjectDetectionDataset
from coco_lib.common import Image
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)
from tqdm.asyncio import tqdm


# Configuration constants
MIN_WORKERS: int = 1
MAX_WORKERS: int = 5
INITIAL_WORKERS: int = 1
SCALE_UP_THRESHOLD: int = 10  # Successful downloads before scaling up
SCALE_DOWN_THRESHOLD: int = 3  # Failures before scaling down
TIMEOUT_SECONDS: float = 30.0
MAX_RETRIES: int = 5


# Configure logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Create console handler with formatting
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.WARNING)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)


@dataclass
class DownloadStats:
    """Statistics for tracking download performance.
    
    Attributes:
        successful: Number of successful downloads
        failed: Number of failed downloads
        retries: Number of retry attempts
        bytes_downloaded: Total bytes downloaded
        current_workers: Current number of active workers
        recent_successes: Recent successful downloads for scaling decisions
        recent_failures: Recent failed downloads for scaling decisions
    """
    successful: int = 0
    failed: int = 0
    retries: int = 0
    bytes_downloaded: int = 0
    current_workers: int = INITIAL_WORKERS
    recent_successes: Deque[bool] = field(default_factory=lambda: deque(maxlen=20))
    recent_failures: Deque[bool] = field(default_factory=lambda: deque(maxlen=20))


@dataclass
class DownloadTask:
    """Represents a single image download task.
    
    Attributes:
        image_id: Unique identifier for the image
        url: Source URL for the image
        output_path: Destination path for the downloaded image
        attempt: Current attempt number
    """
    image_id: int
    url: str
    output_path: Path
    attempt: int = 0


class AdaptiveDownloader:
    """Adaptive parallel image downloader with dynamic worker scaling.
    
    This downloader adjusts the number of concurrent workers based on
    server performance, scaling up during successful operations and
    scaling down during failures or timeouts.
    """
    
    def __init__(
        self,
        output_dir: Path,
        min_workers: int = MIN_WORKERS,
        max_workers: int = MAX_WORKERS,
        initial_workers: int = INITIAL_WORKERS,
    ) -> None:
        """Initialize the adaptive downloader.
        
        Args:
            output_dir: Directory to save downloaded images
            min_workers: Minimum number of concurrent workers
            max_workers: Maximum number of concurrent workers
            initial_workers: Initial number of workers to start with
        """
        self.output_dir = output_dir
        self.min_workers = min_workers
        self.max_workers = max_workers
        self.stats = DownloadStats(current_workers=initial_workers)
        
        self.task_queue: asyncio.Queue[Optional[DownloadTask]] = asyncio.Queue()
        self.completed_tasks: Set[int] = set()
        self.lock = asyncio.Lock()
        self.cancelled = False
        
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    async def download_image(
        self,
        client: httpx.AsyncClient,
        task: DownloadTask,
    ) -> bool:
        """Download a single image with retry logic.
        
        Args:
            client: HTTP client for making requests
            task: Download task containing URL and output path
            
        Returns:
            True if download succeeded, False otherwise
        """
        
        @retry(
            retry=retry_if_exception_type((httpx.TimeoutException, httpx.ConnectError)),
            stop=stop_after_attempt(MAX_RETRIES),
            wait=wait_exponential(multiplier=1, min=2, max=30),
            reraise=True,
        )
        async def _download_with_retry() -> bytes:
            """Inner function with tenacity retry decorator."""
            response = await client.get(
                task.url,
                timeout=TIMEOUT_SECONDS,
                follow_redirects=True,
            )
            response.raise_for_status()
            return response.content
        
        try:
            content = await _download_with_retry()
            
            # Write to file
            task.output_path.write_bytes(content)
            
            async with self.lock:
                self.stats.successful += 1
                self.stats.bytes_downloaded += len(content)
                self.stats.recent_successes.append(True)
                self.stats.recent_failures.append(False)
            
            return True
            
        except httpx.HTTPStatusError as e:
            status_code = e.response.status_code
            error_details = {
                404: "Image not found on server",
                403: "Access forbidden - check URL permissions",
                500: "Server error",
                503: "Service unavailable - server may be overloaded",
            }
            detail = error_details.get(status_code, f"HTTP {status_code} error")
            logger.warning(
                f"Failed to download image {task.image_id}: {detail}\n  URL: {task.url}"
            )
            async with self.lock:
                self.stats.failed += 1
                self.stats.recent_successes.append(False)
                self.stats.recent_failures.append(True)
            return False
            
        except (httpx.TimeoutException, httpx.ConnectError) as e:
            logger.warning(
                f"Network error for image {task.image_id} after {MAX_RETRIES} retries: {e}"
            )
            async with self.lock:
                self.stats.failed += 1
                self.stats.retries += task.attempt
                self.stats.recent_successes.append(False)
                self.stats.recent_failures.append(True)
            return False
            
        except Exception as e:
            logger.error(f"Unexpected error for image {task.image_id}: {e}")
            async with self.lock:
                self.stats.failed += 1
                self.stats.recent_successes.append(False)
                self.stats.recent_failures.append(True)
            return False
    
    def should_scale_up(self) -> bool:
        """Determine if we should add more workers.
        
        Returns:
            True if conditions are right for scaling up
        """
        recent_success_count = sum(self.stats.recent_successes)
        recent_failure_count = sum(self.stats.recent_failures)
        
        return (
            self.stats.current_workers < self.max_workers
            and recent_success_count >= SCALE_UP_THRESHOLD
            and recent_failure_count < 2
        )
    
    def should_scale_down(self) -> bool:
        """Determine if we should remove workers.
        
        Returns:
            True if conditions are right for scaling down
        """
        recent_failure_count = sum(self.stats.recent_failures)
        
        return (
            self.stats.current_workers > self.min_workers
            and recent_failure_count >= SCALE_DOWN_THRESHOLD
        )
    
    async def worker(
        self,
        worker_id: int,
        client: httpx.AsyncClient,
        progress_bar: tqdm,
    ) -> None:
        """Worker coroutine that processes download tasks.
        
        Args:
            worker_id: Unique identifier for this worker
            client: HTTP client for making requests
            progress_bar: Progress bar to update
        """
        while True:
            # Check if cancelled
            if self.cancelled:
                break
                
            task = await self.task_queue.get()
            
            # Poison pill to stop worker
            if task is None:
                self.task_queue.task_done()
                break
            
            # Skip if already completed
            if task.image_id in self.completed_tasks:
                self.task_queue.task_done()
                progress_bar.update(1)
                continue
            
            # Download the image
            success = await self.download_image(client, task)
            
            if success:
                async with self.lock:
                    self.completed_tasks.add(task.image_id)
            
            self.task_queue.task_done()
            progress_bar.update(1)
            
            # Update progress bar description with stats
            progress_bar.set_postfix({
                'workers': self.stats.current_workers,
                'success': self.stats.successful,
                'failed': self.stats.failed,
                'MB': f'{self.stats.bytes_downloaded / 1024 / 1024:.1f}'
            })
    
    async def scale_workers(
        self,
        workers: List[asyncio.Task],
        client: httpx.AsyncClient,
        progress_bar: tqdm,
    ) -> List[asyncio.Task]:
        """Dynamically scale the number of workers based on performance.
        
        Args:
            workers: Current list of worker tasks
            client: HTTP client to pass to new workers
            progress_bar: Progress bar to pass to new workers
            
        Returns:
            Updated list of worker tasks
        """
        if self.should_scale_up():
            new_worker_count = min(
                self.stats.current_workers + 1,
                self.max_workers
            )
            diff = new_worker_count - self.stats.current_workers
            
            if diff > 0:
                logger.info(f"📈 Scaling UP: {self.stats.current_workers} → {new_worker_count} workers")
                for i in range(diff):
                    worker_id = len(workers) + i
                    worker_task = asyncio.create_task(
                        self.worker(worker_id, client, progress_bar)
                    )
                    workers.append(worker_task)
                
                self.stats.current_workers = new_worker_count
        
        elif self.should_scale_down():
            new_worker_count = max(
                self.stats.current_workers - 1,
                self.min_workers
            )
            diff = self.stats.current_workers - new_worker_count
            
            if diff > 0:
                logger.info(f"📉 Scaling DOWN: {self.stats.current_workers} → {new_worker_count} workers")
                for _ in range(diff):
                    await self.task_queue.put(None)
                
                self.stats.current_workers = new_worker_count
        
        return workers
    
    def cancel(self) -> None:
        """Cancel the download operation gracefully."""
        self.cancelled = True
    
    async def download_all(
        self,
        tasks: List[DownloadTask],
    ) -> DownloadStats:
        """Download all images with adaptive worker scaling.
        
        Args:
            tasks: List of download tasks to process
            
        Returns:
            Statistics about the download operation
        """
        # Populate task queue
        for task in tasks:
            await self.task_queue.put(task)
        
        # Create HTTP client with connection pooling
        async with httpx.AsyncClient(
            limits=httpx.Limits(
                max_connections=self.max_workers * 2,
                max_keepalive_connections=self.max_workers,
            ),
            http2=True,
        ) as client:
            
            # Create progress bar
            with tqdm(
                total=len(tasks),
                desc="🚀 Downloading images",
                unit="img",
                colour="green",
                bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}] {postfix} ",
            ) as progress_bar:
                
                # Start initial workers
                workers = [
                    asyncio.create_task(self.worker(i, client, progress_bar))
                    for i in range(self.stats.current_workers)
                ]
                
                # Monitor and scale workers
                scale_interval = 2.0  # seconds
                try:
                    while not self.task_queue.empty() and not self.cancelled:
                        await asyncio.sleep(scale_interval)
                        workers = await self.scale_workers(workers, client, progress_bar)
                    
                    # Wait for all tasks to complete
                    if not self.cancelled:
                        await self.task_queue.join()
                except asyncio.CancelledError:
                    logger.info("Download cancelled by user")
                    self.cancelled = True
                finally:
                    # Stop all workers
                    for _ in workers:
                        await self.task_queue.put(None)
                    
                    await asyncio.gather(*workers, return_exceptions=True)
        
        return self.stats


def create_download_tasks(
    images: List[Image],
    output_dir: Path,
) -> tuple[List[DownloadTask], int, int]:
    """Create download tasks from image list.
    
    Args:
        images: List of image objects
        output_dir: Directory to save images
        
    Returns:
        Tuple of (download tasks, already downloaded count, missing URL count)
    """
    tasks: List[DownloadTask] = []
    already_downloaded = 0
    missing_urls = 0
    
    for image in images:
        # Check if URL exists
        if not image.coco_url or image.coco_url.strip() == "":
            logger.warning(f"Image {image.id} ({image.file_name}) has no URL - skipping")
            missing_urls += 1
            continue
            
        output_path = output_dir / image.file_name
        
        # Skip if already exists
        if output_path.exists():
            logger.debug(f"Skipping existing file: {output_path}")
            already_downloaded += 1
            continue
        
        tasks.append(DownloadTask(
            image_id=image.id,
            url=image.coco_url,
            output_path=output_path,
        ))
    
    return tasks, already_downloaded, missing_urls


async def main_async(args: argparse.Namespace) -> int:
    """Async main function.
    
    Args:
        args: Parsed command-line arguments
        
    Returns:
        Exit code (0 for success, 1 for failure)
    """
    # Validate dataset path
    dataset_path = Path(args.dataset_path)
    if not dataset_path.exists():
        print(f"❌ Error: Dataset file not found: {args.dataset_path}")
        print("   Please check the path and try again.")
        return 1
    
    if not dataset_path.is_file():
        print(f"❌ Error: Dataset path is not a file: {args.dataset_path}")
        return 1
    
    # Validate it's valid JSON
    try:
        with open(dataset_path, 'r') as f:
            json.load(f)
    except json.JSONDecodeError as e:
        print(f"❌ Error: Invalid JSON in dataset file: {e}")
        print(f"   Please check that {args.dataset_path} is a valid COCO-format JSON file.")
        return 1
    except Exception as e:
        print(f"❌ Error: Failed to read dataset file: {e}")
        return 1
    
    # Validate output directory
    output_dir = Path(args.output_dir)
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
        # Test write permissions
        test_file = output_dir / ".write_test"
        test_file.touch()
        test_file.unlink()
    except PermissionError:
        print(f"❌ Error: No write permission for output directory: {args.output_dir}")
        print("   Please check permissions or choose a different directory.")
        return 1
    except Exception as e:
        print(f"❌ Error: Cannot create output directory: {e}")
        return 1
    
    # Load dataset
    print(f"📂 Loading dataset from {args.dataset_path}")
    try:
        dataset = ObjectDetectionDataset.load(args.dataset_path)
    except Exception as e:
        print(f"❌ Error: Failed to load dataset: {e}")
        print("   Please ensure the file is in valid COCO format.")
        return 1
    
    print(f"✓ Found {len(dataset.images)} images in dataset")
    
    # Create download tasks
    tasks, already_downloaded, missing_urls = create_download_tasks(dataset.images, output_dir)
    
    # Show resume information
    if already_downloaded > 0:
        print(f"✓ Skipping {already_downloaded} already downloaded images")
    
    if missing_urls > 0:
        print(f"⚠️  Warning: {missing_urls} images have no URL and will be skipped")
    
    if not tasks:
        print(f"\n✅ All images already downloaded to {output_dir}")
        return 0
    
    print(f"\n🚀 Starting download of {len(tasks)} images to {output_dir}")
    print("   Press Ctrl+C to cancel\n")
    
    # Download with adaptive scaling
    downloader = AdaptiveDownloader(
        output_dir=output_dir,
        min_workers=args.min_workers,
        max_workers=args.max_workers,
        initial_workers=min(args.max_workers, max(args.min_workers, args.initial_workers)),
    )
    
    # Set up signal handler for graceful cancellation
    def signal_handler(sig, frame):
        print("\n\n⚠️  Cancellation requested... finishing current downloads")
        downloader.cancel()
    
    signal.signal(signal.SIGINT, signal_handler)
    
    try:
        stats = await downloader.download_all(tasks)
    except KeyboardInterrupt:
        print("\n\n⚠️  Download cancelled by user")
        stats = downloader.stats
    
    # Print final statistics
    print("\n" + "=" * 70)
    if downloader.cancelled:
        print("⚠️  Download Cancelled (partial results)")
    else:
        print("📊 Download Complete!")
    print("=" * 70)
    print(f"✅ Successful: {stats.successful}")
    print(f"❌ Failed: {stats.failed}")
    if stats.retries > 0:
        print(f"🔄 Retries: {stats.retries}")
    print(f"💾 Downloaded: {stats.bytes_downloaded / 1024 / 1024:.2f} MB")
    print(f"📁 Output directory: {output_dir.absolute()}")
    print("=" * 70)
    
    if stats.failed > 0:
        print(f"\n⚠️  {stats.failed} images failed to download.")
        print("   You can re-run the same command to retry failed downloads.")
    
    if downloader.cancelled:
        print("\nℹ️  To resume, run the same command again.")
        print("   Already downloaded images will be skipped.")
        return 130  # Standard exit code for SIGINT
    
    return 0 if stats.failed == 0 else 1


def main() -> int:
    """Main entry point.
    
    Returns:
        Exit code
    """
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    
    parser.add_argument(
        "dataset_path",
        type=str,
        help="Path to COCO-format dataset JSON file",
    )
    
    parser.add_argument(
        "-o", "--output-dir",
        type=str,
        default="./images",
        help="Output directory for downloaded images",
    )
    
    parser.add_argument(
        "--min-workers",
        type=int,
        default=MIN_WORKERS,
        help="Minimum number of concurrent workers",
    )
    
    parser.add_argument(
        "--max-workers",
        type=int,
        default=MAX_WORKERS,
        help="Maximum number of concurrent workers",
    )
    
    parser.add_argument(
        "--initial-workers",
        type=int,
        default=min(MAX_WORKERS, max(MIN_WORKERS, INITIAL_WORKERS)),
        help="Initial number of workers",
    )
    
    args = parser.parse_args()
    
    # Run async main
    return asyncio.run(main_async(args))


if __name__ == "__main__":
    sys.exit(main())


