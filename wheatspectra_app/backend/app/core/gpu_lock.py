"""
PhytoSpectra-VLM / WheatSpectraChat
Asyncio GPU / Inference lock to prevent concurrent OOM.
"""

import asyncio

# Global lock for GPU / heavy inference tasks
_gpu_lock: asyncio.Lock = asyncio.Lock()


def get_gpu_lock() -> asyncio.Lock:
    return _gpu_lock
