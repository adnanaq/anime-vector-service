"""Vector Service Client Library.

This client library provides easy integration with the Anime Vector Service
for applications that need vector database operations.
"""

from .vector_client import VectorServiceClient, VectorServiceError

__all__ = ["VectorServiceClient", "VectorServiceError"]