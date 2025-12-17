"""
FAISS Index Module
Handles vector similarity search using FAISS
"""

import faiss
import numpy as np
import pickle
from typing import List, Tuple, Optional
from pathlib import Path


class FAISSIndex:
    """
    Vector similarity search using FAISS with HNSW index.
    """
    
    def __init__(self, embedding_dim: int = 512):
        """
        Initialize FAISS index.
        
        Args:
            embedding_dim: Dimension of embedding vectors
        """
        self.embedding_dim = embedding_dim
        self.index = None
        self.metadata = []  # Store product metadata
        
    def build_index(self, embeddings: np.ndarray, metadata: List[dict], 
                   index_type: str = "HNSW", M: int = 32, ef_construction: int = 200):
        """
        Build FAISS index from embeddings.
        
        Args:
            embeddings: Array of embeddings (num_items x embedding_dim)
            metadata: List of metadata dictionaries for each item
            index_type: Type of index (HNSW, IVF, Flat)
            M: HNSW parameter - number of connections per layer
            ef_construction: HNSW parameter - search quality during build
        """
        num_items = embeddings.shape[0]
        print(f"\nBuilding FAISS index...")
        print(f"  - Index type: {index_type}")
        print(f"  - Number of items: {num_items}")
        print(f"  - Embedding dimension: {self.embedding_dim}")
        
        if index_type == "HNSW":
            # HNSW index - best for < 1M items
            self.index = faiss.IndexHNSWFlat(self.embedding_dim, M)
            self.index.hnsw.efConstruction = ef_construction
            self.index.hnsw.efSearch = 64  # Search-time parameter
            
        elif index_type == "IVF":
            # IVF index - good for > 1M items
            nlist = min(int(np.sqrt(num_items)), 100)
            quantizer = faiss.IndexFlatL2(self.embedding_dim)
            self.index = faiss.IndexIVFFlat(quantizer, self.embedding_dim, nlist)
            self.index.train(embeddings)
            
        else:  # Flat
            # Brute force - most accurate but slow
            self.index = faiss.IndexFlatL2(self.embedding_dim)
        
        # Add embeddings to index
        self.index.add(embeddings)
        self.metadata = metadata
        
        print(f"✓ Index built successfully")
        print(f"  - Total indexed items: {self.index.ntotal}")
    
    def search(self, query_embedding: np.ndarray, k: int = 10) -> List[dict]:
        """
        Search for similar items.
        
        Args:
            query_embedding: Query embedding vector (1D array)
            k: Number of results to return
            
        Returns:
            List of result dictionaries with metadata and scores
        """
        if self.index is None:
            raise ValueError("Index not built. Call build_index() first.")
        
        # Ensure query is 2D
        if query_embedding.ndim == 1:
            query_embedding = query_embedding.reshape(1, -1)
        
        # Search
        distances, indices = self.index.search(query_embedding, k)
        
        # Format results
        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx != -1:  # Valid result
                result = self.metadata[idx].copy()
                # Convert L2 distance to cosine similarity (embeddings are normalized)
                similarity = 1 - (dist / 2)  # For normalized vectors
                result['similarity_score'] = float(similarity)
                results.append(result)
        
        return results
    
    def search_batch(self, query_embeddings: np.ndarray, k: int = 10) -> List[List[dict]]:
        """
        Search for multiple queries at once.
        
        Args:
            query_embeddings: Array of query embeddings (num_queries x embedding_dim)
            k: Number of results per query
            
        Returns:
            List of result lists, one per query
        """
        distances, indices = self.index.search(query_embeddings, k)
        
        all_results = []
        for query_dists, query_indices in zip(distances, indices):
            results = []
            for dist, idx in zip(query_dists, query_indices):
                if idx != -1:
                    result = self.metadata[idx].copy()
                    similarity = 1 - (dist / 2)
                    result['similarity_score'] = float(similarity)
                    results.append(result)
            all_results.append(results)
        
        return all_results
    
    def save(self, filepath: str):
        """
        Save index and metadata to disk.
        
        Args:
            filepath: Base path for saving (without extension)
        """
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        # Save FAISS index
        index_path = str(filepath) + ".index"
        faiss.write_index(self.index, index_path)
        
        # Save metadata
        metadata_path = str(filepath) + ".pkl"
        with open(metadata_path, 'wb') as f:
            pickle.dump({
                'metadata': self.metadata,
                'embedding_dim': self.embedding_dim
            }, f)
        
        print(f"\n✓ Index saved to {filepath}")
        print(f"  - Index file: {index_path}")
        print(f"  - Metadata file: {metadata_path}")
    
    def load(self, filepath: str):
        """
        Load index and metadata from disk.
        
        Args:
            filepath: Base path for loading (without extension)
        """
        filepath = Path(filepath)
        
        # Load FAISS index
        index_path = str(filepath) + ".index"
        self.index = faiss.read_index(index_path)
        
        # Load metadata
        metadata_path = str(filepath) + ".pkl"
        with open(metadata_path, 'rb') as f:
            data = pickle.load(f)
            self.metadata = data['metadata']
            self.embedding_dim = data['embedding_dim']
        
        print(f"\n✓ Index loaded from {filepath}")
        print(f"  - Total items: {self.index.ntotal}")
        print(f"  - Embedding dim: {self.embedding_dim}")
    
    def get_stats(self) -> dict:
        """Return statistics about the index."""
        return {
            'total_items': self.index.ntotal if self.index else 0,
            'embedding_dim': self.embedding_dim,
            'metadata_count': len(self.metadata)
        }


if __name__ == "__main__":
    # Test the index
    print("Testing FAISS Index...")
    
    # Create dummy embeddings
    num_items = 1000
    embedding_dim = 512
    embeddings = np.random.randn(num_items, embedding_dim).astype('float32')
    # Normalize
    embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
    
    # Create dummy metadata
    metadata = [{'id': i, 'name': f'Product {i}'} for i in range(num_items)]
    
    # Build index
    index = FAISSIndex(embedding_dim)
    index.build_index(embeddings, metadata)
    
    # Test search
    query = np.random.randn(embedding_dim).astype('float32')
    query = query / np.linalg.norm(query)
    results = index.search(query, k=5)
    
    print(f"\nSearch results: {len(results)} items")
    for i, result in enumerate(results, 1):
        print(f"{i}. {result['name']} (similarity: {result['similarity_score']:.4f})")
