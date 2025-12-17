"""
CLIP Encoder Module
Handles image and text encoding using OpenAI's CLIP model
"""

import torch
import clip
from PIL import Image
from typing import Union, List
import numpy as np


class CLIPEncoder:
    """
    Wrapper class for CLIP model to generate embeddings for images and text.
    """
    
    def __init__(self, model_name: str = "ViT-B/32", device: str = None):
        """
        Initialize CLIP encoder.
        
        Args:
            model_name: CLIP model variant (ViT-B/32, ViT-B/16, ViT-L/14)
            device: Device to run model on (cuda/cpu). Auto-detects if None.
        """
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Loading CLIP model '{model_name}' on {self.device}...")
        
        self.model, self.preprocess = clip.load(model_name, device=self.device)
        self.model.eval()  # Set to evaluation mode
        
        print(f"✓ CLIP model loaded successfully")
        print(f"  - Embedding dimension: {self.model.visual.output_dim}")
    
    @torch.no_grad()
    def encode_image(self, image: Union[str, Image.Image, np.ndarray]) -> np.ndarray:
        """
        Generate embedding for a single image.
        
        Args:
            image: Image path, PIL Image, or numpy array
            
        Returns:
            Normalized embedding vector (numpy array)
        """
        # Load and preprocess image
        if isinstance(image, str):
            image = Image.open(image).convert('RGB')
        elif isinstance(image, np.ndarray):
            image = Image.fromarray(image).convert('RGB')
        
        image_input = self.preprocess(image).unsqueeze(0).to(self.device)
        
        # Generate embedding
        embedding = self.model.encode_image(image_input)
        embedding = embedding / embedding.norm(dim=-1, keepdim=True)  # Normalize
        
        return embedding.cpu().numpy().astype('float32')[0]
    
    @torch.no_grad()
    def encode_images_batch(self, images: List[Union[str, Image.Image]], batch_size: int = 32) -> np.ndarray:
        """
        Generate embeddings for multiple images in batches.
        
        Args:
            images: List of image paths or PIL Images
            batch_size: Number of images to process at once
            
        Returns:
            Array of normalized embeddings (num_images x embedding_dim)
        """
        all_embeddings = []
        
        for i in range(0, len(images), batch_size):
            batch = images[i:i + batch_size]
            
            # Preprocess batch
            batch_tensors = []
            for img in batch:
                if isinstance(img, str):
                    img = Image.open(img).convert('RGB')
                batch_tensors.append(self.preprocess(img))
            
            batch_input = torch.stack(batch_tensors).to(self.device)
            
            # Generate embeddings
            embeddings = self.model.encode_image(batch_input)
            embeddings = embeddings / embeddings.norm(dim=-1, keepdim=True)
            
            all_embeddings.append(embeddings.cpu().numpy())
        
        return np.vstack(all_embeddings).astype('float32')
    
    @torch.no_grad()
    def encode_text(self, text: str) -> np.ndarray:
        """
        Generate embedding for text query.
        
        Args:
            text: Text description
            
        Returns:
            Normalized embedding vector (numpy array)
        """
        text_input = clip.tokenize([text]).to(self.device)
        
        embedding = self.model.encode_text(text_input)
        embedding = embedding / embedding.norm(dim=-1, keepdim=True)
        
        return embedding.cpu().numpy().astype('float32')[0]
    
    @torch.no_grad()
    def encode_texts_batch(self, texts: List[str]) -> np.ndarray:
        """
        Generate embeddings for multiple text queries.
        
        Args:
            texts: List of text descriptions
            
        Returns:
            Array of normalized embeddings
        """
        text_inputs = clip.tokenize(texts).to(self.device)
        
        embeddings = self.model.encode_text(text_inputs)
        embeddings = embeddings / embeddings.norm(dim=-1, keepdim=True)
        
        return embeddings.cpu().numpy().astype('float32')
    
    def get_embedding_dim(self) -> int:
        """Return the dimensionality of embeddings."""
        return self.model.visual.output_dim


if __name__ == "__main__":
    # Test the encoder
    encoder = CLIPEncoder()
    
    # Test text encoding
    text_embedding = encoder.encode_text("a red shirt")
    print(f"\nText embedding shape: {text_embedding.shape}")
    print(f"Text embedding norm: {np.linalg.norm(text_embedding):.4f}")
