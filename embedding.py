import os
import numpy as np
from typing import List, Union
import logging
import requests
import time
from tenacity import retry, stop_after_attempt, wait_exponential

logger = logging.getLogger()

class NuwaEmbeddings:
    """
    A direct implementation of embeddings using OpenAI API.
    """

    def __init__(
        self,
        embedding_model: str = "text-embedding-3-large",
        api_key: Union[str, None] = None,
        chunk_size: int = 1000,
        show_progress_bar: bool = False,
        api_base: str = "https://api.nuwaapi.com/v1/embeddings",
        end_point: str = None,
        verbose: bool = False,
        tokenization_model_name: str = None,
    ) -> None:
        """
        Initializes the NuwaEmbeddings object.

        Args:
            embedding_model (str): The model to use for embedding.
            api_key (str): The API key for OpenAI API.
            chunk_size (int): The maximum number of tokens to send at one time.
            show_progress_bar (bool): Whether to show progress bar during embedding.
            api_base (str): The API endpoint URL.
            end_point (str): Alias for api_base.
            verbose (bool): Whether to show detailed logs.
            tokenization_model_name (str): The model name for tokenization.
        """
        self.model = embedding_model
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("No OpenAI API key provided. Please set OPENAI_API_KEY environment variable.")
            
        self.chunk_size = chunk_size
        self.show_progress_bar = verbose if verbose is not None else show_progress_bar
        self.api_base = end_point if end_point is not None else api_base
        self.tokenization_model_name = tokenization_model_name or embedding_model

        # Log configuration
        logger.info(f"[DEBUG] Initializing embeddings with:")
        logger.info(f"[DEBUG] - Model: {self.model}")
        logger.info(f"[DEBUG] - API Base: {self.api_base}")
        logger.info(f"[DEBUG] - API Key: {'*' * 8 + self.api_key[-4:] if self.api_key else 'None'}")
        logger.info(f"[DEBUG] - Chunk Size: {self.chunk_size}")

    def get_embedding_dimension(self) -> int:
        """
        Returns the dimension of the embedding.

        Returns:
            int: The dimension of the embedding.
        """
        match self.model:
            case "text-embedding-002":
                return 1536
            case "text-embedding-ada-002":
                return 1536
            case "text-embedding-3-large":
                return 3072
            case _:
                raise NotImplementedError(
                    f"Embedding dimension for model {self.model} not implemented"
                )

    @retry(
        stop=stop_after_attempt(5),
        wait=wait_exponential(multiplier=1, min=4, max=30),
        reraise=True
    )
    def _get_embedding(self, text: str) -> List[float]:
        """
        Get embedding for a single text using OpenAI API.

        Args:
            text (str): The text to embed.

        Returns:
            List[float]: The embedding vector.
        """
        logger.info(f"[DEBUG] Preparing API request for text: {text[:100]}...")
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        data = {
            "model": self.model,
            "input": text
        }
        
        logger.info(f"[DEBUG] API Request details:")
        logger.info(f"[DEBUG] - URL: {self.api_base}")
        logger.info(f"[DEBUG] - Model: {self.model}")
        logger.info(f"[DEBUG] - Headers: {headers}")
        logger.info(f"[DEBUG] - Data: {data}")
        
        try:
            logger.info("[DEBUG] Sending API request...")
            response = requests.post(
                self.api_base,
                headers=headers,
                json=data,
                timeout=(10, 30)
            )
            
            if response.status_code != 200:
                logger.error(f"[DEBUG] API returned non-200 status code: {response.status_code}")
                logger.error(f"[DEBUG] Response content: {response.text}")
                response.raise_for_status()
                
            result = response.json()
            logger.info(f"[DEBUG] API Response data: {result}")
            
            if "data" not in result or not result["data"]:
                logger.error("[DEBUG] No data in API response")
                raise ValueError("No data in API response")
                
            embedding = result["data"][0]["embedding"]
            logger.info(f"[DEBUG] Generated embedding of length: {len(embedding)}")
            return embedding
            
        except requests.exceptions.Timeout as e:
            logger.error(f"[DEBUG] Request timeout: {str(e)}")
            raise
        except requests.exceptions.RequestException as e:
            logger.error(f"[DEBUG] Request error: {str(e)}")
            logger.error(f"[DEBUG] Error type: {type(e)}")
            raise
        except ValueError as e:
            logger.error(f"[DEBUG] Value error: {str(e)}")
            raise
        except Exception as e:
            logger.error(f"[DEBUG] Unexpected error: {str(e)}")
            logger.error(f"[DEBUG] Error type: {type(e)}")
            raise

    def embed_documents(self, texts: List[str], chunk_size: int = 32) -> List[List[float]]:
        """
        Embed a list of texts using OpenAI API in batches.

        Args:
            texts (List[str]): List of texts to embed.
            chunk_size (int, optional): Batch size for each API request. Default is 32.

        Returns:
            List[List[float]]: List of embeddings.
        """
        logger.info(f"[DEBUG] Starting embed_documents with {len(texts)} texts")
        logger.info(f"[DEBUG] First text sample: {texts[0][:100] if texts else 'None'}")
        
        if not texts:
            logger.warning("[DEBUG] Empty texts list provided")
            return []
        
        all_embeddings = []
        try:
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            for i in range(0, len(texts), chunk_size):
                batch = texts[i:i+chunk_size]
                data = {
                    "model": self.model,
                    "input": batch
                }
                logger.info(f"[DEBUG] Sending batch {i//chunk_size+1}: {len(batch)} texts")
                
                @retry(
                    stop=stop_after_attempt(5),
                    wait=wait_exponential(multiplier=1, min=4, max=30),
                    reraise=True
                )
                def make_request():
                    response = requests.post(
                        self.api_base,
                        headers=headers,
                        json=data,
                        timeout=(10, 60)
                    )
                    if response.status_code != 200:
                        logger.error(f"[DEBUG] API returned non-200 status code: {response.status_code}")
                        logger.error(f"[DEBUG] Response content: {response.text}")
                        response.raise_for_status()
                    return response
                
                response = make_request()
                result = response.json()
                logger.info(f"[DEBUG] API Response data: {result}")
                
                if "data" not in result or not result["data"]:
                    logger.error("[DEBUG] No data in API response")
                    raise ValueError("No data in API response")
                    
                batch_embeddings = [item["embedding"] for item in result["data"]]
                all_embeddings.extend(batch_embeddings)
                logger.info(f"[DEBUG] Successfully generated {len(batch_embeddings)} embeddings for batch {i//chunk_size+1}")
                
                if i + chunk_size < len(texts):
                    time.sleep(1)
                    
            logger.info(f"[DEBUG] Successfully generated {len(all_embeddings)} embeddings in total")
            return all_embeddings
        except Exception as e:
            logger.error(f"[DEBUG] Error generating embeddings: {str(e)}")
            logger.error(f"[DEBUG] Error type: {type(e)}")
            logger.error(f"[DEBUG] Error details: {e.__dict__ if hasattr(e, '__dict__') else 'No details available'}")
            return []

    def embed_query(self, text: str) -> List[float]:
        """
        Embed a single text using OpenAI API.

        Args:
            text (str): Text to embed.

        Returns:
            List[float]: The embedding vector.
        """
        logger.info("[DEBUG] Starting embed_query")
        logger.info(f"[DEBUG] Text sample: {text[:100] if text else 'None'}")
        
        if not text or not text.strip():
            logger.warning("[DEBUG] Empty text provided")
            return []
            
        try:
            embedding = self._get_embedding(text)
            logger.info("[DEBUG] Successfully generated embedding")
            return embedding
        except Exception as e:
            logger.error(f"[DEBUG] Error generating embedding: {str(e)}")
            return []

class NuwaLongerThanContextEmb:
    """
    A wrapper class for handling longer texts.
    """

    def __init__(
        self,
        openai_api_key: Union[str, None] = None,
        embedding_model: str = "text-embedding-3-large",
        chunk_size: int = 5000,
        verbose: bool = False,
        api_base: str = "https://api.nuwaapi.com/v1/embeddings",
    ) -> None:
        """
        Initialize the NuwaLongerThanContextEmb object.

        Args:
            openai_api_key (str): OpenAI API key.
            embedding_model (str): Model to use for embeddings.
            chunk_size (int): Size of text chunks.
            verbose (bool): Whether to show progress.
            api_base (str): API base URL.
        """
        self.embeddings = NuwaEmbeddings(
            embedding_model=embedding_model,
            api_key=openai_api_key,
            chunk_size=chunk_size,
            verbose=verbose,
            api_base=api_base
        )

    def _emb(self, text: Union[List[str], str]) -> List[List[float]]:
        """
        Generate embeddings for text(s).

        Args:
            text (Union[List[str], str]): Text or list of texts to embed.

        Returns:
            List[List[float]]: List of embedding vectors.
        """
        if isinstance(text, str):
            return [self.embeddings.embed_query(text)]
        return self.embeddings.embed_documents(text)

    def __call__(self, text: Union[List[str], str]) -> np.ndarray:
        """
        Call the embedding function.

        Args:
            text (Union[List[str], str]): Text or list of texts to embed.

        Returns:
            np.ndarray: Array of embedding vectors.
        """
        embeddings = self._emb(text)
        return np.array(embeddings)

    def get_embedding_dimension(self) -> int:
        """
        Get the dimension of the embeddings.

        Returns:
            int: The dimension of the embeddings.
        """
        return self.embeddings.get_embedding_dimension()