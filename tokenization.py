# coding=utf-8
# Copyright 2024 The HuggingFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
R-BPE Tokenizer adapter for transformers library.

This is a lightweight adapter that requires the standalone `rbpe` package.
Install it with: pip install rbpe
"""

from typing import Any, Dict, List, Optional
from transformers import PreTrainedTokenizerBase

# Try to import the rbpe package
try:
    from rbpe import RBPETokenizer as _RBPETokenizerImpl  # type: ignore
    RBPE_AVAILABLE = True
except ImportError:
    RBPE_AVAILABLE = False
    _RBPETokenizerImpl = None  # type: ignore


VOCAB_FILES_NAMES = {"vocab_file": "tokenizer.json"}


class RBPETokenizer(PreTrainedTokenizerBase):
    """
    R-BPE (Reusable BPE) Tokenizer for adapting existing BPE tokenizers to better support target languages.
    
    This is an adapter class that requires the `rbpe` package to be installed separately.
    The R-BPE framework reuses tokens from excluded languages and creates ID-based mappings
    for the target language tokens.
    
    Installation:
        ```bash
        pip install rbpe
        ```
        
        Or install transformers with rbpe support:
        ```bash
        pip install transformers[rbpe]
        ```
    
    Args:
        model_id (`str`, *optional*):
            The HuggingFace model id of the base tokenizer to adapt.
        training_data_dir (`str`, *optional*):
            Directory containing training data for creating a new R-BPE tokenizer.
        clean_data (`bool`, *optional*, defaults to `True`):
            Whether to clean the training data before training.
        cleaned_data_dir (`str`, *optional*):
            Directory to save cleaned training data.
        hf_token (`str`, *optional*):
            HuggingFace API token for downloading models.
        min_reusable_count (`int`, *optional*, defaults to 20000):
            Minimum number of tokens needed for reuse.
        target_language_scripts (`List[str]`, *optional*, defaults to `["arabic"]`):
            List of Unicode script names for the target language.
        preserved_languages_scripts (`List[str]`, *optional*, defaults to `["latin", "greek"]`):
            List of Unicode script names for languages to preserve.
        special_tokens (`dict`, *optional*):
            Dictionary of custom special tokens.
        additional_special_tokens (`List[str]`, *optional*):
            List of additional special tokens.
        apply_rbpe_arabic_norm (`bool`, *optional*, defaults to `True`):
            Whether to apply R-BPE Arabic normalization during encoding.
    
    Example:
        ```python
        >>> from transformers import RBPETokenizer
        >>> 
        >>> # Create a new R-BPE tokenizer
        >>> tokenizer = RBPETokenizer(
        ...     model_id="meta-llama/Llama-3.1-8B",
        ...     training_data_dir="./arabic_data",
        ...     hf_token="YOUR_TOKEN",
        ...     target_language_scripts=["arabic"],
        ... )
        >>> tokenizer.prepare()
        >>> 
        >>> # Save for later use
        >>> tokenizer.save_pretrained("./my_rbpe_tokenizer")
        >>> 
        >>> # Load a pre-trained R-BPE tokenizer
        >>> tokenizer = RBPETokenizer.from_pretrained("./my_rbpe_tokenizer")
        >>> 
        >>> # Use it like any other tokenizer
        >>> encoded = tokenizer("مرحبا")
        >>> decoded = tokenizer.decode(encoded["input_ids"])
        ```
    
    For more information, see: https://github.com/YOUR_ORG/rbpe
    """

    vocab_files_names = VOCAB_FILES_NAMES
    model_input_names = ["input_ids", "attention_mask"]

    def __init__(self, *args, **kwargs):
        """
        Initialize the R-BPE tokenizer adapter.
        
        Raises:
            ImportError: If the `rbpe` package is not installed.
        """
        if not RBPE_AVAILABLE:
            raise ImportError(
                "RBPETokenizer requires the `rbpe` package. "
                "Install it with: pip install rbpe\n"
                "Or install transformers with rbpe support: pip install transformers[rbpe]"
            )
        
        # Initialize the actual R-BPE tokenizer implementation
        if _RBPETokenizerImpl is not None:
            self._rbpe_tokenizer = _RBPETokenizerImpl(*args, **kwargs)
        else:
            self._rbpe_tokenizer = None
    
    def prepare(self):
        """
        Prepare a new R-BPE tokenizer from scratch.
        
        This orchestrates the complete tokenizer preparation process:
        1. Classifies tokens using TokenClassifier
        2. Cleans data using DataCleaner (if needed)
        3. Trains new tokenizer using BPETokenizerTrainer
        4. Creates mappings using MappingTokenizer
        5. Returns final RBPETokenizer instance
        
        Returns:
            The prepared tokenizer instance
        """
        if self._rbpe_tokenizer is not None:
            return self._rbpe_tokenizer.prepare()
        raise RuntimeError("R-BPE tokenizer not properly initialized")
    
    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: str, *args, **kwargs):
        """
        Load a pre-trained R-BPE tokenizer.
        
        Args:
            pretrained_model_name_or_path (`str`):
                Path to the directory containing the saved R-BPE tokenizer,
                or a model identifier on the Hugging Face Hub.
        
        Returns:
            The loaded tokenizer instance
        
        Raises:
            ImportError: If the `rbpe` package is not installed.
        """
        if not RBPE_AVAILABLE:
            raise ImportError(
                "RBPETokenizer requires the `rbpe` package. "
                "Install it with: pip install rbpe\n"
                "Or install transformers with rbpe support: pip install transformers[rbpe]"
            )
        
        # Load using the rbpe package's from_pretrained method
        if _RBPETokenizerImpl is not None:
            return _RBPETokenizerImpl.from_pretrained(pretrained_model_name_or_path, *args, **kwargs)
        raise RuntimeError("R-BPE package not available")
    
    @classmethod
    def from_config(cls, config_path: str):
        """
        Initialize an R-BPE tokenizer from a YAML config file.
        
        Args:
            config_path (`str`): Path to YAML config file
            
        Returns:
            RBPETokenizer: Initialized tokenizer instance
            
        Raises:
            ImportError: If the `rbpe` package is not installed.
        """
        if not RBPE_AVAILABLE:
            raise ImportError(
                "RBPETokenizer requires the `rbpe` package. "
                "Install it with: pip install rbpe"
            )
        
        if _RBPETokenizerImpl is not None:
            return _RBPETokenizerImpl.from_config(config_path)
        raise RuntimeError("R-BPE package not available")
    
    def __getattr__(self, name: str) -> Any:
        """
        Delegate all other method calls to the underlying R-BPE tokenizer implementation.
        
        This allows the adapter to expose all methods from the rbpe package
        without explicitly wrapping each one.
        """
        if self._rbpe_tokenizer is None:
            raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")
        return getattr(self._rbpe_tokenizer, name)
    
    def __call__(self, *args, **kwargs):
        """Allow the tokenizer to be called directly."""
        if self._rbpe_tokenizer is not None:
            return self._rbpe_tokenizer(*args, **kwargs)
        raise RuntimeError("R-BPE tokenizer not properly initialized")
    
    def __repr__(self) -> str:
        """Return a string representation of the tokenizer."""
        if RBPE_AVAILABLE and self._rbpe_tokenizer is not None:
            return repr(self._rbpe_tokenizer)
        return f"RBPETokenizer(rbpe_available={RBPE_AVAILABLE})"