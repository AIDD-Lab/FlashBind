"""Unified resource loader for directory and LMDB access."""

import io
import json
import lmdb
import torch
import pickle
from pathlib import Path
from typing import Optional, Any


class ResourceType:
    """Resource type enumeration."""
    DIRECTORY = "directory"
    LMDB = "lmdb"
    JSON = "json"
    PT = "pt"


class ResourceNotFoundError(Exception):
    """Raised when a requested resource is not found."""
    pass


class ResourceLoader:
    """Unified interface for accessing files from directories or LMDB databases."""
    
    def __init__(self, path: Path, extension: str = ""):
        """Initialize the resource loader.
        
        Args:
            path: Path to directory or LMDB file/directory
        """
        self.path = path
        self.resource_type = None
        self.data = None
        self.extension = extension
        if not path.exists():
            raise FileNotFoundError(f"Resource path does not exist: {path}")
        
        # Determine if this is an LMDB or directory
        self._setup_loader()
    
    def _setup_loader(self):
        """Setup the appropriate loader based on path type."""
        # Check resource type and load data accordingly
        if self.path.suffix == '.json':
            # JSON file
            self.resource_type = ResourceType.JSON
            self.data = json.load(self.path.open('r'))
            print(f"Initialized JSON resource loader: {self.path}")
        elif self.path.suffix == '.pt':
            # PT file
            self.resource_type = ResourceType.PT
            self.data = torch.load(self.path, map_location='cpu')
            print(f"Initialized PT resource loader: {self.path}")
        elif (self.path.suffix == '.lmdb' or 
              (self.path.is_dir() and (self.path / 'data.mdb').exists())):
            # LMDB
            try:
                self.resource_type = ResourceType.LMDB
                # Determine if this is a file-based LMDB or directory-based LMDB
                is_file_lmdb = self.path.is_file() and self.path.suffix == '.lmdb'
                subdir = not is_file_lmdb  # False for file LMDB, True for directory LMDB
                self.data = lmdb.open(str(self.path), readonly=True, lock=False, subdir=subdir)
                print(f"Initialized LMDB resource loader: {self.path}")
            except Exception as e:
                raise RuntimeError(f"Failed to open LMDB at {self.path}: {e}")
        elif self.path.is_dir():
            # Regular directory
            self.resource_type = ResourceType.DIRECTORY
            self.data = self.path
            print(f"Initialized directory resource loader: {self.path}")
        else:
            raise ValueError(f"Invalid path type: {self.path}")
    
    def get(self, key: str) -> Any:
        """Get resource content by key.
        
        Args:
            key: Resource identifier (filename without extension for directories, 
                 key for LMDB/JSON/PT)
            
        Returns:
            Resource content (string, dict, tensor, etc.)
            
        Raises:
            ResourceNotFoundError: If resource is not found
        """
        if self.resource_type == ResourceType.LMDB:
            return self._get_from_lmdb(key)
        elif self.resource_type == ResourceType.JSON:
            return self._get_from_json(key)
        elif self.resource_type == ResourceType.PT:
            return self._get_from_pt(key)
        elif self.resource_type == ResourceType.DIRECTORY:
            return self._get_from_directory(key)
        else:
            raise ValueError(f"Unknown resource type: {self.resource_type}")
    
    def _get_from_lmdb(self, key: str) -> Any:
        """Get content from LMDB using pickle for universal serialization."""
        try:
            with self.data.begin() as txn:
                content = txn.get(key.encode('utf-8'))
                if content is not None:
                    # Try pickle first, then string decode as fallback
                    try:
                        return pickle.loads(content)
                    except:
                        return content.decode('utf-8')
                else:
                    raise ResourceNotFoundError(f"Key '{key}' not found in LMDB")
        except Exception as e:
            if isinstance(e, ResourceNotFoundError):
                raise
            raise RuntimeError(f"Failed to retrieve key '{key}' from LMDB: {e}")
    
    def _get_from_json(self, key: str) -> Any:
        """Get content from JSON data."""
        if key not in self.data:
            raise ResourceNotFoundError(f"Key '{key}' not found in JSON data")
        return self.data[key]
    
    def _get_from_pt(self, key: str) -> Any:
        """Get content from PT data."""
        if key not in self.data:
            raise ResourceNotFoundError(f"Key '{key}' not found in PT data")
        return self.data[key]
    
    def _get_from_directory(self, key: str) -> Any:
        """Get content from directory."""
        filename = f"{key}{self.extension}"
        file_path = self.data / filename
        
        if not file_path.exists():
            raise ResourceNotFoundError(f"File '{filename}' not found in directory {self.data}")
        
        try:
            with file_path.open('r', encoding='utf-8') as f:
                return f.read()
        except Exception as e:
            raise RuntimeError(f"Failed to read file '{filename}': {e}")
    
    def exists(self, key: str) -> bool:
        """Check if resource exists.
        
        Args:
            key: Resource identifier
            
        Returns:
            True if resource exists, False otherwise
        """
        try:
            self.get(key)
            return True
        except ResourceNotFoundError:
            return False
        except Exception:
            return False
    
    def close(self):
        """Close the resource loader and clean up resources."""
        if self.resource_type == ResourceType.LMDB and self.data:
            self.data.close()
        self.data = None
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close() 