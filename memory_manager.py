import json
import datetime
from typing import Dict, Any, List
from pathlib import Path

class MemoryManager:
    def __init__(self, file_path: str = "long_term_memory.json"):
        self.file_path = Path(file_path)
        self.initialize_memory_file()
        
    def initialize_memory_file(self):
        """Create memory file if it doesn't exist"""
        if not self.file_path.exists():
            default_data = {
                "1": [],
                "metadata": {
                    "created_at": datetime.datetime.now().isoformat(),
                    "version": "1.0"
                }
            }
            self.save_memory(default_data)
    
    def save_memory(self, data: Dict[str, Any]) -> bool:
        """Save memory to file"""
        try:
            with open(self.file_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=4, ensure_ascii=False)
            return True
        except Exception as e:
            print(f"Error saving memory: {e}")
            return False
    
    def load_memory(self) -> Dict[str, Any]:
        """Load memory from file"""
        try:
            with open(self.file_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception:
            return {"1": [], "metadata": {"created_at": datetime.datetime.now().isoformat()}}
    
    def add_memory(self, user_id: str, content: Any) -> bool:
        """Add new memory entry"""
        try:
            memory_data = self.load_memory()
            if user_id not in memory_data:
                memory_data[user_id] = []
                
            memory_entry = {
                "content": content,
                "timestamp": datetime.datetime.now().isoformat(),
                "id": len(memory_data[user_id]) + 1
            }
            
            memory_data[user_id].append(memory_entry)
            return self.save_memory(memory_data)
        except Exception as e:
            print(f"Error adding memory: {e}")
            return False
