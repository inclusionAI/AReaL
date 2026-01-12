# Shamelessly stolen from Microsoft Autogen team: thanks to them for this great resource!
# https://github.com/microsoft/autogen/blob/gaia_multiagent_v01_march_1st/autogen/browser_utils.py
import atexit
from collections import OrderedDict
import hashlib
import json
import os
import threading
import time
from typing import Any, Dict, Optional

       
class WebPageCache:
    
    def __init__(self, max_size: int = 100000, cache_file: str = "./webpage_cache.json", save_interval: int = 10):
        self.max_size = max_size
        self.cache_file = cache_file
        self.cache = OrderedDict()
        self.lock = threading.Lock()
        self.stats = {"hits": 0, "misses": 0, "evictions": 0}
        self.save_interval = save_interval
        self.operations_since_save = 0
        
        self.load_from_file()
        
        atexit.register(self.save_to_file)
    
    def _generate_cache_key(self, url: str) -> str:
        return hashlib.md5(url.encode()).hexdigest()
    
    def put(self, url: str, content: str):
        if not url or not content:
            return
            
        cache_key = self._generate_cache_key(url)
        
        with self.lock:
            if cache_key in self.cache:
                del self.cache[cache_key]
            
            while len(self.cache) >= self.max_size:
                self.cache.popitem(last=False)
                self.stats["evictions"] += 1
            
            self.cache[cache_key] = {
                "url": url,
                "content": content,
                "timestamp": time.time()
            }
            
            self.operations_since_save += 1
            if self.operations_since_save >= self.save_interval:
                self.operations_since_save = 0
                import threading
                threading.Thread(target=self._background_save, daemon=True).start()
    
    def get(self, url: str) -> Optional[str]:
        cache_key = self._generate_cache_key(url)
        
        with self.lock:
            if cache_key in self.cache:
                # 移动到末尾（最近使用）
                entry = self.cache.pop(cache_key)
                self.cache[cache_key] = entry
                self.stats["hits"] += 1
                return entry["content"]
            else:
                self.stats["misses"] += 1
                return None
    
    def has(self, url: str) -> bool:
        cache_key = self._generate_cache_key(url)
        with self.lock:
            return cache_key in self.cache
    
    def clear(self):
        with self.lock:
            self.cache.clear()
            self.stats = {"hits": 0, "misses": 0, "evictions": 0}
            self.operations_since_save = 0
    
    def force_save(self):
        self.save_to_file()
        self.operations_since_save = 0
    
    def get_stats(self) -> Dict[str, Any]:
        with self.lock:
            total_requests = self.stats["hits"] + self.stats["misses"]
            hit_rate = self.stats["hits"] / total_requests if total_requests > 0 else 0
            
            return {
                "cache_size": len(self.cache),
                "max_size": self.max_size,
                "hits": self.stats["hits"],
                "misses": self.stats["misses"],
                "evictions": self.stats["evictions"],
                "hit_rate": hit_rate,
                "total_requests": total_requests
            }
    
    def _background_save(self):
        try:
            self.save_to_file()
        except Exception as e:
            print(f"[ERROR] WebPageCache: Background save failed: {e}")

    def save_to_file(self):
        try:
            with self.lock:
                ordered_cache = []
                for key, value in self.cache.items():
                    ordered_cache.append((key, value))
                
                cache_data = {
                    "cache_ordered": ordered_cache,
                    "stats": self.stats,
                    "max_size": self.max_size,
                    "saved_at": time.time()
                }
            
            with open(self.cache_file, 'w', encoding='utf-8') as f:
                json.dump(cache_data, f, indent=2, ensure_ascii=False)
            
            print(f"[DEBUG] WebPageCache: Saved {len(self.cache)} entries to {self.cache_file}")
            
        except Exception as e:
            print(f"[ERROR] WebPageCache: Failed to save cache to {self.cache_file}: {e}")
    
    def load_from_file(self):
        """从JSON文件加载缓存"""
        if not os.path.exists(self.cache_file):
            print(f"[DEBUG] WebPageCache: No existing cache file {self.cache_file}, starting fresh")
            return
        
        try:
            with open(self.cache_file, 'r', encoding='utf-8') as f:
                cache_data = json.load(f)
            
            with self.lock:
                if "cache_ordered" in cache_data:
                    ordered_cache = cache_data["cache_ordered"]
                    self.cache = OrderedDict(ordered_cache)
                    print(f"[DEBUG] WebPageCache: Loaded ordered cache format")
                else:
                    loaded_cache = cache_data.get("cache", {})
                    self.cache = OrderedDict(loaded_cache)
                    print(f"[DEBUG] WebPageCache: Loaded legacy cache format (LRU order may be lost)")
                
                self.stats = cache_data.get("stats", {"hits": 0, "misses": 0, "evictions": 0})
                
                while len(self.cache) > self.max_size:
                    self.cache.popitem(last=False)
                    self.stats["evictions"] += 1
            
            saved_at = cache_data.get("saved_at", 0)
            saved_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(saved_at))
            
            print(f"[DEBUG] WebPageCache: Loaded {len(self.cache)} entries from {self.cache_file} (saved at {saved_time})")
            
        except Exception as e:
            print(f"[ERROR] WebPageCache: Failed to load cache from {self.cache_file}: {e}")
            with self.lock:
                self.cache = OrderedDict()
                self.stats = {"hits": 0, "misses": 0, "evictions": 0}
