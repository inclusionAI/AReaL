import os
import re
import random
import sqlite3
import time
import itertools
import sys
from collections import defaultdict
from contextlib import contextmanager
from typing import (
    Tuple, Any, List, Set, Dict, Optional, Iterator
)

from func_timeout import func_timeout, FunctionTimedOut
import pandas as pd
import sqlparse

# --- Constants ---
DEFAULT_TIMEOUT_MS = 15000  # 15 seconds
QueryResultRow = Tuple[Any, ...]

# --- Utility Functions ---

def extract_sql_from_markdown(text: str) -> str:
    """
    Extracts the last SQL code block from a markdown-formatted string.
    Falls back to cleaning the original text if no markdown blocks found.
    """
    # Try SQL code blocks
    program_pattern = r"```sql[ \t]*[\r\n]+(.*?)[\r\n]+[ \t]*```"
    matches = re.findall(program_pattern, text, re.DOTALL | re.IGNORECASE)
    if matches:
        query = matches[-1].strip()
        return query.replace('> =', '>=').replace('< =', '<=').replace('! =', '!=')
    
    # Try <sql>...</sql> tags
    sql_tag_pattern = r"<sql>(.*?)</sql>"
    matches = re.findall(sql_tag_pattern, text, re.DOTALL | re.IGNORECASE)
    if matches:
        query = matches[-1].strip()
        return query.replace('> =', '>=').replace('< =', '<=').replace('! =', '!=')
    
    # Try <solution>...</solution> tags for final turn compatibility
    solution_pattern = r"<solution>(.*?)</solution>"
    matches = re.findall(solution_pattern, text, re.DOTALL | re.IGNORECASE)
    if matches:
        query = matches[-1].strip()
        return query.replace('> =', '>=').replace('< =', '<=').replace('! =', '!=')
    
    # Fallback: clean the original text
    return text.replace('> =', '>=').replace('< =', '<=').replace('! =', '!=')


def replace_current_year(query: str) -> str:
    """
    Replaces YEAR(CURDATE()) with a fixed year (2020) for consistent evaluation.
    """
    return re.sub(
        r"YEAR\s*\(\s*CURDATE\s*\(\s*\)\s*\)", "2020", query, flags=re.IGNORECASE
    )


# --- Database Manager Class ---

class DatabaseManager:
    """
    Manages SQLite database connections and query execution with timeouts.
    """
    def __init__(self):
        self._connection_pool: Dict[str, sqlite3.Connection] = {}

    @contextmanager
    def _connection(self, db_path: str) -> Iterator[sqlite3.Connection]:
        """Provides a database connection from the pool."""
        if db_path not in self._connection_pool:
            try:
                # Use immutable=1 for read-only access, which is safer and faster.
                uri_path = f"file:{db_path}?immutable=1"
                conn = sqlite3.connect(uri_path, uri=True, check_same_thread=False)
                # Performance and cleanup pragmas
                conn.execute('PRAGMA journal_mode=DELETE;')  # Avoid WAL files
                conn.execute('PRAGMA synchronous=OFF;')
                conn.execute('PRAGMA temp_store=MEMORY;')
                conn.text_factory = lambda b: b.decode(errors="ignore")
                self._connection_pool[db_path] = conn
            except sqlite3.Error as e:
                raise ConnectionError(f"Failed to connect to database at {db_path}: {e}")

        db_conn = self._connection_pool[db_path]
        yield db_conn

    @staticmethod
    @contextmanager
    def _query_timeout(conn: sqlite3.Connection, timeout_ms: int):
        """A context manager to enforce a timeout on a query."""
        deadline = time.perf_counter() + (timeout_ms / 1000)
        
        def handler():
            if time.perf_counter() >= deadline:
                return 1  # Returning 1 interrupts the query
            return 0

        # The progress handler is checked every N virtual machine instructions.
        # Set a low N for short timeouts.
        n_instructions = 100 if timeout_ms <= 100 else 1000
        conn.set_progress_handler(handler, n_instructions)
        try:
            yield
        finally:
            conn.set_progress_handler(None, n_instructions)
            
    def execute_query(
        self,
        db_path: str,
        query: str,
        timeout_ms: int = DEFAULT_TIMEOUT_MS
    ) -> Tuple[Optional[str], Optional[List[QueryResultRow]]]:
        """
        Executes a SQL query against the specified database.

        Args:
            db_path: Path to the SQLite database file.
            query: The SQL query string to execute.
            timeout_ms: Timeout in milliseconds.

        Returns:
            A tuple (error_message, results). If execution is successful,
            error_message is None. If it fails, results is None.
        """
        clean_query = replace_current_year(query)
        
        try:
            with self._connection(db_path) as conn:
                with self._query_timeout(conn, timeout_ms):
                    cursor = conn.cursor()
                    conn.execute("BEGIN TRANSACTION;")
                    cursor.execute(clean_query)
                    results = cursor.fetchall()
                    conn.rollback()  # Always rollback to avoid any changes
                    cursor.close()
                    return None, results
        except sqlite3.OperationalError as e:
            if "interrupted" in str(e):
                return "SQL Timeout", None
            return f"Error executing SQL: {e}", None
        except Exception as e:
            return f"Error executing SQL: {e}", None

    def close_all_connections(self):
        """Closes all active connections in the pool and cleans up WAL files."""
        for db_path, conn in self._connection_pool.items():
            try:
                # Force WAL checkpoint and cleanup
                conn.execute("PRAGMA wal_checkpoint(TRUNCATE);")
                conn.execute("VACUUM;")
            except Exception:
                pass  # Ignore cleanup errors
            finally:
                conn.close()
        self._connection_pool = {}


# --- Execution-Based Evaluation Class ---

class ExecutionEvaluator:
    """
    Compares two lists of query results for equivalence.
    """
    
    @staticmethod
    def are_results_equivalent(
        result1: List[QueryResultRow],
        result2: List[QueryResultRow],
        order_matters: bool = False
    ) -> bool:
        """
        Checks if two query results are equivalent.
        """
        if len(result1) != len(result2):
            return False
        if not result1:  # Both are empty
            return True
        if len(result1[0]) != len(result2[0]):
            return False

        # Quick rejection test
        s1 = {tuple(sorted(row, key=lambda x: str(x) + str(type(x)))) for row in result1}
        s2 = {tuple(sorted(row, key=lambda x: str(x) + str(type(x)))) for row in result2}
        if s1 != s2:
            return False
            
        if order_matters:
            return result1 == result2
            
        # Permutation check for column order independence
        num_cols = len(result1[0])
        col_sets1 = [{row[i] for row in result1} for i in range(num_cols)]
        
        possible_perms = ExecutionEvaluator._get_constrained_permutations(col_sets1, result2)
        
        for perm in possible_perms:
            if len(perm) != len(set(perm)):
                continue
            
            result2_permuted = [
                tuple(element[i] for i in perm) for element in result2
            ]

            if ExecutionEvaluator._are_multisets_equal(result1, result2_permuted):
                return True
                
        return False

    @staticmethod
    def _are_multisets_equal(list1: List, list2: List) -> bool:
        """Efficiently checks if two lists are equal as multisets."""
        if len(list1) != len(list2):
            return False
        counts = defaultdict(int)
        for item in list1:
            counts[item] += 1
        for item in list2:
            counts[item] -= 1
            if counts[item] < 0:
                return False
        return all(v == 0 for v in counts.values())
        
    @staticmethod
    def _get_constrained_permutations(
        col_sets1: List[Set],
        result2: List[QueryResultRow]
    ) -> Iterator[Tuple[int, ...]]:
        """Generates valid column permutations, pruning impossible ones."""
        num_cols = len(col_sets1)
        perm_constraints = [set(range(num_cols)) for _ in range(num_cols)]

        if num_cols > 3:
            sample_size = min(20, len(result2))
            for _ in range(sample_size):
                random_row2 = random.choice(result2)
                for i in range(num_cols):
                    for j in list(perm_constraints[i]):
                        if random_row2[j] not in col_sets1[i]:
                            perm_constraints[i].remove(j)
                            
        return itertools.product(*perm_constraints)
        
# --- Main API Functions (keeping original signatures) ---

def score(
    predicted_query_str: str,
    ground_truth_info: Dict[str, Any]
) -> Tuple[float, str, str]:
    """
    Evaluates a predicted SQL query by executing it and comparing results.

    Args:
        predicted_query_str: The predicted SQL, potentially in a markdown block.
        ground_truth_info: A dictionary containing the gold SQL, db_id, etc.

    Returns:
        score: float, (1.0 for a match, 0.0 otherwise)
        pred_results: str, the results of the predicted query execution
        message: str, a message detailing the outcome (e.g., error details).
    """
    db_manager = DatabaseManager()
    evaluator = ExecutionEvaluator()
    
    try:
        # Get database path
        db_path = ground_truth_info.get('db_path')
        if not db_path:
            cache_dir = os.getenv('SQL_CACHE_DIR', 'data/nl2sql/cache')
            db_path = os.path.join(cache_dir, ground_truth_info['db_id'])
        
        gt_sql = ground_truth_info.get('gold_sql') or ground_truth_info.get('gt_sql')
        
        if gt_sql is None:
            return 0.0, "", "No ground truth SQL provided in ground_truth_info"
        
        # Check if database file exists
        if not os.path.exists(db_path):
            return 0.0, "", f"Database file {db_path} does not exist"
        
        # Execute ground truth SQL
        gt_error, gt_results = db_manager.execute_query(db_path, gt_sql)
        if gt_error:
            return 0.0, "", ""
        
        # Extract and execute predicted SQL
        predicted_sql = extract_sql_from_markdown(predicted_query_str)
        if not predicted_sql:
            return 0.0, "", ""
        
        pred_error, pred_results = db_manager.execute_query(db_path, predicted_sql)
        if pred_error:
            return 0.0, "", ""
        
        # Compare results using the improved evaluator
        comparison_method = ground_truth_info.get('cmp_method', 'bird')
        if comparison_method == "spider":
            order_matters = 'order by' in gt_sql.lower()
            is_match = evaluator.are_results_equivalent(gt_results, pred_results, order_matters)
        else:  # Default or 'bird' method
            is_match = evaluator.are_results_equivalent(gt_results, pred_results, order_matters=False)
        
        return (1.0 if is_match else 0.0), "", ""
        
    finally:
        db_manager.close_all_connections()


def sql_observation(
    predicted_query_str: str,
    ground_truth_info: Dict[str, Any],
    timeout: int = 5
) -> str:
    """
    Generate an observation string for the SQL query.
    """
    db_path = ground_truth_info.get('db_path')
    if not db_path:
        cache_dir = os.getenv('SQL_CACHE_DIR', 'data/nl2sql/cache')
        db_path = os.path.join(cache_dir, ground_truth_info['db_id'])
    
    sql = extract_sql_from_markdown(predicted_query_str)
    
    if sql is None or sql == "":
        return "Your previous action is invalid. Follow the format of outputting thinking process and sql tool, and try again."
    elif not os.path.exists(db_path):
        return f"The database file {db_path} does not exist."
    
    # Use DatabaseManager for proper connection handling
    db_manager = DatabaseManager()
    try:
        timeout_ms = timeout * 1000  # Convert to milliseconds
        error, results = db_manager.execute_query(db_path, sql, timeout_ms)
        
        if error:
            if "timeout" in error.lower():
                return f"SQL Timeout:\n{sql}"
            else:
                return error
        
        # Convert results to DataFrame and format
        if results is not None:
            df = pd.DataFrame(results)
            result_str = df.to_string(index=False)
            
            # Truncate if too long
            if len(result_str) > 9000:
                truncated_df = df.head(50)
                return "Truncated to 50 lines since returned response too long: " + truncated_df.to_string(index=False)
            
            return result_str
        else:
            return "No results returned"
            
    except KeyboardInterrupt:
        sys.exit(0)
    except Exception as e:
        return str(e)
    finally:
        db_manager.close_all_connections()