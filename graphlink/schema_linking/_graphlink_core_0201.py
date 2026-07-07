import os
# 禁用Ray自动启动
os.environ["VLLM_DISABLE_RAY"] = "1"
import json
import csv
from tqdm import tqdm
from chat import GPTChat
from concurrent.futures import ThreadPoolExecutor, as_completed
import argparse
import sys
import re
import numpy as np
from utils import search_file, get_api_name, get_dictionary, get_tb_info, get_external, compute_precision_recall, is_csv_empty, clear_name
from reconstruct_data import remove_digits, compress_ddl
import logging
import time
from typing import Dict, Optional, List, Tuple, Set
from datetime import datetime, timedelta
from collections import defaultdict
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# 设置日志配置
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),  # 输出到终端
        logging.FileHandler('desc_generation.log', encoding='utf-8')  # 输出到文件
    ]
)
logger = logging.getLogger(__name__)
import torch
from transformers import AutoTokenizer, is_torch_npu_available
# vllm 延迟导入（仅在使用 QwenEmbeddingModel 时导入）
# from vllm import LLM, SamplingParams
# from vllm.distributed.parallel_state import destroy_model_parallel
# from vllm.inputs.data import TokensPrompt
import gc
import math
csv.field_size_limit(sys.maxsize)
THRESHOLD = 0
DEPS_DEV_V1 = ["sf_bq016", "sf_bq062", "sf_bq063", "sf_bq028"]

os.environ["TORCH_CUDA_ARCH_LIST"] = "8.0"

# 全局时间戳，确保同一次运行的所有任务使用相同的时间戳
GLOBAL_RUN_TIMESTAMP = None


# ==================== 分片表检测器 ====================
class PartitionTableDetector:
    """检测时间分片表并根据query时间范围扩展共现表"""
    
    PARTITION_PATTERNS = [
        (r'(.+)_(\d{8})$', 'daily', '%Y%m%d'),  # ga_sessions_20160801
        (r'(.+)_(\d{4}-\d{2}-\d{2})$', 'daily', '%Y-%m-%d'),
        (r'(.+)_(\d{6})$', 'monthly', '%Y%m'),
        (r'(.+)_(\d{4})$', 'yearly', '%Y'),
        (r'(.+[_\.])(19\d{2}|20\d{2})$', 'yearly', '%Y'),  # storms_1980
    ]
    
    def detect_partition_pattern(self, table_name: str) -> Optional[Dict]:
        """检测表是否为分片表"""
        for pattern, granularity, date_format in self.PARTITION_PATTERNS:
            match = re.match(pattern, table_name)
            if match:
                base_table = match.group(1)
                date_suffix = match.group(2)
                try:
                    datetime.strptime(date_suffix, date_format)
                    return {
                        'base_table': base_table,
                        'date_suffix': date_suffix,
                        'granularity': granularity,
                        'date_format': date_format,
                        'full_table_name': table_name
                    }
                except ValueError:
                    continue
        return None
    
    def extract_time_range_from_query(self, query: str) -> Optional[Tuple[str, str]]:
        """从query中提取时间范围，返回 (start_date, end_date) 格式为 'YYYY-MM-DD'"""
        
        # 规则1: from YYYY-MM-DD to YYYY-MM-DD 或 from YYYY to YYYY
        pattern1 = r'from\s+(\d{4}(?:[-/]\d{2}[-/]\d{2})?)\s+to\s+(\d{4}(?:[-/]\d{2}[-/]\d{2})?)'
        match = re.search(pattern1, query, re.IGNORECASE)
        if match:
            start_str = match.group(1).replace('/', '-')
            end_str = match.group(2).replace('/', '-')
            if len(start_str) == 4:
                start_str = f"{start_str}-01-01"
            if len(end_str) == 4:
                end_str = f"{end_str}-12-31"
            return (start_str, end_str)
        
        # 规则2: between YYYY and YYYY
        pattern2 = r'between\s+(\d{4}(?:[-/]\d{2}[-/]\d{2})?)\s+and\s+(\d{4}(?:[-/]\d{2}[-/]\d{2})?)'
        match = re.search(pattern2, query, re.IGNORECASE)
        if match:
            start_str = match.group(1).replace('/', '-')
            end_str = match.group(2).replace('/', '-')
            if len(start_str) == 4:
                start_str = f"{start_str}-01-01"
            if len(end_str) == 4:
                end_str = f"{end_str}-12-31"
            return (start_str, end_str)
        
        # 规则3: in YYYY 或 during YYYY
        pattern3 = r'(?:in|during)\s+(\d{4})'
        match = re.search(pattern3, query, re.IGNORECASE)
        if match:
            year = match.group(1)
            return (f"{year}-01-01", f"{year}-12-31")
        
        # 规则4: since YYYY 或 after YYYY
        pattern4 = r'(?:since|after)\s+(\d{4}(?:[-/]\d{2}[-/]\d{2})?)'
        match = re.search(pattern4, query, re.IGNORECASE)
        if match:
            date_str = match.group(1).replace('/', '-')
            if len(date_str) == 4:
                return (f"{date_str}-01-01", "2099-12-31")
            else:
                return (date_str, "2099-12-31")
        
        return None
    
    def generate_partition_range(self, base_table: str, granularity: str, 
                                 date_format: str, start_date: str, end_date: str,
                                 all_available_tables: Set[str]) -> List[str]:
        """根据时间范围生成需要的所有分片表"""
        try:
            start = datetime.strptime(start_date, '%Y-%m-%d')
            end = datetime.strptime(end_date, '%Y-%m-%d')
        except ValueError:
            return []
        
        needed_partitions = []
        
        if granularity == 'yearly':
            for year in range(start.year, end.year + 1):
                partition_name = f"{base_table}_{year}"
                if partition_name in all_available_tables:
                    needed_partitions.append(partition_name)
        elif granularity == 'monthly':
            current = start
            while current <= end:
                partition_name = f"{base_table}_{current.strftime(date_format)}"
                if partition_name in all_available_tables:
                    needed_partitions.append(partition_name)
                if current.month == 12:
                    current = current.replace(year=current.year+1, month=1)
                else:
                    current = current.replace(month=current.month+1)
        else:  # daily
            current = start
            while current <= end:
                partition_name = f"{base_table}_{current.strftime(date_format)}"
                if partition_name in all_available_tables:
                    needed_partitions.append(partition_name)
                current += timedelta(days=1)
        
        return needed_partitions
# ==================================================

def get_or_create_run_timestamp() -> str:
    """获取或创建全局运行时间戳"""
    global GLOBAL_RUN_TIMESTAMP
    if GLOBAL_RUN_TIMESTAMP is None:
        GLOBAL_RUN_TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
    return GLOBAL_RUN_TIMESTAMP

def create_time_based_log_dir(base_log_dir: str = "outputs/task_logs", run_name: str = None) -> str:
    """创建基于时间的日志目录
    
    Args:
        base_log_dir: 基础日志目录
        run_name: 可选的运行名称，如果不提供则使用时间戳
    
    Returns:
        创建的时间日志目录路径
    """
    if run_name is None:
        timestamp = get_or_create_run_timestamp()
        run_name = f"run_{timestamp}"
    
    time_based_log_dir = os.path.join(base_log_dir, run_name)
    os.makedirs(time_based_log_dir, exist_ok=True)
    
    # 创建运行信息文件
    run_info_file = os.path.join(time_based_log_dir, "run_info.txt")
    if not os.path.exists(run_info_file):
        with open(run_info_file, 'w', encoding='utf-8') as f:
            f.write(f"Run started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Run directory: {time_based_log_dir}\n")
            f.write(f"Run name: {run_name}\n")
    
    return time_based_log_dir

# 添加任务级别的日志记录器
def create_task_logger(task_id: str, log_dir: str = "outputs/task_logs", run_name: str = None) -> logging.Logger:
    """为每个任务创建独立的日志记录器，按时间创建子目录"""
    
    # 创建基于时间的子目录
    time_based_log_dir = create_time_based_log_dir(log_dir, run_name)
    log_file = os.path.join(time_based_log_dir, f"{task_id}_detailed.log")
    
    # 创建专用的logger
    timestamp = get_or_create_run_timestamp()
    task_logger = logging.getLogger(f"task_{task_id}_{timestamp}")
    task_logger.setLevel(logging.INFO)
    
    # 清除之前的handlers
    for handler in task_logger.handlers[:]:
        task_logger.removeHandler(handler)
    
    # 创建文件handler
    file_handler = logging.FileHandler(log_file, mode='w', encoding='utf-8')
    file_handler.setLevel(logging.INFO)
    
    # 创建详细的formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    file_handler.setFormatter(formatter)
    task_logger.addHandler(file_handler)
    
    # 防止传播到根logger
    task_logger.propagate = False
    
    # 记录日志目录信息
    task_logger.info(f"Task logger initialized for {task_id}")
    task_logger.info(f"Log directory: {time_based_log_dir}")
    task_logger.info(f"Run timestamp: {timestamp}")
    
    return task_logger


def load_task_specific_database_graphs(task_id: str, database_graphs_dir: str, task_logger=None, 
                                      prefer_enhanced: bool = True) -> Dict[str, any]:
    """
    为特定任务加载相关的数据库图，确保任务隔离
    🆕 支持优先加载 workload-enhanced graphs
    
    Args:
        task_id: 任务ID
        database_graphs_dir: 数据库图目录（base）
        task_logger: 任务日志记录器
        prefer_enhanced: 是否优先使用 enhanced graph (默认 True)
    
    Returns:
        Dict[str, any]: 任务相关的数据库图字典
    """
    database_graphs = {}
    
    # 🆕 自动选择 enhanced 或 base graph 目录
    enhanced_dir = database_graphs_dir + "_enhanced"
    actual_dir = database_graphs_dir
    
    if prefer_enhanced and os.path.exists(enhanced_dir):
        # 检查是否有图文件
        enhanced_graph_files = [f for f in os.listdir(enhanced_dir) if f.endswith('.gpickle')]
        if enhanced_graph_files:
            actual_dir = enhanced_dir
            if task_logger:
                task_logger.info(f"✨ 使用 workload-enhanced graphs: {enhanced_dir}")
            logger.info(f"✨ 使用 workload-enhanced graphs: {enhanced_dir}")
    
    if not os.path.exists(actual_dir):
        logger.warning(f"❌ Database graphs directory not found: {actual_dir}")
        return database_graphs
    
    database_graphs_dir = actual_dir  # 使用实际目录
    
    # 获取所有可用的图文件
    graph_files = [f for f in os.listdir(database_graphs_dir) if f.endswith('_schema_graph.gpickle')]
    
    if task_logger:
        task_logger.info(f"📁 Found {len(graph_files)} graph files in {database_graphs_dir}")
        task_logger.info(f"🎯 Loading database graphs for task: {task_id}")
    
    # 尝试多种策略来匹配任务相关的数据库图
    loaded_graphs = []
    
    # 策略1: 直接匹配任务特定的数据库图
    # 支持多种格式：
    # - local任务: local007_Baseball_schema_graph.gpickle
    # - BigQuery任务: bq001.bigquery-public-data.google_analytics_sample_schema_graph.gpickle
    task_specific_patterns = [
        f"{task_id}_*_schema_graph.gpickle",      # local任务格式: local007_Baseball_schema_graph.gpickle
        f"{task_id}.*_schema_graph.gpickle",      # BigQuery任务格式: bq001.bigquery-public-data.google_analytics_sample_schema_graph.gpickle
        f"{task_id}_schema_graph.gpickle"         # 简单格式兼容
    ]
    
    for pattern in task_specific_patterns:
        if '*' in pattern:
            # 使用模式匹配
            import fnmatch
            matching_files = [f for f in graph_files if fnmatch.fnmatch(f, pattern)]
        else:
            # 直接匹配
            matching_files = [pattern] if pattern in graph_files else []
            
        for target_graph_file in matching_files:
            graph_path = os.path.join(database_graphs_dir, target_graph_file)
            try:
                graph = load_graph(graph_path)
                # 提取数据库名（去掉_schema_graph.gpickle后缀）
                db_key = target_graph_file.replace('_schema_graph.gpickle', '')
                database_graphs[db_key] = graph
                loaded_graphs.append(db_key)
                if task_logger:
                    task_logger.info(f"  ✅ Task-specific match: {db_key} ({graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges)")
            except Exception as e:
                logger.error(f"  ❌ Error loading {target_graph_file}: {e}")
                
    if loaded_graphs:
        if task_logger:
            task_logger.info(f"🎯 Successfully loaded {len(loaded_graphs)} task-specific database graphs")
    
    # 策略2: 从任务配置文件中提取数据库信息
    if not loaded_graphs:
        # 尝试从prompts.txt文件中提取数据库名称
        task_config_path = os.path.join(database_graphs_dir.replace('database_graphs_0827', 'examples_lite'), task_id, 'prompts.txt')
        extracted_db_names = set()
        
        if os.path.exists(task_config_path):
            try:
                with open(task_config_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                    # 查找 "Table full name:" 模式来提取数据库名
                    import re
                    table_patterns = re.findall(r'Table full name:\s*([^.]+)\.', content)
                    for db_name in table_patterns:
                        extracted_db_names.add(db_name.strip())
                    
                    if task_logger and extracted_db_names:
                        task_logger.info(f"📋 从任务配置中提取到数据库: {list(extracted_db_names)}")
            except Exception as e:
                if task_logger:
                    task_logger.warning(f"⚠️ 读取任务配置文件失败: {e}")
        
        # 尝试直接匹配提取到的数据库名
        for db_name in extracted_db_names:
            target_file = f"{db_name}_schema_graph.gpickle"
            if target_file in graph_files:
                graph_path = os.path.join(database_graphs_dir, target_file)
                try:
                    graph = load_graph(graph_path)
                    database_graphs[db_name] = graph
                    loaded_graphs.append(db_name)
                    if task_logger:
                        task_logger.info(f"  ✅ Config match: {db_name} ({graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges)")
                except Exception as e:
                    if task_logger:
                        task_logger.error(f"  ❌ Error loading {target_file}: {e}")
    
    # 策略3: 如果仍没有匹配，尝试模糊匹配（只选择最佳匹配）
    if not loaded_graphs:
        best_match = None
        best_score = 0
        
        for graph_file in graph_files:
            database_name = graph_file.replace('_schema_graph.gpickle', '')
            score = 0
            
            # 计算匹配分数
            if task_id.lower() in database_name.lower():
                score += 10  # 任务ID包含在数据库名中
            elif database_name.lower() in task_id.lower():
                score += 8   # 数据库名包含在任务ID中
            else:
                # 检查关键词匹配
                task_parts = [part for part in task_id.lower().split('_') if len(part) > 2]
                db_parts = [part for part in database_name.lower().split('-') + database_name.lower().split('_') if len(part) > 2]
                
                common_parts = set(task_parts).intersection(set(db_parts))
                if common_parts:
                    score += len(common_parts) * 3  # 每个匹配的关键词得3分
            
            if score > best_score:
                best_score = score
                best_match = (database_name, graph_file)
        
        # 只加载最佳匹配的数据库图
        if best_match and best_score >= 3:  # 至少需要3分才认为是有效匹配
            database_name, graph_file = best_match
            graph_path = os.path.join(database_graphs_dir, graph_file)
            try:
                graph = load_graph(graph_path)
                database_graphs[database_name] = graph
                loaded_graphs.append(database_name)
                if task_logger:
                    task_logger.info(f"  ✅ Best fuzzy match: {database_name} (score: {best_score}, {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges)")
            except Exception as e:
                logger.error(f"  ❌ Error loading {graph_file}: {e}")
    
    # 策略3: 如果仍然没有匹配到，加载最大的数据库图作为回退
    if not loaded_graphs:
        logger.warning(f"⚠️  No specific database graph found for task {task_id}, using fallback strategy")
        if task_logger:
            task_logger.warning(f"⚠️  No specific database graph found, trying fallback strategy")
        
        # 选择节点数最多的图作为回退
        largest_graph = None
        largest_db_name = None
        largest_node_count = 0
        
        for graph_file in graph_files:
            database_name = graph_file.replace('_schema_graph.gpickle', '')
            graph_path = os.path.join(database_graphs_dir, graph_file)
            try:
                graph = load_graph(graph_path)
                if graph.number_of_nodes() > largest_node_count:
                    largest_node_count = graph.number_of_nodes()
                    largest_graph = graph
                    largest_db_name = database_name
            except Exception as e:
                logger.error(f"  ❌ Error loading {graph_file}: {e}")
        
        if largest_graph:
            database_graphs[largest_db_name] = largest_graph
            loaded_graphs.append(largest_db_name)
            if task_logger:
                task_logger.info(f"  ✅ Fallback: {largest_db_name} ({largest_graph.number_of_nodes()} nodes, {largest_graph.number_of_edges()} edges)")
    
    logger.info(f"🎯 Task {task_id}: Loaded {len(database_graphs)} database graph(s): {loaded_graphs}")
    if task_logger:
        task_logger.info(f"🎯 Successfully loaded {len(database_graphs)} database graph(s) for this task")
        task_logger.info("📊 任务数据库隔离验证:")
        for db_name in loaded_graphs:
            graph = database_graphs[db_name]
            task_logger.info(f"  ✅ {db_name}: {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges")
        task_logger.info(f"🔒 任务隔离确认: 当前任务只能访问 {len(database_graphs)} 个数据库图，确保了搜索空间的独立性")
    
    return database_graphs


# ================= Qwen Embedding 模型 =================

class QwenEmbeddingModel:
    """使用vllm调用Qwen3-Embedding-8B的封装类"""
    
    def __init__(self, model_name: str = None):
        self.model_name = model_name or os.environ.get("GRAPHLINK_EMBEDDING_MODEL", "Qwen3-Embedding-8B")
        self.model = None
        self.tokenizer = None
        self.max_model_len = 40960  # Qwen3-Embedding的最大长度
        self.task_description = "Given table schema information, retrieve semantically similar table schemas"
        self._init_model()
    
    def _init_model(self):
        """初始化vllm模型"""
        try:
            # 延迟导入 vllm（仅在需要时导入）
            from vllm import LLM, SamplingParams
            
            # 确保使用GPU 0
            import os
            original_cuda = os.environ.get("CUDA_VISIBLE_DEVICES", "")
            os.environ["CUDA_VISIBLE_DEVICES"] = "0"
            
            self.model = LLM(
                model=self.model_name, 
                task="embed",
                tensor_parallel_size=1,
                gpu_memory_utilization=0.45,
                trust_remote_code=True,
                max_model_len=self.max_model_len
            )
            
            # 恢复原始CUDA设置
            if original_cuda:
                os.environ["CUDA_VISIBLE_DEVICES"] = original_cuda
            else:
                os.environ.pop("CUDA_VISIBLE_DEVICES", None)
            
            # 初始化tokenizer用于文本截断
            from transformers import AutoTokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=True)
            
            logger.info(f"Successfully initialized Qwen embedding model: {self.model_name}")
            logger.info(f"Max model length: {self.max_model_len}")
        except Exception as e:
            logger.error(f"Failed to initialize Qwen embedding model: {e}")
            raise RuntimeError(f"Cannot initialize Qwen3-Embedding model: {e}")
    
    def _truncate_text(self, text: str, max_tokens: int = None) -> str:
        """截断文本以适应模型长度限制"""
        if max_tokens is None:
            # 预留一些token给instruction部分
            max_tokens = self.max_model_len - 200
        
        if self.tokenizer is None:
            # 如果没有tokenizer，使用字符截断（粗略估计）
            # 假设平均1个token ≈ 2.5个字符（对于中英混合文本）
            max_chars = int(max_tokens * 2.5)
            if len(text) > max_chars:
                logger.warning(f"Text truncated from {len(text)} to {max_chars} characters")
                return text[:max_chars] + "...[truncated]"
            return text
        
        try:
            # 使用tokenizer精确计算token数量
            tokens = self.tokenizer.encode(text, add_special_tokens=False)
            
            if len(tokens) <= max_tokens:
                return text
            
            # 截断token并解码回文本
            truncated_tokens = tokens[:max_tokens]
            truncated_text = self.tokenizer.decode(truncated_tokens, skip_special_tokens=True)
            
            logger.warning(f"Text truncated from {len(tokens)} to {len(truncated_tokens)} tokens")
            return truncated_text + "...[truncated]"
            
        except Exception as e:
            logger.warning(f"Error in text truncation: {e}, using character-based truncation")
            # 回退到字符截断
            max_chars = int(max_tokens * 2.5)
            if len(text) > max_chars:
                return text[:max_chars] + "...[truncated]"
            return text
    
    def get_detailed_instruct(self, query: str) -> str:
        """构建带instruction的查询"""
        return f'Instruct: {self.task_description}\nQuery: {query}'
    
    def encode(self, texts, normalize_embeddings=True):
        """
        编码文本，兼容SentenceTransformer的接口
        :param texts: 单个文本字符串或文本列表
        :param normalize_embeddings: 是否归一化embedding
        :return: numpy array 或 tensor
        """
        if self.model is None:
            raise RuntimeError("Qwen3-Embedding model is not initialized")
        
        # 处理单个文本的情况
        is_single = isinstance(texts, str)
        if is_single:
            texts = [texts]
        
        # 截断过长的文本并为每个文本添加instruction
        input_texts = []
        for text in texts:
            truncated_text = self._truncate_text(text)
            input_texts.append(self.get_detailed_instruct(truncated_text))
        
        # 使用vllm生成embeddings
        outputs = self.model.embed(input_texts)
        embeddings = torch.tensor([o.outputs.embedding for o in outputs])
        
        # 归一化处理
        if normalize_embeddings:
            embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
        
        # 转换为numpy格式（兼容原有代码）
        embeddings_np = embeddings.cpu().numpy()
        
        # 如果输入是单个文本，返回单个向量
        if is_single:
            return embeddings_np[0]
        else:
            return embeddings_np
    
    @staticmethod
    def cos_sim(a, b):
        """计算余弦相似度，兼容sentence_transformers.util.cos_sim"""
        # 转换为tensor
        if not isinstance(a, torch.Tensor):
            a = torch.tensor(a)
        if not isinstance(b, torch.Tensor):
            b = torch.tensor(b)
        
        # 确保是2D tensor
        if a.dim() == 1:
            a = a.unsqueeze(0)
        if b.dim() == 1:
            b = b.unsqueeze(0)
        
        # 计算余弦相似度
        return torch.mm(a, b.transpose(0, 1))

# 全局embedding模型实例
_global_embedding_model = None

def get_embedding_model():
    """获取全局embedding模型实例"""
    global _global_embedding_model
    if _global_embedding_model is None:
        _global_embedding_model = QwenEmbeddingModel()
    return _global_embedding_model

# ================= 语义向量存储和加载功能 =================

def save_table_embeddings(embeddings_dict: Dict[str, np.ndarray], example_id: str, example_root: str):
    """
    将表的语义向量保存到对应样本目录下
    :param embeddings_dict: {table_fullname: embedding_vector}
    :param example_id: 样本ID
    :param example_root: 样本根目录
    """
    sample_dir = os.path.join(example_root, example_id)
    os.makedirs(sample_dir, exist_ok=True)
    
    # 保存为.npz格式（压缩的numpy格式，适合存储多个数组）
    embeddings_path = os.path.join(sample_dir, "table_embeddings.npz")
    
    # 将字典转换为numpy可保存的格式
    table_names = list(embeddings_dict.keys())
    embeddings_array = np.array([embeddings_dict[name] for name in table_names])
    
    np.savez_compressed(embeddings_path, 
                       table_names=table_names, 
                       embeddings=embeddings_array)
    
    logger.info(f"Saved {len(embeddings_dict)} table embeddings to {embeddings_path}")
    
    # 同时保存一个JSON元数据文件，便于查看
    metadata_path = os.path.join(sample_dir, "table_embeddings_metadata.json")
    metadata = {
        "total_tables": len(embeddings_dict),
        "embedding_dim": embeddings_array.shape[1] if len(embeddings_array) > 0 else 0,
        "table_names": table_names
    }
    with open(metadata_path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)


def load_table_embeddings(example_id: str, example_root: str) -> Optional[Dict[str, np.ndarray]]:
    """
    从样本目录加载表的语义向量
    :param example_id: 样本ID
    :param example_root: 样本根目录
    :return: {table_fullname: embedding_vector} 或 None
    """
    embeddings_path = os.path.join(example_root, example_id, "table_embeddings.npz")
    
    if not os.path.exists(embeddings_path):
        logger.warning(f"No embeddings found for {example_id} at {embeddings_path}")
        return None
    
    try:
        data = np.load(embeddings_path, allow_pickle=True)
        table_names = data['table_names']
        embeddings_array = data['embeddings']
        
        # 重建字典
        embeddings_dict = {}
        for i, table_name in enumerate(table_names):
            embeddings_dict[str(table_name)] = embeddings_array[i]
        
        logger.info(f"Loaded {len(embeddings_dict)} table embeddings from {embeddings_path}")
        return embeddings_dict
        
    except Exception as e:
        logger.error(f"Error loading embeddings from {embeddings_path}: {e}")
        return None


def load_sample_embeddings(example_root: str, target_example_ids: List[str] = None) -> Dict[str, np.ndarray]:
    """
    加载指定样本的所有语义向量
    :param example_root: 样本根目录
    :param target_example_ids: 目标样本ID列表，None表示加载所有
    :return: 合并后的 {table_fullname: embedding_vector}
    """
    all_embeddings = {}
    
    if target_example_ids is None:
        # 获取所有样本ID
        target_example_ids = [d for d in os.listdir(example_root) 
                            if os.path.isdir(os.path.join(example_root, d)) and d != 'local']
    
    for example_id in target_example_ids:
        embeddings = load_table_embeddings(example_id, example_root)
        if embeddings:
            all_embeddings.update(embeddings)
    
    logger.info(f"Loaded total {len(all_embeddings)} embeddings from {len(target_example_ids)} samples")
    return all_embeddings


def batch_compute_and_save_embeddings(example_root: str, model: QwenEmbeddingModel = None, 
                                    force_recompute: bool = False):
    """
    批量计算并保存所有样本的表语义向量
    :param example_root: 样本根目录
    :param model: 语义模型
    :param force_recompute: 是否强制重新计算
    """
    if model is None:
        model = get_embedding_model()
    
    dictionaries = [d for d in os.listdir(example_root) 
                   if os.path.isdir(os.path.join(example_root, d)) and d != 'local']
    
    logger.info(f"Starting batch embedding computation for {len(dictionaries)} samples")
    
    for example_id in tqdm(dictionaries, desc="Computing embeddings"):
        embeddings_path = os.path.join(example_root, example_id, "table_embeddings.npz")
        
        # 检查是否已存在且不需要重新计算
        if os.path.exists(embeddings_path) and not force_recompute:
            logger.info(f"Embeddings already exist for {example_id}, skipping")
            continue
        
        # 读取表描述
        descriptions_path = os.path.join(example_root, example_id, "table_descriptions.json")
        if not os.path.exists(descriptions_path):
            logger.warning(f"No table descriptions found for {example_id}, skipping")
            continue
        
        try:
            with open(descriptions_path, 'r', encoding='utf-8') as f:
                table_descriptions = json.load(f)
            
            if not table_descriptions:
                logger.warning(f"Empty table descriptions for {example_id}, skipping")
                continue
            
            # 计算语义向量
            embeddings_dict = {}
            for table_name, description in table_descriptions.items():
                # 兼容转换数据集中 description 为 dict 的情况（如 BIRD/Spider 转换格式）
                if isinstance(description, dict):
                    description = description.get("description", str(description))
                if description:  # 确保描述不为空
                    try:
                        embedding = model.encode(description, normalize_embeddings=True)
                        embeddings_dict[table_name] = embedding
                    except Exception as e:
                        logger.warning(f"Error computing embedding for {table_name}: {e}")
            
            if embeddings_dict:
                save_table_embeddings(embeddings_dict, example_id, example_root)
                logger.info(f"Computed and saved {len(embeddings_dict)} embeddings for {example_id}")
            else:
                logger.warning(f"No valid embeddings computed for {example_id}")
                
        except Exception as e:
            logger.error(f"Error processing {example_id}: {e}")


def get_related_database_embeddings(query_example_id: str, example_root: str) -> Dict[str, np.ndarray]:
    """
    获取与查询样本相关的数据库的语义向量
    :param query_example_id: 查询样本ID
    :param example_root: 样本根目录
    :return: 相关数据库的 {table_fullname: embedding_vector}
    """
    # 这里实现获取相关数据库的逻辑
    # 目前先返回查询样本本身的向量，后续可以根据数据库关联规则扩展
    return load_table_embeddings(query_example_id, example_root) or {}


def update_embeddings_in_graphs(database_graphs: Dict[str, any], 
                               embeddings_dict: Dict[str, np.ndarray]):
    """
    将预计算的语义向量更新到数据库图中，避免重复计算
    :param database_graphs: 数据库图字典
    :param embeddings_dict: 预计算的向量字典
    """
    for db_name, graph in database_graphs.items():
        for node in graph.nodes():
            if node in embeddings_dict:
                # 将向量作为节点属性存储
                graph.nodes[node]['embedding'] = embeddings_dict[node]
                logger.debug(f"Updated embedding for table {node} in database {db_name}")
    
    logger.info(f"Updated embeddings for {len(embeddings_dict)} tables across {len(database_graphs)} database graphs")

# ================= NetworkX兼容性工具函数 =================

def save_graph(graph, file_path):
    """
    保存NetworkX图，兼容不同版本的NetworkX
    :param graph: NetworkX图对象
    :param file_path: 保存路径
    """
    try:
        # 尝试使用pickle直接保存（推荐方法）
        import pickle
        with open(file_path, 'wb') as f:
            pickle.dump(graph, f, pickle.HIGHEST_PROTOCOL)
        logger.debug(f"Graph saved using pickle: {file_path}")
    except Exception as e:
        try:
            # 回退到旧版本NetworkX的方法
            nx.write_gpickle(graph, file_path)
            logger.debug(f"Graph saved using nx.write_gpickle: {file_path}")
        except AttributeError:
            # 如果都不行，再次尝试pickle
            import pickle
            with open(file_path, 'wb') as f:
                pickle.dump(graph, f, pickle.HIGHEST_PROTOCOL)
            logger.debug(f"Graph saved using fallback pickle: {file_path}")

def load_graph(file_path):
    """
    加载NetworkX图，兼容不同版本的NetworkX
    :param file_path: 图文件路径
    :return: NetworkX图对象
    """
    try:
        # 尝试使用pickle直接加载
        import pickle
        with open(file_path, 'rb') as f:
            graph = pickle.load(f)
        logger.debug(f"Graph loaded using pickle: {file_path}")
        return graph
    except Exception as e:
        try:
            # 回退到旧版本NetworkX的方法
            graph = nx.read_gpickle(file_path)
            logger.debug(f"Graph loaded using nx.read_gpickle: {file_path}")
            return graph
        except AttributeError:
            # 如果都不行，再次尝试pickle
            import pickle
            with open(file_path, 'rb') as f:
                graph = pickle.load(f)
            logger.debug(f"Graph loaded using fallback pickle: {file_path}")
            return graph


def format_instruction(instruction, query, doc):
    text = [
        {"role": "system", "content": "Judge whether the Document meets the requirements based on the Query and the Instruct provided. Note that the answer can only be \"yes\" or \"no\"."},
        {"role": "user", "content": f"<Instruct>: {instruction}\n\n<Query>: {query}\n\n<Document>: {doc}"}
    ]
    return text

def process_inputs(pairs, instruction, max_length, suffix_tokens, tokenizer):
    """
    处理输入，适配GPTChat格式
    如果tokenizer为None（GPTChat模式），返回文本格式
    否则保持原有的token格式（兼容性）
    """
    if tokenizer is None:
        # GPTChat模式：返回文本格式
        formatted_messages = []
        for query, doc in pairs:
            # 构建完整的prompt
            prompt = f"""You are doing table level schema linking. Given a table with schema information and the task, you should think step by step and decide whether this table is related to the task.

<Instruction>: {instruction}

<Task>: {query}

<Table>: {doc}

Please answer only 'yes' or 'no' based on whether the table might be relevant to the task."""
            
            # 截断过长的prompt
            if len(prompt) > max_length:
                # 智能截断策略：优先保留Table Description，智能截断列信息
                task_part = f"<Task>: {query}"
                instruction_part = f"<Instruction>: {instruction}"
                prefix = """You are doing table level schema linking. Given a table with schema information and the task, you should think step by step and decide whether this table is related to the task.

"""
                suffix = "\n\nPlease answer only 'yes' or 'no' based on whether the table is relevant to the task."
                
                available_length = max_length - len(prefix) - len(instruction_part) - len(task_part) - len(suffix) - 50
                
                # 智能截断：分离表信息的不同部分
                truncated_doc = _smart_truncate_table_info(doc, available_length)
                
                prompt = f"{prefix}{instruction_part}\n\n{task_part}\n\n<Table>: {truncated_doc}{suffix}"
            
            formatted_messages.append(prompt)
        return formatted_messages
    else:
        # 原有的tokenizer模式：保持兼容性
        from vllm.inputs.data import TokensPrompt  # 延迟导入
        
        messages = [format_instruction(instruction, query, doc) for query, doc in pairs]
        messages = tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=False, enable_thinking=False
        )
        messages = [ele[:max_length] + suffix_tokens for ele in messages]
        messages = [TokensPrompt(prompt_token_ids=ele) for ele in messages]
        return messages

def _smart_truncate_table_info(doc: str, available_length: int) -> str:
    """
    智能截断表信息，优先保留Table Description和重要的结构信息
    """
    if len(doc) <= available_length:
        return doc
    
    lines = doc.split('\n')
    
    # 分离不同部分
    table_name_line = ""
    description_lines = []
    column_lines = []
    other_lines = []
    
    in_description = False
    
    for line in lines:
        if line.startswith("Table full name:"):
            table_name_line = line
        elif line.startswith("Table Description:"):
            in_description = True
            description_lines.append(line)
        elif in_description and line.strip() and not line.startswith("Columns:") and not line.startswith("Total columns:"):
            description_lines.append(line)
        elif line.startswith("Columns:"):
            in_description = False
            column_lines.append(line)
        elif line.startswith("Total columns:"):
            in_description = False
            other_lines.append(line)
        else:
            if in_description:
                description_lines.append(line)
            else:
                other_lines.append(line)
    
    # 构建优先级顺序：表名 -> 描述 -> 总列数 -> 列信息
    result_parts = []
    current_length = 0
    
    # 1. 表名（必须保留）
    if table_name_line:
        result_parts.append(table_name_line)
        current_length += len(table_name_line) + 1
    
    # 2. 表描述（高优先级）
    if description_lines:
        description_text = '\n'.join(description_lines)
        if current_length + len(description_text) + 1 <= available_length:
            result_parts.extend(description_lines)
            current_length += len(description_text) + len(description_lines)  # +换行符
        else:
            # 截断描述但保留开头
            remaining = available_length - current_length - 20  # 留点空间给后面
            if remaining > 50:  # 至少保留50个字符的描述
                truncated_desc = description_text[:remaining] + "...(description truncated)"
                result_parts.append("Table Description: " + truncated_desc[18:] if description_text.startswith("Table Description:") else truncated_desc)
                current_length = available_length - 20
    
    # 3. 总列数（中等优先级）
    for line in other_lines:
        if line.startswith("Total columns:") and current_length + len(line) + 1 <= available_length:
            result_parts.append(line)
            current_length += len(line) + 1
            break
    
    # 4. 列信息（如果还有空间）
    for line in column_lines:
        if current_length + len(line) + 1 <= available_length:
            result_parts.append(line)
            current_length += len(line) + 1
        else:
            # 截断列信息
            remaining = available_length - current_length - 15
            if remaining > 20:
                if line.startswith("Columns:"):
                    columns_part = line[8:].strip()  # 去掉"Columns:"
                    truncated_columns = columns_part[:remaining] + "...(columns truncated)"
                    result_parts.append(f"Columns: {truncated_columns}")
            break
    
    return '\n'.join(result_parts)

def compute_logits(model, messages, sampling_params, true_token, false_token):
    """
    使用GPTChat进行判断，返回True/False的布尔值列表
    为了兼容性，True映射为0.9，False映射为0.1
    """
    logger.info(f"🚀 compute_logits: 开始处理 {len(messages)} 个LLM请求")
    scores = []
    
    # messages现在是文本格式的列表（GPTChat模式）或TokensPrompt格式（兼容模式）
    for i, message in enumerate(messages, 1):
        try:
            # 判断是否为GPTChat模式
            if isinstance(message, str):
                # GPTChat模式：message已经是完整的prompt文本
                prompt = message
            elif hasattr(message, 'prompt_token_ids'):
                # 兼容模式：这是TokensPrompt格式，暂时跳过
                logger.warning("TokensPrompt format not supported with GPTChat, using fallback")
                scores.append(0.5)
                continue
            else:
                prompt = str(message)
            
            # 使用GPTChat生成回答
            logger.info(f"🤖 LLM请求 {i}/{len(messages)}: 正在调用GPTChat...")
            model.init_messages()
            response = model.get_model_response_txt(prompt)
            logger.info(f"✅ LLM请求 {i}/{len(messages)}: GPTChat调用完成")
            
            # get_model_response_txt() 返回字符串，不是列表
            answer = str(response).lower().strip()
            
            # 记录LLM输入和输出（过滤掉 <think> 标签）
            prompt_preview = prompt[:500] + "..." if len(prompt) > 500 else prompt
            # 去除 <think> 标签后再截断预览
            answer_filtered = remove_think_tags(answer)
            answer_preview = answer_filtered[:200] + "..." if len(answer_filtered) > 200 else answer_filtered
            logger.info(f"🤖 LLM Input: {prompt_preview}")
            logger.info(f"🤖 LLM Output: {answer_preview}")
            
            # 基于回答判断True或False（使用去除 think 标签后的文本，避免 think 块中的否定词干扰）
            is_relevant = parse_relevance_answer(answer_filtered)
            
            # 为了兼容现有代码（期望分数），将布尔值转换为分数
            score = 0.9 if is_relevant else 0.1
            scores.append(score)
            
            logger.info(f"🤖 LLM Decision: {is_relevant} (score: {score})")
            logger.debug(f"GPTChat rerank: '{answer[:50]}...' -> {is_relevant} -> score: {score}")
            
        except Exception as e:
            logger.warning(f"Error in GPTChat rerank: {e}")
            scores.append(0.5)  # 出错时给中性分数
    
    return scores

def remove_think_tags(text: str) -> str:
    """
    去除文本中的 <think>...</think> 标签及其内容
    
    Args:
        text: 原始文本
    
    Returns:
        去除 think 标签后的文本
    """
    # 使用正则表达式去除 <think>...</think> 标签及其内容
    # re.DOTALL 使得 . 匹配包括换行符在内的任意字符
    cleaned_text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL | re.IGNORECASE)
    return cleaned_text.strip()

def parse_relevance_answer(answer: str) -> bool:
    """
    解析GPTChat的回答，返回True/False布尔值
    """
    answer = answer.lower().strip()
    
    # 明确的肯定回答
    if any(word in answer for word in ["yes", "true", "relevant", "related"]):
        # 检查是否有否定词
        if any(neg in answer for neg in ["not", "no", "isn't", "aren't", "not relevant", "not related"]):
            return False  # 明确否定
        return True  # 明确肯定
    
    # 明确的否定回答
    elif any(word in answer for word in ["no", "false", "not relevant", "not related", "irrelevant", "unrelated"]):
        return False
    
    # 偏向肯定的回答
    elif any(word in answer for word in ["likely", "probably", "possible", "might", "could", "somewhat", "partially"]):
        return True  # 模糊肯定当作True
    
    # 偏向否定的回答
    elif any(word in answer for word in ["unlikely", "probably not", "doubt", "questionable"]):
        return False
    
    # 完全不确定，默认为False
    else:
        logger.debug(f"Unclear answer: '{answer}', defaulting to False")
        return False

def get_rerank_model():
    """获取rerank模型 - 使用GPTChat"""
    logger.info("Loading GPTChat rerank model...")
    
    # 初始化GPTChat，使用合适的模型
    chat_model = GPTChat(model="Qwen3-235B-A22B-Instruct-2507-FP8", temperature=0)
    
    logger.info("GPTChat rerank model loaded successfully")
    
    return {
        'model': chat_model,
        'max_length': 8192,  # 保持兼容性
        'tokenizer': None,   # GPTChat不需要tokenizer
        'suffix_tokens': [], # 保持兼容性
        'true_token': None,  # 不再需要
        'false_token': None, # 不再需要
        'sampling_params': None # 不再需要
    }

prefix = '<|im_start|>system\nYou are doing table level schema linking. Given a table with schema information and the task, you should think step by step and decide whether this table is related to the task. Note that the answer can only be "yes" or "no".<|im_end|>\n<|im_start|>user\n'
suffix = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
query_template = "{prefix}<Instruct>: {instruction}\n<task>: {query}\n"
document_template = "<table>: {doc}{suffix}"

def ask_model_sl(example_path, json_save_pth, score_threshold=0.5, use_description=True, 
                 use_semantic_graph_search=False, database_graphs_dir="database_graphs",
                 use_subquery_decomposition=False, max_samples_debug=None,
                 enable_topk_rerank=False, top_k_preselection=5,
                 use_coverage_bonus=False, coverage_beta=0.3,
                 enable_sql_validation=False, max_validation_iterations=3,
                enable_batch_rerank=False, batch_size=10,
                use_workload_evolution=False, workload_stats_path='graph_evolution_data/workload_stats.json',
                workload_weight=1.0,
                edge_workload_weight=None, node_workload_weight=None,
                enable_graph_topology=True,
                task="lite"):
    """
    Schema Linking主函数，支持传统算法和新的语义图搜索算法
    
    Args:
        edge_workload_weight: λ - 边增强权重（默认使用 workload_weight）
        node_workload_weight: γ - 节点先验权重（默认使用 workload_weight）
    """
    # 🔬 处理消融实验参数
    if edge_workload_weight is None:
        edge_workload_weight = workload_weight
    if node_workload_weight is None:
        node_workload_weight = workload_weight
    
    linked_dic = {}
    rerank_components = get_rerank_model()
    
    # 加载任务字典
    dictionaries, task_dict = get_dictionary(example_path, task)
    
    # 全局统计收集
    global_stats = []
    
    # 加载数据库图（如果使用语义图搜索）
    # 🎯 优化：将数据库图加载移至每个任务内部，确保任务隔离
    # 不再在全局加载所有数据库图，而是在每个任务中按需加载
    if use_semantic_graph_search:
        if not os.path.exists(database_graphs_dir):
            logger.warning(f"❌ Database graphs directory not found: {database_graphs_dir}")
            logger.warning("⚠️  Semantic graph search will work without graph expansion")
        else:
            logger.info(f"🗂️  Database graphs directory found: {database_graphs_dir}")
            logger.info("🎯 Database graphs will be loaded per-task for better isolation")
    


    def process_example(ex_id):
        # 创建任务专用的日志记录器
        task_logger = create_task_logger(ex_id)
        task_logger.info(f"🚀 开始处理任务: {ex_id}")
        
        try:
            # Load table information from directory or SQLite
            if ex_id.startswith("local"):
                # Process local samples (SQLite files)
                sample_dir = os.path.join(example_path, ex_id)
                sqlite_files = [f for f in os.listdir(sample_dir) if f.endswith('.sqlite')]
                
                if not sqlite_files:
                    print(f"[DEBUG] No SQLite files found for local example {ex_id}")
                    return None, None
                
                # Usually local samples have only one sqlite file
                sqlite_file = sqlite_files[0]
                sqlite_path = os.path.join(sample_dir, sqlite_file)
                tbs = get_table_info_from_sqlite(sqlite_path, ex_id)
                
                if not tbs:
                    print(f"[DEBUG] No tables found in SQLite for {ex_id}")
                    return None, None
                    
                print(f"[DEBUG] Loaded {len(tbs)} tables from SQLite for {ex_id}")
                print(f"[DEBUG] SQLite path: {sqlite_path}")
                
                # Load table descriptions for local samples  
                table_descriptions = load_table_descriptions(example_path, ex_id) if use_description else {}
            else:
                # Load table information from directory
                tbs = get_table_info_from_directory(example_path, ex_id)
                if not tbs:
                    print(f"[DEBUG] No tables found for {ex_id}")
                    return None, None
                
                # For non-local samples, sqlite_path is None (not used for validation)
                sqlite_path = None
                
                # Load table descriptions
                table_descriptions = load_table_descriptions(example_path, ex_id) if use_description else {}
                if use_description and table_descriptions:
                    print(f"[DEBUG] Loaded {len(table_descriptions)} table descriptions for {ex_id}")
                elif use_description:
                    print(f"[DEBUG] No table descriptions found for {ex_id}")
            
            # Get external knowledge
            external = get_external_knowledge(example_path, ex_id)
            task = task_dict[ex_id]
            
            # 🆕 获取数据库引擎类型（基于 example_id 前缀）
            db_engine = get_api_name(ex_id)
            print(f"[DEBUG] Processing {ex_id}, found {len(tbs)} tables, DB engine: {db_engine}")
            
            # 🎯 为当前任务加载相关的数据库图，确保任务隔离
            task_specific_database_graphs = {}
            if use_semantic_graph_search:
                task_specific_database_graphs = load_task_specific_database_graphs(
                    ex_id, database_graphs_dir, task_logger
                )
                if task_logger:
                    task_logger.info(f"📊 任务数据库图隔离: 加载了 {len(task_specific_database_graphs)} 个相关数据库图")
            
            # 选择搜索算法
            if use_semantic_graph_search:
                logger.info(f"🔬 Using SEMANTIC GRAPH SEARCH for {ex_id}")
                # 记录任务基本信息
                task_logger.info(f"📝 任务描述: {task}")
                task_logger.info(f"🗂️ 可用表格数量: {len(tbs)}")
                task_logger.info(f"🎯 搜索策略: {'子查询分解' if use_subquery_decomposition else '传统语义图搜索'}")
                
                result = ask_model_sl_semantic_graph_search(
                    tbs, task, rerank_components, score_threshold, external, 
                    table_descriptions, use_description, task_specific_database_graphs, example_path,
                    max_expansions=10, current_example_id=ex_id, global_stats=global_stats,
                    use_subquery_decomposition=use_subquery_decomposition, task_logger=task_logger,
                    enable_topk_rerank=enable_topk_rerank, top_k_preselection=top_k_preselection,
                    use_coverage_bonus=use_coverage_bonus, coverage_beta=coverage_beta,
                    enable_sql_validation=enable_sql_validation, 
                    max_validation_iterations=max_validation_iterations,
                    sqlite_path=sqlite_path,  # 🔧 传递sqlite_path（对于SQLite类型），其他类型会使用current_example_id
                    db_engine=db_engine,  # 🆕 使用动态获取的数据库引擎类型
                    enable_batch_rerank=enable_batch_rerank,  # 🚀 批量rerank判断
                    batch_size=batch_size,  # 🚀 批量判断批次大小
                    use_workload_evolution=use_workload_evolution,  # 🆕 Workload evolution
                    workload_stats_path=workload_stats_path,  # 🆕
                    workload_weight=workload_weight,  # 🆕
                    edge_workload_weight=edge_workload_weight,  # 🔬 λ: 边增强权重
                    node_workload_weight=node_workload_weight,  # 🔬 γ: 节点先验权重
                    enable_graph_topology=enable_graph_topology
                )
            else:
                logger.info(f"🔍 Using TRADITIONAL SEARCH for {ex_id}")
                result = ask_model_sl_(
                    tbs, task, rerank_components, score_threshold, external, 
                    table_descriptions, use_description
                )
                
            print(f"[DEBUG] {ex_id} completed with {len(result)} results")
            
            if task_logger:
                task_logger.info("=" * 80)
                task_logger.info("任务完成总结")
                task_logger.info("=" * 80)
                task_logger.info(f"🎉 任务 {ex_id} 处理完成")
                task_logger.info(f"✅ 最终选中表数量: {len(result)}")
                task_logger.info("📝 最终选中的表:")
                for table_result in result:
                    table_name = table_result.get('table name', 'Unknown')
                    answer = table_result.get('answer', 'Unknown')
                    expansion_level = table_result.get('expansion_level', 0)
                    task_logger.info(f"  {answer}: {table_name} (层级: {expansion_level})")
                task_logger.info("=" * 80)
            
            return ex_id, result
            
        except Exception as e:
            print(f"Error processing {ex_id}: {e}")
            import traceback
            traceback.print_exc()
            return None, None

    search_method = "semantic graph search" if use_semantic_graph_search else "traditional"
    print(f"Doing table-level schema linking using {search_method}")
    
    # 处理调试模式的样本数量限制
    if max_samples_debug is not None:
        original_count = len(dictionaries)
        dictionaries = dictionaries[:max_samples_debug]
        print(f"🐛 调试模式: 限制处理 {len(dictionaries)}/{original_count} 个任务")
        logger.info(f"🐛 调试模式: 限制处理 {len(dictionaries)}/{original_count} 个任务")
    
    with ThreadPoolExecutor(max_workers=1) as executor:
        futures = [executor.submit(process_example, ex_id) for ex_id in dictionaries]
        for i, future in enumerate(tqdm(as_completed(futures), total=len(futures), desc="Processing"), 1):
            try:
                ex_id, result = future.result()
                if ex_id is not None:
                    linked_dic[ex_id] = result
                    print(f"✅ 任务 {i}/{len(futures)}: {ex_id} 处理完成，选中 {len(result) if result else 0} 个表")
                    logger.info(f"✅ 任务 {i}/{len(futures)}: {ex_id} 处理完成，选中 {len(result) if result else 0} 个表")
            except Exception as e:
                print(f"❌ 任务处理错误: {e}")
                logger.error(f"❌ 任务处理错误: {e}")
                continue

        with open(json_save_pth, "w") as f:
            json.dump(linked_dic, f, indent=4)
    
    # 全局统计分析
    if global_stats and use_semantic_graph_search:
        analyze_global_search_stats(global_stats)

    # 📊 保存 GPTChat 调用统计信息
    try:
        if rerank_components and 'model' in rerank_components:
            chat_model = rerank_components['model']
            if hasattr(chat_model, 'save_statistics'):
                # 生成带时间戳的统计文件名
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                stats_file = f"gptchat_statistics_{timestamp}.json"
                stats_file = os.path.join(os.path.dirname(json_save_pth), stats_file)
                
                # 保存统计信息
                chat_model.save_statistics(stats_file, include_history=True)
                chat_model.print_statistics()
                
                logger.info(f"📊 GPTChat 统计信息已保存到: {stats_file}")
    except Exception as e:
        logger.warning(f"⚠️ 保存 GPTChat 统计信息失败: {e}")

def analyze_global_search_stats(global_stats):
    """
    分析全局搜索统计，计算平均值、最大值、最小值
    """
    if not global_stats:
        logger.warning("No global stats available for analysis")
        return
    
    logger.info("📊 === GLOBAL SEARCH STATISTICS ANALYSIS ===")
    logger.info(f"📈 Total tasks analyzed: {len(global_stats)}")
    
    # 提取各项指标
    search_efficiencies = [stat['search_efficiency'] for stat in global_stats]
    search_space_sizes = [stat['search_space_size'] for stat in global_stats]
    search_counts = [stat['search_count'] for stat in global_stats]
    expansion_search_counts = [stat['expansion_search_count'] for stat in global_stats]
    total_searches = [stat['total_searches'] for stat in global_stats]
    relevant_tables = [stat['relevant_tables'] for stat in global_stats]
    irrelevant_tables = [stat['irrelevant_tables'] for stat in global_stats]
    expanded_tables = [stat['expanded_tables'] for stat in global_stats]
    
    # 计算统计指标
    def calculate_stats(values, name):
        if not values:
            return
        avg_val = sum(values) / len(values)
        min_val = min(values)
        max_val = max(values)
        logger.info(f"📊 {name}:")
        logger.info(f"    Average: {avg_val:.2f}")
        logger.info(f"    Min: {min_val:.2f}")
        logger.info(f"    Max: {max_val:.2f}")
        logger.info(f"    Range: {max_val - min_val:.2f}")
    
    # 输出各项统计
    calculate_stats(search_efficiencies, "Search Efficiency (%)")
    calculate_stats(search_space_sizes, "Search Space Size")
    calculate_stats(search_counts, "Base Search Count")
    calculate_stats(expansion_search_counts, "Expansion Search Count")
    calculate_stats(total_searches, "Total Searches")
    calculate_stats(relevant_tables, "Relevant Tables Found")
    calculate_stats(irrelevant_tables, "Irrelevant Tables")
    calculate_stats(expanded_tables, "Expanded Tables")
    
    # 效率分析
    high_efficiency_tasks = [i for i, eff in enumerate(search_efficiencies) if eff < 50]
    low_efficiency_tasks = [i for i, eff in enumerate(search_efficiencies) if eff > 80]
    
    logger.info(f"🎯 Efficiency Analysis:")
    logger.info(f"    High efficiency (early stop): {len(high_efficiency_tasks)} tasks")
    logger.info(f"    Low efficiency (full search): {len(low_efficiency_tasks)} tasks")
    logger.info(f"    Average efficiency: {sum(search_efficiencies)/len(search_efficiencies):.1f}%")
    
    # 扩展分析
    tasks_with_expansions = [i for i, exp in enumerate(expansion_search_counts) if exp > 0]
    logger.info(f"🌳 Expansion Analysis:")
    logger.info(f"    Tasks with expansions: {len(tasks_with_expansions)}/{len(global_stats)}")
    if tasks_with_expansions:
        avg_expansions = sum(expansion_search_counts) / len(global_stats)
        logger.info(f"    Average expansions per task: {avg_expansions:.1f}")
    
    logger.info("📊 === END GLOBAL ANALYSIS ===")

def load_table_descriptions(example_path, example_id):
    """Load table descriptions from table_descriptions.json file."""
    desc_file = os.path.join(example_path, example_id, "table_descriptions.json")
    if os.path.exists(desc_file):
        try:
            with open(desc_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            print(f"[DEBUG] Failed to load descriptions for {example_id}: {e}")
            return {}
    else:
        print(f"[DEBUG] No description file found for {example_id}")
        return {}

def extract_table_name(table_schema: str) -> str:
    """从表schema字符串中提取表名"""
    try:
        match = re.search(r'^Table full name:\s*(.+)$', table_schema, re.MULTILINE)
        return match.group(1) if match else "unknown"
    except Exception:
        return "unknown"


def extract_related_example_ids_from_tables(tbs: List[str], example_root: str) -> List[str]:
    """
    从表信息中提取相关的样本ID，确保schema linking范围独立
    """
    related_example_ids = set()
    
    # 获取所有可用的样本ID
    available_example_ids = [d for d in os.listdir(example_root) 
                           if os.path.isdir(os.path.join(example_root, d))]
    
    logger.debug(f"Available example IDs: {available_example_ids}")
    
    for tb in tbs:
        table_name = extract_table_name(tb)
        logger.debug(f"Analyzing table: {table_name}")
        
        # 尝试多种策略来匹配样本ID
        
        # 策略1: 检查表名是否包含样本ID信息（如bigquery-public-data等）
        for example_id in available_example_ids:
            # 检查表名中是否包含数据库名
            if table_name.startswith(example_id + ".") or example_id in table_name:
                related_example_ids.add(example_id)
                logger.debug(f"  Matched via database prefix: {example_id}")
                continue
        
        # 策略2: 检查是否有该表的JSON文件存在于某个样本中
        for example_id in available_example_ids:
            example_dir = os.path.join(example_root, example_id)
            if os.path.isdir(example_dir):
                # 递归查找JSON文件
                for root, dirs, files in os.walk(example_dir):
                    # 跳过特殊目录
                    dirs[:] = [d for d in dirs if d not in ['spider', 'output', '__pycache__']]
                    
                    for file in files:
                        if file.endswith('.json') and 'table_descriptions' not in file:
                            file_path = os.path.join(root, file)
                            try:
                                with open(file_path, 'r', encoding='utf-8') as f:
                                    table_json = json.load(f)
                                
                                # 检查表名匹配
                                table_fullname = table_json.get('table_fullname', '')
                                table_name_simple = table_json.get('table_name', '')
                                
                                if (table_fullname == table_name or 
                                    table_name_simple == table_name.split('.')[-1] or
                                    table_name.endswith('.' + table_name_simple)):
                                    related_example_ids.add(example_id)
                                    logger.debug(f"  Matched via JSON file in: {example_id}")
                                    break
                            except Exception as e:
                                logger.debug(f"  Error reading {file_path}: {e}")
                                continue
    
    # 如果没有找到相关样本，返回当前可用的所有样本（保守策略）
    if not related_example_ids:
        logger.warning("No specific related samples found, using all available samples")
        related_example_ids = set(available_example_ids)
    
    result = list(related_example_ids)
    logger.debug(f"Final related example IDs: {result}")
    return result


def ask_model_sl_semantic_graph_search(tbs, task, rerank_components, score_threshold=0.5, external="", 
                                      table_descriptions=None, use_description=True, 
                                      database_graphs=None, example_root=None, max_expansions=30,
                                      current_example_id=None, global_stats=None, max_expansion_level=10,
                                      use_subquery_decomposition=False, task_logger=None,
                                      enable_topk_rerank=False, top_k_preselection=5,
                                      use_coverage_bonus=False, coverage_beta=0.3,
                                      enable_sql_validation=False, max_validation_iterations=3,
                                      sqlite_path=None, db_engine=None,
                                     enable_batch_rerank=False, batch_size=10,
                                     use_workload_evolution=False, workload_stats_path='graph_evolution_data/workload_stats.json',
                                     workload_weight=1.0,
                                     edge_workload_weight=None, node_workload_weight=None,
                                     enable_graph_topology=True):
    """
    基于语义相似度排序和图扩展的Schema Linking搜索算法，支持多层级递归扩展
    1. 计算所有表的语义相似度并排序（优先使用预计算向量）
    2. 依次进行rerank判断，成功则基于MinHash扩展相邻节点
    3. 支持多层级递归扩展：扩展的表可以继续扩展其邻居
    4. 子节点判断时包含完整的父节点链信息
    5. 连续失败则停止扩展，每层都有独立的早停机制
    """
    # 🔬 处理消融实验参数
    if edge_workload_weight is None:
        edge_workload_weight = workload_weight
    if node_workload_weight is None:
        node_workload_weight = workload_weight
    
    logger.info("🚀 Starting semantic graph-based schema linking search")
    logger.info(f"📊 Input: {len(tbs)} tables, task: '{task[:100]}...'")
    logger.info(f"⚙️  Parameters: score_threshold={score_threshold}, max_expansions={max_expansions}")
    
    if current_example_id:
        logger.info(f"🎯 Current sample scope: {current_example_id}")
    
    if table_descriptions is None:
        table_descriptions = {}
    
    if database_graphs is None:
        database_graphs = {}
    
    logger.info(f"📚 Available resources: {len(table_descriptions)} descriptions, {len(database_graphs)} database graphs")
    logger.info(f"🔍 Database graphs: {list(database_graphs.keys())}")
    
    # 尝试加载预计算的语义向量（仅当前样本相关）
    logger.info("🔍 Step 1: Loading precomputed embeddings for semantic similarity")
    all_embeddings = {}
    if example_root:
        try:
            # 构建当前样本的表集合，用于搜索空间限制
            current_table_set = set(extract_table_name(tb) for tb in tbs)
            logger.info(f"🔒 Search scope limited to {len(current_table_set)} tables from current sample")
            
            # 从tbs中提取相关的样本ID，确保范围独立
            if current_example_id:
                # 如果有当前样本ID，优先使用
                related_example_ids = [current_example_id]
                logger.info(f"🎯 Using current sample ID: {current_example_id}")
            else:
                # 否则从表信息中推断
                related_example_ids = extract_related_example_ids_from_tables(tbs, example_root)
                logger.info(f"🎯 Identified {len(related_example_ids)} related sample(s): {related_example_ids}")
            
            # 只加载相关样本的预计算embedding
            all_embeddings = load_sample_embeddings(example_root, related_example_ids)
            
            # 过滤embedding，只保留当前表集合中的
            filtered_embeddings = {k: v for k, v in all_embeddings.items() if k in current_table_set}
            all_embeddings = filtered_embeddings
            
            logger.info(f"✅ Loaded {len(all_embeddings)} precomputed table embeddings (filtered to current scope)")
        except Exception as e:
            logger.warning(f"⚠️  Could not load precomputed embeddings: {e}")
    
    # 初始化embedding模型（仅在需要时）
    embedding_model = None
    # 检查是否需要embedding模型
    missing_embeddings = []
    for tb in tbs:
        table_name = extract_table_name(tb)
        if table_name not in all_embeddings:
            missing_embeddings.append(table_name)
    
    need_embedding_model = len(all_embeddings) == 0 or len(missing_embeddings) > 0
    
    if missing_embeddings and len(all_embeddings) > 0:
        logger.info(f"⚠️  Missing embeddings for {len(missing_embeddings)} tables: {missing_embeddings[:5]}{'...' if len(missing_embeddings) > 5 else ''}")
    
    if need_embedding_model:
        logger.info("🔧 Initializing embedding model for missing embeddings")
        embedding_model = get_embedding_model()
    else:
        logger.info("✅ All embeddings available from precomputed cache")
    
    # 计算查询向量
    logger.info("🔍 Encoding query for semantic similarity comparison")
    if embedding_model is None:
        embedding_model = get_embedding_model()
    query_embedding = embedding_model.encode(task, normalize_embeddings=True)
    
    logger.info("📈 Step 2: Computing semantic similarities for all tables")
    table_similarities = []
    
    logger.info(f"🔄 Processing {len(tbs)} tables for semantic similarity computation...")
    
    # 统计precomputed vs computed on-the-fly
    precomputed_count = 0
    computed_count = 0
    
    for idx, tb in enumerate(tbs, 1):
        table_name = extract_table_name(tb)
        
        logger.debug(f"  📋 [{idx}/{len(tbs)}] Processing table: {table_name}")
        
        # 构建用于语义计算的文本
        enhanced_tb = tb
        has_description = False
        if use_description and table_name in table_descriptions and table_descriptions[table_name]:
            desc_raw = table_descriptions[table_name]
            desc = desc_raw.get("description", str(desc_raw)) if isinstance(desc_raw, dict) else str(desc_raw)
            enhanced_tb = f"{tb}\n\nTable Description: {desc}"
            has_description = True
            logger.debug(f"    ✅ Enhanced with description: {desc[:100]}...")
        else:
            logger.debug(f"    ⚠️  No description available for {table_name}")
        
        # 优先使用预计算的向量
        similarity = 0.0
        if table_name in all_embeddings:
            # 使用预计算的向量
            try:
                table_embedding = all_embeddings[table_name]
                similarity = float(np.dot(query_embedding, table_embedding))
                precomputed_count += 1
                logger.debug(f"    ⚡ Using precomputed embedding, similarity: {similarity:.4f}")
            except Exception as e:
                logger.warning(f"❌ Error using precomputed embedding for {table_name}: {e}")
                # 降级到实时计算
                if embedding_model:
                    try:
                        table_embedding = embedding_model.encode(enhanced_tb, normalize_embeddings=True)
                        similarity = float(np.dot(query_embedding, table_embedding))
                        computed_count += 1
                        logger.debug(f"    🔧 Fallback computed embedding, similarity: {similarity:.4f}")
                    except Exception as e2:
                        logger.warning(f"❌ Error computing similarity for {table_name}: {e2}")
                        similarity = 0.0
        else:
            # 实时计算语义相似度
            if embedding_model:
                try:
                    table_embedding = embedding_model.encode(enhanced_tb, normalize_embeddings=True)
                    similarity = float(np.dot(query_embedding, table_embedding))
                    computed_count += 1
                    logger.debug(f"    🔧 Computed embedding, similarity: {similarity:.4f}")
                except Exception as e:
                    logger.warning(f"❌ Error computing similarity for {table_name}: {e}")
                    similarity = 0.0
            else:
                logger.warning(f"❌ No embedding available for {table_name} and no model initialized")
                similarity = 0.0
        
        table_similarities.append({
            'table_name': table_name,
            'table_schema': tb,
            'enhanced_schema': enhanced_tb,
            'similarity': similarity,
            'has_description': has_description
        })
    
    logger.info(f"📊 Embedding usage: {precomputed_count} precomputed, {computed_count} computed on-the-fly")
    
    # 按语义相似度降序排序
    table_similarities.sort(key=lambda x: x['similarity'], reverse=True)
    logger.info(f"✅ Ranked {len(table_similarities)} tables by semantic similarity")
    
    # 效率统计
    if precomputed_count > 0:
        efficiency_pct = (precomputed_count / len(tbs)) * 100
        logger.info(f"⚡ Efficiency: {efficiency_pct:.1f}% used precomputed embeddings")
    
    # 显示top排名
    logger.info("🏆 Top 5 tables by semantic similarity:")
    for i, table_info in enumerate(table_similarities[:5]):
        desc_status = "📝" if table_info.get('has_description', False) else "📄"
        logger.info(f"  {i+1}. {desc_status} {table_info['table_name']} (similarity: {table_info['similarity']:.4f})")
    
    if len(table_similarities) > 5:
        logger.info(f"  ... and {len(table_similarities) - 5} more tables")
    
    # Step 3: 依次进行rerank判断和图扩展
    logger.info("🤖 Step 3: Starting rerank evaluation and graph expansion")
    
    # 🎯 子查询分解逻辑
    if use_subquery_decomposition:
        logger.info("🚀 启用子查询分解方法")
        logger.info(f"📝 原始任务: {task[:100]}...")
        
        # 构建当前表集合
        current_table_set = set(extract_table_name(tb) for tb in tbs)
        
        # 使用子查询分解方法
        result = schema_linking_with_graph_search(
            task=task,
            rerank_components=rerank_components,
            database_graphs=database_graphs,
            table_descriptions=table_descriptions,
            use_description=use_description,
            max_consecutive_failures=5,
            all_embeddings=all_embeddings,
            current_table_set=current_table_set,
            expansion_search_count=[0],
            example_root=example_root,
            current_example_id=current_example_id,
            use_subquery_decomposition=True,
            embedding_model=embedding_model,
            task_logger=task_logger,
            enable_topk_rerank=enable_topk_rerank,
            top_k_preselection=top_k_preselection,
            use_coverage_bonus=use_coverage_bonus,
            coverage_beta=coverage_beta,
            enable_sql_validation=enable_sql_validation,
            max_validation_iterations=max_validation_iterations,
            sqlite_path=sqlite_path,
            db_engine=db_engine,
            enable_batch_rerank=enable_batch_rerank,
            batch_size=batch_size,
            use_workload_evolution=use_workload_evolution,  # 🆕
            workload_stats_path=workload_stats_path,  # 🆕
            workload_weight=workload_weight,  # 🆕
            edge_workload_weight=edge_workload_weight,  # 🔬 λ: 边增强权重
            node_workload_weight=node_workload_weight,  # 🔬 γ: 节点先验权重
            enable_graph_topology=enable_graph_topology
        )
        
        logger.info(f"🎉 子查询分解搜索完成，选中 {len(result)} 个表")
        
        # 🚀 只返回实际评估过的表（不包括未评估的表）
        logger.info(f"📊 输出格式：只包含实际评估过的 {len(result)} 个表")
        return result
    
    linked = []
    processed_tables = set()
    expansion_count = 0
    consecutive_failures = 0
    max_consecutive_failures = 5  # 3次连续失败触发早停
    
    # 🚀 优化：新增全局候选表缓存，避免重复构建搜索空间
    global_candidate_cache = set()  # 全局候选表缓存
    
    # 新增搜索统计变量
    search_space_size = len(table_similarities)  # 搜索空间大小
    search_count = 0  # 实际搜索次数
    expansion_search_count = [0]  # 扩展搜索次数 (使用列表以便在函数间传递)
    duplicate_candidates_avoided = [0]  # 避免的重复候选表数量
    
    # 🔥 新增：路径追踪
    successful_paths = {}  # 记录每个父表的成功扩展路径
    
    logger.info(f"🎯 Search strategy: max_consecutive_failures={max_consecutive_failures}, max_expansions={max_expansions}, max_expansion_level={max_expansion_level}")
    logger.info(f"📊 Search space: {search_space_size} tables available")
    
    for idx, table_info in enumerate(table_similarities):
        table_name = table_info['table_name']
        
        if table_name in processed_tables:
            logger.debug(f"⏭️  Skipping already processed table: {table_name}")
            continue
        
        similarity = table_info['similarity']
        logger.info(f"🔍 [{idx+1}/{len(table_similarities)}] Evaluating: {table_name}")
        logger.info(f"    📊 Semantic similarity: {similarity:.4f}")
        logger.info(f"    🔄 Consecutive failures so far: {consecutive_failures}")
        
        # Step 2.1: 对当前表进行rerank判断
        search_count += 1  # 增加搜索计数
        logger.info(f"    🤖 Running rerank evaluation... (search #{search_count}/{search_space_size})")
        is_relevant = rerank_single_table(
            task, table_info['enhanced_schema'], rerank_components, 
            parent_info=None, instruction_suffix="judge whether this table is relevant to the task"
        )
        
        if is_relevant:
            logger.info(f"    ✅ RELEVANT: {table_name} passed rerank evaluation")
            linked.append({
                "think": "",
                "answer": "Y",
                "columns": [],
                "table name": table_name,
                "score": 0.9,
                "similarity": table_info['similarity'],
                "expansion_level": 0
            })
            processed_tables.add(table_name)
            consecutive_failures = 0
            
            # Step 2.2: 基于MinHash扩展相邻节点
            if expansion_count < max_expansions:
                logger.info(f"    🌳 Starting recursive expansion from {table_name} (expansion {expansion_count+1}/{max_expansions}, max level: {max_expansion_level})")
                expanded_tables = expand_via_minhash(
                    table_name, table_info, task, rerank_components, 
                    database_graphs, table_descriptions, processed_tables,
                    use_description, max_consecutive_failures, all_embeddings,
                    current_table_set, expansion_search_count,
                    current_level=1, max_level=max_expansion_level, parent_chain=None,
                    global_candidate_cache=global_candidate_cache, duplicate_candidates_avoided=duplicate_candidates_avoided,
                    successful_paths=successful_paths, example_root=example_root, current_example_id=current_example_id
                )
                
                expansion_added = 0
                for expanded_table in expanded_tables:
                    if expansion_count < max_expansions:
                        linked.append(expanded_table)
                        processed_tables.add(expanded_table['table name'])
                        expansion_count += 1
                        expansion_added += 1
                    else:
                        logger.info(f"    🛑 Max expansions ({max_expansions}) reached, stopping")
                        break
                
                if expansion_added > 0:
                    logger.info(f"    ✅ Successfully expanded {expansion_added} tables from {table_name}")
                else:
                    logger.info(f"    ❌ No tables expanded from {table_name}")
            else:
                logger.info(f"    🚫 Expansion limit reached ({expansion_count}/{max_expansions}), skipping expansion")
        else:
            logger.info(f"    ❌ NOT RELEVANT: {table_name} failed rerank evaluation")
            linked.append({
                "think": "",
                "answer": "N", 
                "columns": [],
                "table name": table_name,
                "score": 0.1,
                "similarity": table_info['similarity'],
                "expansion_level": 0
            })
            consecutive_failures += 1
            
            # 连续失败则停止
            if consecutive_failures >= max_consecutive_failures:
                remaining_count = len(table_similarities) - idx - 1
                actual_searched_so_far = search_count + expansion_search_count[0]
                current_efficiency = (actual_searched_so_far / search_space_size) * 100
                
                logger.info(f"🛑 EARLY STOPPING: {consecutive_failures} consecutive failures reached")
                logger.info(f"    📝 Skipping remaining {remaining_count} tables (no rerank evaluation)")
                logger.info(f"    📊 Current search efficiency: {actual_searched_so_far}/{search_space_size} tables searched ({current_efficiency:.1f}%)")
                logger.info(f"    🎯 Early stop saved {remaining_count} rerank evaluations")
                
                # 🚀 修复：不再将剩余表标记为N，直接跳过
                # 这样早停才能真正节省计算，提高搜索效率
                break
    
    # 最终统计
    relevant_tables = [r for r in linked if r['answer'] == 'Y']
    irrelevant_tables = [r for r in linked if r['answer'] == 'N']
    expanded_tables = [r for r in relevant_tables if r.get('expansion_level', 0) > 0]
    
    # 统计不同层级的扩展表
    level_stats = {}
    for table in expanded_tables:
        level = table.get('expansion_level', 0)
        level_stats[level] = level_stats.get(level, 0) + 1
    
    logger.info("🎉 Semantic graph search completed!")
    logger.info(f"📊 Final statistics:")
    logger.info(f"    ✅ Relevant tables: {len(relevant_tables)}")
    logger.info(f"    ❌ Irrelevant tables: {len(irrelevant_tables)}")
    logger.info(f"    🌳 Expanded tables: {len(expanded_tables)}")
    logger.info(f"    📊 Expansion level breakdown: {level_stats}")
    logger.info(f"    📈 Total processed: {len(processed_tables)}")
    logger.info(f"    🔄 Expansion efficiency: {expansion_count}/{max_expansions} used")
    # 🚀 修复：计算真实的搜索效率（只包含实际rerank的表）
    actual_searched = search_count + expansion_search_count[0]
    real_search_efficiency = (actual_searched / search_space_size) * 100
    early_stop_saved = search_space_size - actual_searched
    
    logger.info(f"    🔍 Search efficiency: {actual_searched}/{search_space_size} tables searched ({real_search_efficiency:.1f}%)")
    logger.info(f"    🌳 Expansion searches: {expansion_search_count[0]} additional searches")
    logger.info(f"    🎯 Early stop saved: {early_stop_saved} rerank evaluations")
    
    # 记录全局统计
    if global_stats is not None:
        task_stats = {
            'search_space_size': search_space_size,
            'search_count': search_count,
            'expansion_search_count': expansion_search_count[0],
            'actual_searched': actual_searched,
            'search_efficiency': real_search_efficiency,
            'early_stop_saved': early_stop_saved,
            'total_searches': actual_searched,
            'relevant_tables': len(relevant_tables),
            'irrelevant_tables': len(irrelevant_tables),
            'expanded_tables': len(expanded_tables),
            'max_expansion_level': max_expansion_level,
            'expansion_level_stats': level_stats
        }
        global_stats.append(task_stats)
        logger.info(f"    📊 Task stats recorded for global analysis")
    
    if relevant_tables:
        logger.info("🏆 Relevant tables found:")
        for i, table in enumerate(relevant_tables[:10]):  # 显示前10个
            level = table.get('expansion_level', 0)
            parent = table.get('parent_table', '')
            if level == 0:
                logger.info(f"    {i+1}. 🎯 {table['table name']} (base, sim: {table['similarity']:.4f})")
            else:
                logger.info(f"    {i+1}. 🔗 {table['table name']} (via {parent}, level: {level})")
        
        if len(relevant_tables) > 10:
            logger.info(f"    ... and {len(relevant_tables) - 10} more relevant tables")
    
    # 显示扩展层级统计
    if level_stats:
        logger.info("🌳 Expansion level statistics:")
        for level in sorted(level_stats.keys()):
            count = level_stats[level]
            logger.info(f"    Level {level}: {count} tables")
    
    # 🚀 显示搜索空间优化统计
    logger.info("⚡ Search space optimization statistics:")
    logger.info(f"    📊 Total search space: {search_space_size} tables")
    logger.info(f"    🔍 Actual searches performed: {search_count + expansion_search_count[0]}")
    logger.info(f"    🗂️  Duplicate candidates avoided: {duplicate_candidates_avoided[0]}")
    logger.info(f"    💾 Global candidate cache size: {len(global_candidate_cache)}")
    
    efficiency = ((search_count + expansion_search_count[0]) / search_space_size) * 100 if search_space_size > 0 else 0
    logger.info(f"    📈 Search efficiency: {efficiency:.1f}% of search space actually processed")
    
    if duplicate_candidates_avoided[0] > 0:
        reduction_rate = (duplicate_candidates_avoided[0] / (search_count + expansion_search_count[0] + duplicate_candidates_avoided[0])) * 100
        logger.info(f"    🎯 Duplication reduction: {reduction_rate:.1f}% redundant searches avoided")
    
    # 🔥 保存扩展路径到文件
    if successful_paths and current_example_id:
        try:
            # 使用指定的data目录
            output_dir = os.environ.get("GRAPHLINK_EXPANSION_PATHS_DIR", "data/expansion_paths")
            
            save_expansion_paths_to_file(current_example_id, task, successful_paths, output_dir)
            logger.info(f"🛤️  Expansion paths saved for {current_example_id}: {len(successful_paths)} root parents, {sum(len(paths) for paths in successful_paths.values())} total paths")
        except Exception as e:
            logger.error(f"❌ Failed to save expansion paths for {current_example_id}: {e}")
    
    return linked


def rerank_with_embedding_prefilter(task, table_name, table_schema, rerank_components, 
                                   graph, embedding_model, subquery_embedding, 
                                   similarity_threshold=0.05, parent_info=None, 
                                   instruction_suffix="", task_logger=None):
    """
    两阶段筛选：先用embedding相似度快速筛选，再用LLM精确判断
    
    Args:
        task: 任务描述
        table_name: 表名
        table_schema: 表schema
        rerank_components: LLM组件
        graph: 数据库图
        embedding_model: embedding模型
        subquery_embedding: 预计算的子查询embedding
        similarity_threshold: embedding相似度阈值
        parent_info: 父表信息
        instruction_suffix: 指令后缀
        task_logger: 任务日志记录器
        
    Returns:
        bool: 是否相关
    """
    
    # 第一阶段：Embedding相似度快速筛选
    try:
        # 获取表的embedding
        table_embedding = None
        if hasattr(graph, '_table_embeddings_cache') and table_name in graph._table_embeddings_cache:
            table_embedding = graph._table_embeddings_cache[table_name]
        else:
            # 如果缓存中没有，快速计算
            if embedding_model:
                table_embedding = embedding_model.encode([table_schema])[0]
                # 缓存结果
                if not hasattr(graph, '_table_embeddings_cache'):
                    graph._table_embeddings_cache = {}
                graph._table_embeddings_cache[table_name] = table_embedding
        
        if table_embedding is not None and subquery_embedding is not None:
            # 确保embedding都是1D数组
            subquery_emb = np.array(subquery_embedding).flatten()
            table_emb = np.array(table_embedding).flatten()
            
            # 计算cosine相似度
            dot_product = np.dot(subquery_emb, table_emb)
            norm_product = np.linalg.norm(subquery_emb) * np.linalg.norm(table_emb)
            similarity = float(dot_product / norm_product) if norm_product > 0 else 0.0
            
            if task_logger:
                task_logger.debug(f"📊 Embedding similarity for {table_name}: {similarity:.3f}")
            
            # 如果相似度低于阈值，直接返回False，跳过LLM调用
            if similarity < similarity_threshold:
                if task_logger:
                    task_logger.info(f"⚡ 快速筛选: {table_name} 相似度 {similarity:.3f} < {similarity_threshold}，跳过LLM判断")
                return False
            
            if task_logger:
                task_logger.info(f"✅ 通过embedding筛选: {table_name} 相似度 {similarity:.3f} >= {similarity_threshold}，进行LLM判断")
        
    except Exception as e:
        if task_logger:
            task_logger.warning(f"⚠️ Embedding筛选失败，直接进行LLM判断: {e}")
    
    # 第二阶段：LLM精确判断（只对通过embedding筛选的表）
    return rerank_single_table(task, table_schema, rerank_components, parent_info, instruction_suffix)


def _build_batch_topology_text(table_names: list, graph) -> str:
    """
    从图中提取 batch 内部表之间的关系，格式化为 prompt 段落。
    仅展示 batch 内节点之间的边。
    """
    if graph is None:
        return ""
    batch_set = set(table_names)
    edges = []
    try:
        for u, v, data in graph.edges(data=True):
            if u in batch_set and v in batch_set:
                edges.append((u, v, data))
    except Exception:
        return ""
    if not edges:
        return ""

    lines = ["## Known Relationships Between Candidate Tables:",
             "(Use these as strong hints for JOIN necessity and table selection)"]
    for u, v, data in edges:
        fk_type = data.get('fk_type')
        reason = data.get('reason', '')
        if fk_type == 'explicit':
            rel = "foreign_key (explicit)"
        elif fk_type == 'IND':
            conf = data.get('ind_confidence', data.get('weight', 0.0))
            col_pair = data.get('ind_column_pair') or ('?', '?')
            rel = f"IND/implicit-FK (conf={conf:.2f}, via columns: {col_pair[0]} ↔ {col_pair[1]})"
        elif fk_type == 'AIND':
            conf = data.get('ind_confidence', data.get('weight', 0.0))
            col_pair = data.get('ind_column_pair') or ('?', '?')
            rel = f"AIND/approx-FK (conf={conf:.2f}, via columns: {col_pair[0]} ↔ {col_pair[1]})"
        elif 'minhash' in reason:
            mh = data.get('mh_sim', data.get('weight', 0.0))
            rel = f"similar columns (minhash={mh:.2f})"
        elif 'desc_sim' in reason:
            ds = data.get('desc_sim', data.get('weight', 0.0))
            rel = f"similar semantics (desc_sim={ds:.2f})"
        else:
            rel = f"related (weight={data.get('weight', 0.0):.2f})"
        lines.append(f"- **{u}** <--[{rel}]--> **{v}**")
    return "\n".join(lines)


def rerank_batch_tables(task, table_schemas_dict, rerank_components, parent_info=None, instruction_suffix="", task_logger=None, all_available_tables=None, graph=None, enable_graph_topology=True):
    """
    批量对多个表进行rerank判断（一次LLM调用判断多个表）
    使用结构化prompt，要求LLM返回JSON格式的判断结果
    支持分片表共现扩展和图拓扑增强
    
    Args:
        task: 任务描述
        table_schemas_dict: Dict[table_name -> table_schema] 表名到schema的映射
        rerank_components: LLM组件
        parent_info: 父节点信息（可选）
        instruction_suffix: 指令后缀
        task_logger: 任务日志记录器
        all_available_tables: 所有可用的表名集合（用于分片表扩展）
        graph: NetworkX图对象（可选），用于向LLM提供batch内部的拓扑关系（FK/IND/相似度边）
        enable_graph_topology: 是否将batch内部图拓扑关系注入prompt（默认True；传入graph=None时自动跳过）
        
    Returns:
        Tuple[Dict[table_name -> bool], Dict[table_name -> List[str]]]:
          - 第一个元素：表名到相关性判断结果的映射
          - 第二个元素：表名到相关列名列表的映射（列粒度，空列表表示使用全部列）
    """
    if not table_schemas_dict:
        return {}, {}
    
    table_names = list(table_schemas_dict.keys())
    batch_size = len(table_names)
    
    logger.info(f"🚀 批量rerank判断: 一次性评估 {batch_size} 个表")
    if task_logger:
        task_logger.info(f"🚀 批量rerank判断: 一次性评估 {batch_size} 个表")
        task_logger.info(f"   候选表完整列表:")
        for i, tbl in enumerate(list(table_schemas_dict.keys()), 1):
            task_logger.info(f"     {i}. {tbl}")
    
    # 🚀 构建结构化的批量判断prompt
    # 1. 构建候选表列表（直接使用真实表名）
    candidate_tables_list = []
    schema_truncated_count = 0
    for table_name, table_schema in table_schemas_dict.items():
        # 截断过长的schema描述
        max_schema_len = 1200  # 增加到1200字符以包含更多列信息
        if len(table_schema) > max_schema_len:
            schema_preview = table_schema[:max_schema_len] + "..."
            schema_truncated_count += 1
        else:
            schema_preview = table_schema
        candidate_tables_list.append(f"**{table_name}**:\n{schema_preview}\n")
    
    candidates_text = "\n".join(candidate_tables_list)
    
    if schema_truncated_count > 0:
        logger.info(f"⚠️  {schema_truncated_count}/{batch_size} 个表的schema被截断到{1200}字符")
        if task_logger:
            task_logger.info(f"⚠️  {schema_truncated_count}/{batch_size} 个表的schema被截断到{1200}字符")
    
    # 1.5 构建拓扑信息段落（batch内部边关系），受 enable_graph_topology 控制
    topology_text = _build_batch_topology_text(table_names, graph) if enable_graph_topology else ""
    # 在 candidates_text 和 Output Format 之间注入，空行分隔
    topology_section = f"\n{topology_text}\n" if topology_text else ""
    if topology_text:
        edge_count = topology_text.count("\n- ")
        logger.info(f"🔗 Batch拓扑增强: {edge_count} 条表间关系注入prompt")
        if task_logger:
            task_logger.info(f"🔗 Batch拓扑增强: {edge_count} 条表间关系注入prompt")
    elif not enable_graph_topology:
        logger.debug("🔗 Batch拓扑增强已禁用（enable_graph_topology=False）")
    
    # 2. 构建结构化prompt
    structured_prompt = f"""You are a schema pruning policy for Text-to-SQL. Output MUST be valid JSON only.
Do not output any explanation outside JSON.

Given a natural language question and candidate tables, select the minimal set of tables needed to answer the question.
Also predict which conditional functions are needed to solve this question.

## Definitions (gates):

- **join**: 1 if answering requires combining >=2 tables (explicit JOIN or implicit comma-separated tables in FROM); else 0.
  Examples: "users who made purchases" → join users & orders
  Examples: FROM users u, orders o → join=1 (implicit join)
  
- **predicate**: 1 if answering requires filtering rows with conditions (WHERE/HAVING with =, >, <, BETWEEN, IN, LIKE, date ranges, IS NOT NULL, etc.); else 0.
  Examples: "users from California", "orders after 2023", "price > 100"
  
- **aggregation**: 1 if answering requires aggregation or grouping (COUNT/SUM/AVG/MAX/MIN, GROUP BY, HAVING, or DISTINCT for counting); else 0.
  Examples: "total revenue", "average age", "how many users"

- **partition_cooccurrence**: 1 if the question involves a TIME RANGE and candidate tables include partitioned tables (tables with date/year suffixes like `table_20160801`, `table_2016`, `storms_1980`). When this is 1, you MUST also specify the time range in "time_range" field; else 0.
  Examples:
  - "data from August 1-10, 2016" + see `ga_sessions_20160801` → partition_cooccurrence=1, time_range={{"start": "2016-08-01", "end": "2016-08-10"}}
  - "storms from 1980 to 1985" + see `storms_1980` → partition_cooccurrence=1, time_range={{"start": "1980-01-01", "end": "1985-12-31"}}
  - "all 2023 data" + see `table_2023` → partition_cooccurrence=1, time_range={{"start": "2023-01-01", "end": "2023-12-31"}}
  - Single table with no date suffix → partition_cooccurrence=0, time_range=null

## Important Rules:

1. **Prefer fewer tables**. Do NOT include tables unless you can justify they are needed for the answer.

2. **If join=0** then selected_tables should usually be **1 table** (single-table query).

3. **If join=1** then you typically need **>=2 tables** (multi-table query with JOIN or comma-separated FROM).

4. **Gates are diagnostic**: They help explain WHY you selected these tables. Think:
   - Does the question ask for aggregated/summary statistics? → aggregation=1
   - Does it filter by attributes? → predicate=1
   - Does it relate data across tables? → join=1

5. **Output JSON** with keys exactly: `selected_tables`, `gates`, `reasoning`.

6. **Gates values** must be 0 or 1 (integers).

7. **Reasoning** must be short and structured (help humans understand your logic).

8. **CRITICAL: Use EXACT table names from the candidate list below**. Copy the complete table name EXACTLY as shown between the `**` markers, including ALL parts (e.g., `database.schema.table` or `DB.DB.TABLE`). Do NOT shorten, simplify, or modify table names in any way.

9. **selected_columns**: For each selected table, list ONLY the column names that are directly needed to answer the question (for SELECT, JOIN ON, WHERE, GROUP BY, ORDER BY, etc.). Use the column names EXACTLY as they appear in the schema. If ALL columns of a table are needed, list them all. Omit irrelevant columns.

## Example:

**Question**: "What is the average salary of employees in the IT department?"

**Candidate Tables**:
**company.hr.employees**:
id INT, name VARCHAR, department_id INT, salary DECIMAL

**company.hr.departments**:
id INT, name VARCHAR

**Correct Output** (note the EXACT table names from candidate list):
{{
  "selected_tables": ["company.hr.employees", "company.hr.departments"],
  "selected_columns": {{
    "company.hr.employees": ["department_id", "salary"],
    "company.hr.departments": ["id", "name"]
  }},
  "gates": {{
    "join": 1,
    "predicate": 1,
    "aggregation": 1,
    "partition_cooccurrence": 0
  }},
  "time_range": null,
  "reasoning": {{
    "key_terms": ["average salary", "IT department"],
    "why_tables": {{
      "company.hr.employees": "contains salary data",
      "company.hr.departments": "needed to filter IT department"
    }},
    "notes": "Need JOIN to match dept name, WHERE for IT filter, AVG for salary"
  }}
}}

## Your Turn:

## Question:
{task}

## Candidate Tables (with schemas):
{candidates_text}
{topology_section}
## Output Format:

Return JSON only in this exact format:

{{
  "selected_tables": ["EXACT_TABLE_NAME_1", "EXACT_TABLE_NAME_2"],
  "selected_columns": {{
    "EXACT_TABLE_NAME_1": ["col_a", "col_b"],
    "EXACT_TABLE_NAME_2": ["col_c", "col_d"]
  }},
  "gates": {{
    "join": 0,
    "predicate": 0,
    "aggregation": 0,
    "partition_cooccurrence": 0
  }},
  "time_range": {{"start": "YYYY-MM-DD", "end": "YYYY-MM-DD"}} or null,
  "reasoning": {{
    "key_terms": ["term1", "term2"],
    "why_tables": {{
      "EXACT_TABLE_NAME_1": "one short reason",
      "EXACT_TABLE_NAME_2": "one short reason"
    }},
    "notes": "optional short note"
  }}
}}

**Critical Rules**:
1. Output ONLY the JSON object. No markdown, no code blocks, no explanations before or after.
2. In "selected_tables" array, use the COMPLETE table names EXACTLY as they appear between ** markers in the candidate list above.
3. Table names may contain multiple dots (e.g., "DB.SCHEMA.TABLE" or "project.dataset.table") - you MUST include ALL parts.
4. In "selected_columns", keys must be the same EXACT table names as in "selected_tables". Values are arrays of column names from the schema.
"""
    
    try:
        # 使用Chat模式而不是logits模式，以便获取结构化JSON输出
        model = rerank_components['model']
        
        logger.info(f"🤖 批量表相关性判断: 正在调用LLM评估 {batch_size} 个表（结构化JSON输出）")
        if task_logger:
            task_logger.info(f"🤖 批量表相关性判断: 正在调用LLM评估 {batch_size} 个表（结构化JSON输出）")
        
        # 调用LLM生成JSON响应
        response_text = model.get_model_response_txt(structured_prompt)
        
        # 记录原始响应到INFO级别以便调试
        logger.info(f"📝 LLM原始响应（前500字符）: {response_text[:500]}...")
        if task_logger:
            task_logger.info(f"📝 LLM原始响应（前500字符）: {response_text[:500]}...")
            task_logger.debug(f"📝 LLM完整响应: {response_text}")
        
        # 解析JSON响应
        import json
        import re
        
        # 尝试提取JSON（可能被其他文本包裹）
        # 方法1: 尝试直接解析整个响应
        result_data = None
        try:
            result_data = json.loads(response_text)
        except json.JSONDecodeError:
            # 方法2: 使用平衡括号匹配提取JSON
            # 找到第一个 { 并追踪括号平衡
            start_idx = response_text.find('{')
            if start_idx >= 0:
                brace_count = 0
                for i in range(start_idx, len(response_text)):
                    if response_text[i] == '{':
                        brace_count += 1
                    elif response_text[i] == '}':
                        brace_count -= 1
                        if brace_count == 0:
                            # 找到匹配的右括号
                            json_text = response_text[start_idx:i+1]
                            try:
                                result_data = json.loads(json_text)
                                break
                            except json.JSONDecodeError:
                                continue
        
        if result_data:
            # 记录解析成功的JSON结构
            logger.info(f"🔍 JSON解析成功，顶层keys: {list(result_data.keys())}")
            if task_logger:
                task_logger.info(f"🔍 JSON解析成功，顶层keys: {list(result_data.keys())}")
                task_logger.debug(f"🔍 完整JSON数据: {result_data}")
            
            # 新格式: {"selected_tables": [...], "selected_columns": {...}, "gates": {...}, "reasoning": {...}}
            selected_tables = result_data.get("selected_tables", [])
            selected_columns_raw = result_data.get("selected_columns", {})
            gates = result_data.get("gates", {})
            reasoning = result_data.get("reasoning", {})
            
            # 将selected_tables转换为Dict[table_name -> bool]格式
            results = {}
            columns_map = {}  # Dict[table_name -> List[str]]
            all_table_names = list(table_schemas_dict.keys())
            
            # 🔧 智能表名匹配：处理表名格式不一致的情况
            # 例如：LLM返回"db.table"，但实际表名是"db.db.table"
            matched_tables = []
            unmatched_tables = []
            llm_to_actual = {}  # LLM表名 -> 实际表名的映射，用于后续列名绑定
            for llm_table in selected_tables:
                matched = False
                for actual_table in all_table_names:
                    # 尝试多种匹配策略
                    if (actual_table == llm_table or  # 精确匹配
                        actual_table.endswith(llm_table) or  # 后缀匹配
                        llm_table in actual_table):  # 包含匹配
                        matched_tables.append(actual_table)
                        llm_to_actual[llm_table] = actual_table
                        matched = True
                        break
                if not matched:
                    unmatched_tables.append(llm_table)
            
            # 🔧 构建 columns_map：将 selected_columns 中的列绑定到实际表名
            if selected_columns_raw and isinstance(selected_columns_raw, dict):
                for llm_table, cols in selected_columns_raw.items():
                    if not isinstance(cols, list):
                        continue
                    # 找到对应的实际表名
                    actual_table = llm_to_actual.get(llm_table)
                    if actual_table is None:
                        # 尝试和实际表名做模糊匹配
                        for at in all_table_names:
                            if at == llm_table or at.endswith(llm_table) or llm_table in at:
                                actual_table = at
                                break
                    if actual_table and cols:
                        columns_map[actual_table] = cols
                        logger.debug(f"   列粒度: {actual_table} → {cols}")
            
            # 如果有未匹配的表，记录警告
            if unmatched_tables:
                logger.warning(f"⚠️  无法匹配以下LLM返回的表名: {unmatched_tables}")
                if task_logger:
                    task_logger.warning(f"⚠️  无法匹配以下LLM返回的表名: {unmatched_tables}")
            
            # 🔥 分片表共现扩展：如果partition_cooccurrence=1，扩展时间分片表
            partition_cooccurrence = gates.get("partition_cooccurrence", 0)
            if partition_cooccurrence == 1 and matched_tables and all_available_tables:
                detector = PartitionTableDetector()
                
                # 优先从LLM响应中获取时间范围（更准确）
                time_range_dict = result_data.get("time_range", None)
                if time_range_dict and isinstance(time_range_dict, dict):
                    time_range = (time_range_dict.get("start"), time_range_dict.get("end"))
                else:
                    # Fallback: 使用规则从query中提取
                    time_range = detector.extract_time_range_from_query(task)
                
                if time_range and time_range[0] and time_range[1]:
                    time_source = "LLM输出" if time_range_dict else "规则提取"
                    logger.info(f"🔄 检测到partition_cooccurrence=1，时间范围: {time_range[0]} 到 {time_range[1]} (来源: {time_source})")
                    if task_logger:
                        task_logger.info(f"🔄 检测到partition_cooccurrence=1，时间范围: {time_range[0]} 到 {time_range[1]} (来源: {time_source})")
                    
                    expanded_tables = set(matched_tables)
                    partition_groups = {}
                    
                    # 对每个匹配的表检测是否为分片表
                    for table in matched_tables:
                        partition_info = detector.detect_partition_pattern(table)
                        if partition_info:
                            base_table = partition_info['base_table']
                            
                            # 生成该分片表的完整范围
                            needed_partitions = detector.generate_partition_range(
                                base_table,
                                partition_info['granularity'],
                                partition_info['date_format'],
                                time_range[0],
                                time_range[1],
                                all_available_tables
                            )
                            
                            if needed_partitions:
                                partition_groups[base_table] = needed_partitions
                                expanded_tables.update(needed_partitions)
                                
                                logger.info(f"   📊 扩展 {base_table}: {len(needed_partitions)} 个分片")
                                if task_logger:
                                    task_logger.info(f"   📊 扩展 {base_table}: 从1个 → {len(needed_partitions)} 个分片")
                    
                    # 更新matched_tables为扩展后的列表
                    if len(expanded_tables) > len(matched_tables):
                        original_count = len(matched_tables)
                        matched_tables = list(expanded_tables)
                        logger.info(f"✅ 分片表共现扩展完成: {original_count} → {len(matched_tables)} 个表")
                        if task_logger:
                            task_logger.info(f"✅ 分片表共现扩展完成: {original_count} → {len(matched_tables)} 个表")
                            for group, partitions in partition_groups.items():
                                task_logger.debug(f"   {group}: {partitions[:5]}... (共{len(partitions)}个)")
            
            # 标记匹配的表为True（包括扩展后的分片表）
            # 🔥 关键修复：确保扩展的分片表也被添加到results中
            for table_name in all_table_names:
                results[table_name] = table_name in matched_tables
            
            # 🔥 将扩展的分片表（不在原始候选表中）也添加到results
            for table_name in matched_tables:
                if table_name not in results:
                    results[table_name] = True
            
            # 统计结果
            relevant_count = len(matched_tables)
            logger.info(f"✅ 批量表相关性判断: 解析成功，选中 {relevant_count}/{batch_size} 个表")
            logger.info(f"   LLM返回的表: {selected_tables}")
            logger.info(f"   匹配到的实际表: {matched_tables}")
            logger.info(f"   Gates: join={gates.get('join', 0)}, predicate={gates.get('predicate', 0)}, aggregation={gates.get('aggregation', 0)}, partition_cooccurrence={gates.get('partition_cooccurrence', 0)}")
            if task_logger:
                task_logger.info(f"✅ 批量表相关性判断: 解析成功，选中 {relevant_count}/{batch_size} 个表")
                task_logger.info(f"   LLM返回的表: {selected_tables}")
                task_logger.info(f"   匹配到的实际表: {matched_tables}")
                task_logger.info(f"   Gates: join={gates.get('join', 0)}, predicate={gates.get('predicate', 0)}, aggregation={gates.get('aggregation', 0)}, partition_cooccurrence={gates.get('partition_cooccurrence', 0)}")
                if reasoning:
                    task_logger.debug(f"   Reasoning: {reasoning}")
                if columns_map:
                    task_logger.info(f"   列粒度映射 ({len(columns_map)} 个表): { {t: cols for t, cols in list(columns_map.items())[:3]} }")
            
            return results, columns_map
        else:
            logger.warning(f"⚠️ 无法从LLM响应中提取有效JSON，使用fallback策略")
            logger.warning(f"   响应文本长度: {len(response_text)}, 第一个{{位置: {response_text.find('{')}")
            if task_logger:
                task_logger.warning(f"⚠️ 无法从LLM响应中提取有效JSON，使用fallback策略")
                task_logger.warning(f"   响应文本: {response_text[:1000]}")
            
            # Fallback: 使用原有的logits方法（无列粒度信息）
            return _fallback_batch_rerank(task, table_schemas_dict, rerank_components, task_logger), {}
        
    except Exception as e:
        logger.error(f"批量rerank判断失败: {e}")
        if task_logger:
            task_logger.error(f"批量rerank判断失败: {e}")
        import traceback
        logger.error(traceback.format_exc())
        
        # 出错时使用fallback（无列粒度信息）
        return _fallback_batch_rerank(task, table_schemas_dict, rerank_components, task_logger), {}


def _fallback_batch_rerank(task, table_schemas_dict, rerank_components, task_logger=None):
    """
    Fallback策略：使用compute_logits的批量判断
    """
    table_names = list(table_schemas_dict.keys())
    
    logger.info(f"🔄 使用fallback策略: compute_logits批量判断")
    if task_logger:
        task_logger.info(f"🔄 使用fallback策略: compute_logits批量判断")
    
    try:
        # 构建简化的prompt
        tables_text = "\n\n".join([
            f"Table {i}: {name}\n{schema[:300]}..."
            for i, (name, schema) in enumerate(table_schemas_dict.items(), 1)
        ])
        
        instruction = "You are doing table level schema linking. Given multiple tables, judge if they are relevant to the task."
        pairs = [(task, tables_text)]
        inputs = process_inputs(
            pairs, instruction,
            rerank_components['max_length'],
            rerank_components['suffix_tokens'],
            rerank_components['tokenizer']
        )
        
        score = compute_logits(
            rerank_components['model'],
            inputs,
            rerank_components['sampling_params'],
            rerank_components['true_token'],
            rerank_components['false_token']
        )[0]
        
        # 简化策略：整体相关则都相关
        is_relevant = score > 0.5
        logger.info(f"✅ Fallback判断完成: 整体得分 {score:.3f}, 判断结果: {'相关' if is_relevant else '不相关'}")
        if task_logger:
            task_logger.info(f"✅ Fallback判断完成: 整体得分 {score:.3f}")
        
        return {table_name: is_relevant for table_name in table_names}
        
    except Exception as e:
        logger.error(f"Fallback策略也失败: {e}")
        # 最终fallback：默认都不相关
        return {table_name: False for table_name in table_names}


def rerank_single_table(task, table_schema, rerank_components, parent_info=None, instruction_suffix=""):
    """
    对单个表进行rerank判断，支持父节点信息增强
    """
    logger.debug(f"🤖 Starting rerank evaluation...")
    logger.debug(f"    📝 Has parent info: {parent_info is not None}")
    
    # expansion时与初始rerank使用完全相同的逻辑
    instruction = "You are doing table level schema linking."
    
    if instruction_suffix:
        instruction += f" Please {instruction_suffix}."
    
    # 简化table信息构建 - 保持schema linking的一致性
    if parent_info:
        # 简单地把父表信息作为额外的schema上下文，但不改变判断本质
        enhanced_table = f"""Parent table information:
{parent_info}

Current table:
{table_schema}"""
        logger.debug(f"    📋 Added related table context ({len(parent_info)} chars)")
    else:
        enhanced_table = table_schema
    
    logger.debug(f"    🎯 Schema linking mode: {'with context' if parent_info else 'standalone'}")
    logger.debug(f"    📏 Input length: {len(enhanced_table)} chars")
    
    # 记录判断的表信息摘要
    table_name_match = re.search(r'Table full name:\s*(.+)', table_schema)
    table_name = table_name_match.group(1) if table_name_match else "unknown"
    logger.info(f"    🔍 Evaluating table: {table_name}")
    
    # 记录任务信息
    task_preview = task[:100] + "..." if len(task) > 100 else task
    logger.info(f"    🎯 Task: {task_preview}")
    
    try:
        pairs = [(task, enhanced_table)]
        inputs = process_inputs(
            pairs, instruction,
            rerank_components['max_length'],
            rerank_components['suffix_tokens'],
            rerank_components['tokenizer']
        )
        
        logger.info(f"🤖 表相关性判断: 正在调用LLM评估表 {table_name}")
        logger.debug(f"    💬 Sending to GPTChat for evaluation...")
        score = compute_logits(
            rerank_components['model'],
            inputs,
            rerank_components['sampling_params'],
            rerank_components['true_token'],
            rerank_components['false_token']
        )[0]
        logger.info(f"✅ 表相关性判断: LLM评估完成，表 {table_name} 得分 {score:.3f}")
        
        # 判断是否相关（score > 0.5表示更倾向于True）
        is_relevant = score > 0.5
        confidence = "high" if abs(score - 0.5) > 0.3 else "medium" if abs(score - 0.5) > 0.1 else "low"
        
        logger.info(f"    📊 Final Decision: {table_name} → {'RELEVANT' if is_relevant else 'NOT RELEVANT'} (score: {score:.3f}, confidence: {confidence})")
        logger.debug(f"    📊 Rerank result: score={score:.3f}, relevant={is_relevant}, confidence={confidence}")
        return is_relevant
        
    except Exception as e:
        logger.warning(f"❌ Error in rerank evaluation: {e}")
        return False


def expand_via_subquery_decomposition(task, rerank_components, database_graphs, 
                                     table_descriptions, processed_tables, use_description, 
                                     max_consecutive_failures, all_embeddings=None, 
                                     current_table_set=None, expansion_search_count=None,
                                     example_root=None, current_example_id=None, 
                                     embedding_model=None, task_logger=None, top_k_preselection=5,
                                     enable_topk_rerank=False, use_coverage_bonus=False, coverage_beta=0.3,
                                     enable_batch_rerank=False, batch_size=10,
                                     conditional_manager=None, query_context=None, workload_weight=0.1,
                                          edge_workload_weight=None, node_workload_weight=None,
                                     enable_graph_topology=True):
    """
    基于子查询分解的表搜索和扩展方法 - 优化版本：预选Top-K表作为全局初始节点
    
    Args:
        task: 原始任务
        rerank_components: LLM组件
        database_graphs: 数据库图字典
        table_descriptions: 表描述字典
        processed_tables: 已处理表集合
        use_description: 是否使用描述
        max_consecutive_failures: 最大连续失败次数
        all_embeddings: 预计算的嵌入向量
        current_table_set: 当前表集合
        expansion_search_count: 扩展搜索计数
        example_root: 样本根目录
        current_example_id: 当前样本ID
        embedding_model: 嵌入模型
        top_k_preselection: 预选的Top-K表数量，默认为5
        enable_topk_rerank: 是否对Top-K预选表进行LLM判断，默认False（直接标记为相关）
        use_coverage_bonus: 是否使用Coverage Bonus增强多样性 (新增)
        coverage_beta: Coverage Bonus权重系数 (新增)
    
    Returns:
        List[Dict]: 选中的表列表
    """
    # 🔬 处理消融实验参数
    if edge_workload_weight is None:
        edge_workload_weight = workload_weight
    if node_workload_weight is None:
        node_workload_weight = workload_weight
    logger.info(f"🚀 开始基于子查询分解的表搜索 (优化版本)")
    logger.info(f"🎯 原始任务: {task[:100]}...")
    logger.info(f"⚙️ Top-K预选数量: {top_k_preselection}")
    
    if task_logger:
        task_logger.info(f"🎯 原始任务: {task}")
        task_logger.info("=" * 80)
        task_logger.info("🚀 启用优化版本：预选Top-K表作为全局初始节点")
        task_logger.info("=" * 80)

    # ========== 🎯 新增：Step 0 - 预选Top-K最相关表作为全局初始节点 ==========
    logger.info("🔍 Step 0: 预选Top-K最相关表作为全局初始节点")
    if task_logger:
        task_logger.info("第0步: 预选Top-K最相关表")
        task_logger.info("=" * 80)
    
    # 1. 收集所有可用的表
    all_table_names = []
    table_to_db_mapping = {}  # 记录表到数据库的映射
    for db_name, graph in database_graphs.items():
        for table_name in graph.nodes():
            all_table_names.append(table_name)
            table_to_db_mapping[table_name] = db_name
    
    if task_logger:
        task_logger.info(f"📊 从 {len(database_graphs)} 个数据库图中收集了 {len(all_table_names)} 个表")
        task_logger.info(f"🎯 开始计算与原始任务的语义相似度...")

    # 2. 初始化embedding模型（如果需要）
    if embedding_model is None:
        logger.info("🔧 初始化embedding模型用于Top-K预选")
        embedding_model = get_embedding_model()

    # 3. 计算原始任务的embedding（用于Top-K预选和后续权重更新）
    logger.info("🤖 编码原始任务...")
    task_embedding = embedding_model.encode(task, normalize_embeddings=True)
    logger.info("✅ 原始任务embedding计算完成")
    
    # 4. 计算所有表与原始任务的语义相似度
    table_similarities = []
    precomputed_count = 0
    computed_count = 0
    
    logger.info(f"📈 计算 {len(all_table_names)} 个表与原始任务的语义相似度...")
    
    for table_name in all_table_names:
        try:
            # 构建表的文本表示
            table_text = _build_table_text_representation(table_name, table_descriptions)
            if use_description and table_name in table_descriptions and table_descriptions[table_name]:
                table_text = f"{table_text}\n\nTable Description: {table_descriptions[table_name]}"
            
            # 优先使用预计算的embedding
            similarity = 0.0
            if all_embeddings and table_name in all_embeddings:
                try:
                    table_embedding = all_embeddings[table_name]
                    similarity = float(np.dot(task_embedding, table_embedding))
                    precomputed_count += 1
                except Exception as e:
                    logger.debug(f"使用预计算embedding失败 {table_name}: {e}")
                    # 降级到实时计算
                    table_embedding = embedding_model.encode(table_text, normalize_embeddings=True)
                    similarity = float(np.dot(task_embedding, table_embedding))
                    computed_count += 1
            else:
                # 实时计算embedding
                table_embedding = embedding_model.encode(table_text, normalize_embeddings=True)
                similarity = float(np.dot(task_embedding, table_embedding))
                computed_count += 1
            
            table_similarities.append({
                'table_name': table_name,
                'similarity': similarity,
                'db_name': table_to_db_mapping[table_name],
                'has_description': table_name in table_descriptions and bool(table_descriptions[table_name])
            })
            
        except Exception as e:
            logger.warning(f"计算表 {table_name} 相似度失败: {e}")
            table_similarities.append({
                'table_name': table_name,
                'similarity': 0.0,
                'db_name': table_to_db_mapping[table_name],
                'has_description': False
            })
    
    logger.info(f"📊 相似度计算完成: {precomputed_count} 个预计算, {computed_count} 个实时计算")
    
    # 5. 选择Top-K最相关的表
    table_similarities.sort(key=lambda x: x['similarity'], reverse=True)
    top_k_tables = table_similarities[:top_k_preselection]
    
    logger.info(f"🏆 预选Top-{top_k_preselection}最相关表:")
    if task_logger:
        task_logger.info(f"🏆 预选Top-{top_k_preselection}最相关表:")
    
    for i, table_info in enumerate(top_k_tables, 1):
        desc_status = "📝" if table_info['has_description'] else "📄"
        log_msg = f"  {i}. {desc_status} {table_info['table_name']} (相似度: {table_info['similarity']:.4f}, DB: {table_info['db_name']})"
        logger.info(log_msg)
        if task_logger:
            task_logger.info(log_msg)
    
    # 6. 🚀 可选：对Top-K表进行LLM判断或直接标记为相关
    preselected_tables = []
    global_initial_nodes = set()
    
    if enable_topk_rerank and enable_batch_rerank:
        # 🚀 批量LLM判断模式
        logger.info(f"🤖 Top-K预选表处理模式: 批量LLM判断 ({len(top_k_tables)} 个表)")
        if task_logger:
            task_logger.info(f"🤖 Top-K预选表处理模式: 批量LLM判断 ({len(top_k_tables)} 个表)")
        
        # 构建批量schema字典
        batch_schemas = {}
        for table_info in top_k_tables:
            table_name = table_info['table_name']
            table_text = _build_table_text_representation(table_name, table_descriptions)
            if use_description and table_name in table_descriptions and table_descriptions[table_name]:
                table_text = f"{table_text}\n\nTable Description: {table_descriptions[table_name]}"
            batch_schemas[table_name] = table_text
        
        # 收集所有可用表（用于分片表扩展）
        all_available_tables = set()
        for db_name, graph in database_graphs.items():
            all_available_tables.update(graph.nodes())
        
        # 构建 Top-K 表之间的合并拓扑图（跨数据库的边 union）
        topk_names_set = set(batch_schemas.keys())
        topk_merged_graph = nx.Graph()
        for _db_name, _g in database_graphs.items():
            for u, v, data in _g.edges(data=True):
                if u in topk_names_set and v in topk_names_set:
                    topk_merged_graph.add_edge(u, v, **data)
        topk_graph_for_prompt = topk_merged_graph if topk_merged_graph.number_of_edges() > 0 else None

        # 一次性批量判断所有Top-K表
        try:
            # 返回 (relevance_dict, columns_map)
            batch_results, batch_columns_map = rerank_batch_tables(
                task, batch_schemas, rerank_components,
                parent_info=None, 
                instruction_suffix="judge whether these top-k candidate tables are relevant to the task",
                task_logger=task_logger,
                all_available_tables=all_available_tables,
                graph=topk_graph_for_prompt,
                enable_graph_topology=enable_graph_topology
            )
            
            # 处理批量判断结果
            top_k_table_names = [info['table_name'] for info in top_k_tables]
            for i, table_info in enumerate(top_k_tables, 1):
                table_name = table_info['table_name']
                is_relevant = batch_results.get(table_name, False)
                
                if is_relevant:
                    answer = "Y"
                    score = 0.9
                    think = f"Pre-selected by top-k similarity (#{i}) and confirmed by batch LLM"
                    selection_method = "top_k_preselection_batch_confirmed"
                    global_initial_nodes.add(table_name)
                    
                    logger.info(f"✅ 预选表 #{i} {table_name} 批量判断为相关")
                    if task_logger:
                        task_logger.info(f"✅ 预选表 #{i} {table_name} 批量判断为相关")
                else:
                    answer = "N"
                    score = 0.3
                    think = f"Pre-selected by top-k similarity (#{i}) but rejected by batch LLM"
                    selection_method = "top_k_preselection_batch_rejected"
                    
                    logger.info(f"❌ 预选表 #{i} {table_name} 批量判断为不相关")
                    if task_logger:
                        task_logger.info(f"❌ 预选表 #{i} {table_name} 批量判断为不相关")
                
                # 添加到预选结果
                preselected_tables.append({
                    "think": think,
                    "answer": answer,
                    "columns": batch_columns_map.get(table_name, []) if is_relevant else [],
                    "table name": table_name,
                    "score": score,
                    "similarity": table_info['similarity'],
                    "expansion_level": 0,
                    "selection_method": selection_method,
                    "db_name": table_info['db_name']
                })
            
            # 🔥 处理扩展的分片表（不在top_k中但在batch_results中为True的表）
            # 注意：不加入global_initial_nodes，不影响搜索路径，但加入最终schema
            expanded_tables = [table for table, is_relevant in batch_results.items() 
                             if is_relevant and table not in top_k_table_names]
            if expanded_tables:
                logger.info(f"🎯 分片表扩展: 添加 {len(expanded_tables)} 个扩展表到最终schema（不影响搜索路径）")
                if task_logger:
                    task_logger.info(f"🎯 分片表扩展: 添加 {len(expanded_tables)} 个扩展表到最终schema（不影响搜索路径）")
                
                # 推断数据库名（从原始top_k_tables中的表）
                db_name = top_k_tables[0]['db_name'] if top_k_tables else None
                
                for table in expanded_tables:
                    preselected_tables.append({
                        "think": f"Auto-expanded partition table (co-occurrence, not in search path)",
                        "answer": "Y",
                        "columns": batch_columns_map.get(table, []),
                        "table name": table,
                        "score": 0.9,
                        "similarity": 0.0,  # 扩展的表没有相似度分数
                        "expansion_level": 0,
                        "selection_method": "top_k_preselection_batch_partition_expanded",
                        "db_name": db_name
                    })
                    
                    logger.info(f"   ✅ 扩展表 {table} 添加到最终schema")
                    if task_logger:
                        task_logger.info(f"   ✅ 扩展表 {table} 添加到最终schema")
        
        except Exception as e:
            logger.error(f"批量判断Top-K表失败: {e}，回退到全部标记为相关")
            if task_logger:
                task_logger.error(f"批量判断Top-K表失败: {e}，回退到全部标记为相关")
            
            # 出错时回退：全部标记为相关
            for i, table_info in enumerate(top_k_tables, 1):
                table_name = table_info['table_name']
                answer = "Y"
                score = 0.85
                think = f"Pre-selected by top-k similarity (#{i}) - batch error, fallback to relevant"
                selection_method = "top_k_preselection_batch_fallback"
                global_initial_nodes.add(table_name)
                
                preselected_tables.append({
                    "think": think,
                    "answer": answer,
                    "columns": [],
                    "table name": table_name,
                    "score": score,
                    "similarity": table_info['similarity'],
                    "expansion_level": 0,
                    "selection_method": selection_method,
                    "db_name": table_info['db_name']
                })
    
    elif enable_topk_rerank and not enable_batch_rerank:
        # 🔄 逐个LLM判断模式（默认）
        logger.info(f"🤖 Top-K预选表处理模式: 逐个LLM判断 ({len(top_k_tables)} 个表)")
        if task_logger:
            task_logger.info(f"🤖 Top-K预选表处理模式: 逐个LLM判断 ({len(top_k_tables)} 个表)")
        
        # 逐个判断每个Top-K表
        for i, table_info in enumerate(top_k_tables, 1):
            table_name = table_info['table_name']
            table_text = _build_table_text_representation(table_name, table_descriptions)
            if use_description and table_name in table_descriptions and table_descriptions[table_name]:
                table_text = f"{table_text}\n\nTable Description: {table_descriptions[table_name]}"
            
            try:
                # 调用单表判断
                is_relevant = rerank_single_table(
                    task, table_text, rerank_components,
                    parent_info=None
                )
                
                if is_relevant:
                    answer = "Y"
                    score = 0.9
                    think = f"Pre-selected by top-k similarity (#{i}) and confirmed by LLM"
                    selection_method = "top_k_preselection_confirmed"
                    global_initial_nodes.add(table_name)
                    
                    logger.info(f"✅ 预选表 #{i} {table_name} 逐个判断为相关")
                    if task_logger:
                        task_logger.info(f"✅ 预选表 #{i} {table_name} 逐个判断为相关")
                else:
                    answer = "N"
                    score = 0.3
                    think = f"Pre-selected by top-k similarity (#{i}) but rejected by LLM"
                    selection_method = "top_k_preselection_rejected"
                    
                    logger.info(f"❌ 预选表 #{i} {table_name} 逐个判断为不相关")
                    if task_logger:
                        task_logger.info(f"❌ 预选表 #{i} {table_name} 逐个判断为不相关")
                
            except Exception as e:
                logger.warning(f"判断表 {table_name} 失败: {e}，标记为相关")
                answer = "Y"
                score = 0.85
                think = f"Pre-selected by top-k similarity (#{i}) - error, fallback to relevant"
                selection_method = "top_k_preselection_fallback"
                global_initial_nodes.add(table_name)
            
            # 添加到预选结果
            preselected_tables.append({
                "think": think,
                "answer": answer,
                "columns": [],
                "table name": table_name,
                "score": score,
                "similarity": table_info['similarity'],
                "expansion_level": 0,
                "selection_method": selection_method,
                "db_name": table_info['db_name']
            })
    
    else:
        # 🚀 默认模式：直接标记为相关
        logger.info(f"🤖 Top-K预选表处理模式: 直接标记为相关")
        if task_logger:
            task_logger.info(f"🤖 Top-K预选表处理模式: 直接标记为相关")
        
        for i, table_info in enumerate(top_k_tables, 1):
            table_name = table_info['table_name']
            answer = "Y"
            score = 0.95
            think = f"Pre-selected by top-k similarity ranking (#{i})"
            selection_method = "top_k_preselection"
            global_initial_nodes.add(table_name)
            
            # 添加到预选结果
            preselected_tables.append({
                "think": think,
                "answer": answer,
                "columns": [],
                "table name": table_name,
                "score": score,
                "similarity": table_info['similarity'],
                "expansion_level": 0,
                "selection_method": selection_method,
                "db_name": table_info['db_name']
            })
    
    # 🚀 重要修改：预选表不添加到processed_tables，允许作为起始节点进行扩展
    # processed_tables.add(table_name)  # 注释掉：预选表需要能够进行邻居扩展
    
    # 统计相关和不相关的表
    relevant_count = sum(1 for t in preselected_tables if t['answer'] == 'Y')
    irrelevant_count = len(preselected_tables) - relevant_count
    
    logger.info(f"✅ 预选阶段完成：评估了 {len(preselected_tables)} 个表")
    logger.info(f"   ✅ 相关: {relevant_count} 个")
    if irrelevant_count > 0:
        logger.info(f"   ❌ 不相关: {irrelevant_count} 个")
    
    # 根据模式输出不同的日志
    if enable_topk_rerank and enable_batch_rerank:
        logger.info(f"   📊 使用批量LLM判断 (1次调用评估{len(top_k_tables)}个表)")
    elif enable_topk_rerank and not enable_batch_rerank:
        logger.info(f"   📊 使用逐个LLM判断 ({len(top_k_tables)}次调用)")
    else:
        logger.info(f"   📊 直接基于embedding相似度标记")
    
    if task_logger:
        task_logger.info(f"✅ 预选阶段完成：评估了 {len(preselected_tables)} 个表")
        task_logger.info(f"   ✅ 相关: {relevant_count} 个")
        if irrelevant_count > 0:
            task_logger.info(f"   ❌ 不相关: {irrelevant_count} 个")
        task_logger.info(f"🎯 全局初始节点集合 ({len(global_initial_nodes)}个): {list(global_initial_nodes)}")

    # ========== 🆕 增强的任务分解逻辑（一次性分解+分析） ==========
    if task_logger:
        task_logger.info("=" * 80)
        task_logger.info("第一步: 任务分解+查询特征分析")
        task_logger.info("=" * 80)
        task_logger.info("🚀 使用增强版本：一次LLM调用完成分解+分析")

    # 🆕 使用新的分解函数：一次性完成分解+分析
    subqueries_with_analysis = decompose_task_with_analysis(task, rerank_components, all_table_names, task_logger)
    
    if task_logger:
        task_logger.info(f"✅ 任务分解+分析完成，共 {len(subqueries_with_analysis)} 个步骤")
    
    # 初始化结果集合，包含预选的表
    all_selected_tables = preselected_tables.copy()  # 🚀 包含预选表
    all_selected_names = global_initial_nodes.copy()  # 🚀 包含预选表名
    all_evaluated_tables = []  # 跟踪所有评估过的表
    
    # 为每个数据库图创建副本，用于动态权重更新
    working_graphs = {}
    for db_name, graph in database_graphs.items():
        working_graphs[db_name] = graph.copy()
    
    # 加载预计算的表embedding以提高效率
    if task_logger:
        task_logger.info("📊 加载预计算的表embedding以提高搜索效率...")
    _load_precomputed_table_embeddings(working_graphs, example_root, current_example_id)
    
    # 🎯 准备节点与总任务的相似度字典（用于权重更新）
    # 从之前计算的 table_similarities 中提取
    node_task_similarities = {}
    for table_info in table_similarities:
        table_name = table_info['table_name']
        similarity = table_info['similarity']
        node_task_similarities[table_name] = similarity
    
    if task_logger:
        task_logger.info(f"📊 准备节点-总任务相似度字典: {len(node_task_similarities)} 个节点")
    
    # 2. 🔄 层级化子查询处理 - 使用预选的全局初始节点
    current_layer_nodes = global_initial_nodes.copy()  # 🚀 使用预选的节点作为初始层
    layer_history = []  # 记录每层的搜索历史
    
    if task_logger:
        task_logger.info("=" * 80)
        task_logger.info("第二步: 分层扩展搜索")
        task_logger.info("=" * 80)
        task_logger.info(f"🎯 使用预选的 {len(current_layer_nodes)} 个表作为全局初始节点")
    
    # 🆕 遍历子查询及其分析结果
    for subquery_item in subqueries_with_analysis:
        i = subquery_item['index']
        step = subquery_item['subquery']
        
        # 🆕 从LLM分析结果构建QueryContext（如果有conditional_manager）
        step_query_context = None
        if conditional_manager:
            from conditional_functions import QueryContext
            step_query_context = QueryContext(
                has_aggregation=subquery_item.get('has_aggregation', False),
                aggregation_types=subquery_item.get('aggregation_types', []),
                has_join=subquery_item.get('has_join', False),
                has_groupby=subquery_item.get('has_groupby', False),
                has_orderby=subquery_item.get('has_orderby', False),
                comparison_ops=subquery_item.get('comparison_ops', []),
                has_time_filter=subquery_item.get('has_time_filter', False),
                entity_types=subquery_item.get('entity_types', []),
                query_complexity=subquery_item.get('query_complexity', 'simple')
            )
        
        logger.info(f"\n🎯 处理第 {i} 层步骤: {step}")
        if step_query_context:
            logger.info(f"   📊 LLM分析结果:")
            logger.info(f"      - 聚合: {step_query_context.has_aggregation} {step_query_context.aggregation_types}")
            logger.info(f"      - 连接: {step_query_context.has_join}")
            logger.info(f"      - 分组: {step_query_context.has_groupby}")
            logger.info(f"      - 过滤: {step_query_context.comparison_ops}")
            logger.info(f"      - 实体: {step_query_context.entity_types}")
            logger.info(f"      - 复杂度: {step_query_context.query_complexity}")
        
        if task_logger:
            task_logger.info("=" * 80)
            task_logger.info(f"第 {i} 层: 处理步骤 {i}/{len(subqueries_with_analysis)}")
            task_logger.info("=" * 80)
            task_logger.info(f"🎯 当前步骤: {step}")
            if step_query_context:
                task_logger.info(f"📊 LLM分析结果:")
                task_logger.info(f"   聚合={step_query_context.has_aggregation}, 连接={step_query_context.has_join}, "
                               f"分组={step_query_context.has_groupby}, 复杂度={step_query_context.query_complexity}")
            if current_layer_nodes:
                task_logger.info(f"📍 基于上一层的 {len(current_layer_nodes)} 个节点进行扩展: {list(current_layer_nodes)[:5]}...")
            else:
                task_logger.info("📍 当前层没有起始节点")
            task_logger.info("开始动态权重更新...")
        
        # 2.1 优化的动态权重更新 - 只计算一次子查询embedding
        logger.info(f"🔄 更新图权重...")
        
        # 只计算一次步骤的embedding
        step_embedding = None
        if embedding_model is not None:
            try:
                step_embedding = embedding_model.encode(step, normalize_embeddings=True)
                if task_logger:
                    task_logger.info(f"📊 计算步骤embedding: {step[:50]}...")
            except Exception as e:
                logger.warning(f"计算步骤embedding失败: {e}")
        
        # 智能图权重更新：只更新当前任务相关的数据库图
        if task_logger:
            task_logger.info("🎯 识别任务相关的数据库...")
        
        # 使用智能函数识别相关数据库
        relevant_databases = _identify_relevant_databases(
            working_graphs, current_table_set, example_root, current_example_id
        )
        
        if task_logger:
            task_logger.info(f"📊 识别到相关数据库: {list(relevant_databases)}")
            if len(relevant_databases) == 1:
                db_name = list(relevant_databases)[0]
                graph = working_graphs.get(db_name)
                if graph:
                    task_logger.info(f"  📋 数据库 {db_name}: {len(graph.nodes())} 个表, {len(graph.edges())} 个边")
            else:
                for db_name in relevant_databases:
                    graph = working_graphs.get(db_name)
                    if graph:
                        task_logger.info(f"  📋 数据库 {db_name}: {len(graph.nodes())} 个表")
        
        # 只更新相关数据库的图权重
        updated_graphs = 0
        for db_name in relevant_databases:
            if db_name in working_graphs and working_graphs[db_name].nodes():
                # 🎯 传入预计算的节点-总任务相似度（在Top-K预选阶段已计算）
                working_graphs[db_name] = update_graph_weights_for_subquery_optimized(
                    working_graphs[db_name], step, table_descriptions, embedding_model, 
                    precomputed_subquery_embedding=step_embedding,
                    node_task_similarities=node_task_similarities,
                    conditional_manager=conditional_manager,  # 🆕
                    query_context=step_query_context,  # 🆕 使用子查询特定的context
                    workload_weight=workload_weight,  # 🆕

                    edge_workload_weight=edge_workload_weight,  # 🔬 λ: 边增强权重

                    node_workload_weight=node_workload_weight  # 🔬 γ: 节点先验权重
                )
                updated_graphs += 1
                if task_logger:
                    task_logger.info(f"  ✅ 更新数据库 {db_name} 的图权重")
        
        if task_logger:
            task_logger.info(f"✅ 权重更新完成：实际更新了 {updated_graphs}/{len(working_graphs)} 个数据库图")
        
        # 2.2 🚀 优化：确定当前层的起始节点 - 使用预选节点或上层结果
        layer_start_nodes = set()
        
        if current_layer_nodes:
            # 使用当前层的节点作为起始节点（可能是预选节点或上一层结果）
            layer_start_nodes = current_layer_nodes.copy()
            if task_logger:
                if i == 1:
                    task_logger.info(f"🎯 第一层使用预选的 {len(layer_start_nodes)} 个全局初始节点作为起始点")
                else:
                    task_logger.info(f"🔗 第{i}层使用上一层的 {len(layer_start_nodes)} 个节点作为起始点")
        else:
            # 异常情况：当前层没有起始节点，回退到原有逻辑
            logger.warning(f"⚠️ 第 {i} 层没有起始节点，回退到子查询节点选择")
            if task_logger:
                task_logger.warning(f"⚠️ 第{i}层没有起始节点，回退到子查询节点选择")
            
            for db_name in relevant_databases:
                if db_name in working_graphs and working_graphs[db_name].nodes():
                    start_nodes = select_best_start_node_for_subquery(
                        step, working_graphs[db_name], table_descriptions, embedding_model,
                        precomputed_subquery_embedding=step_embedding, top_k=1
                    )
                    if start_nodes:
                        layer_start_nodes.update(start_nodes)
                        if task_logger:
                            task_logger.info(f"🔄 第{i}层从数据库 {db_name} 回退选择起始节点: {start_nodes}")
                        break  # 只选择一个数据库的节点作为回退
        
        # 最终检查：如果仍然没有起始节点，跳过当前子查询
        if not layer_start_nodes:
            logger.warning(f"⚠️ 第 {i} 层无法确立起始节点，跳过当前子查询")
            if task_logger:
                task_logger.warning(f"⚠️ 第{i}层无法确立起始节点，跳过当前子查询")
            
            layer_history.append({
                'layer': i,
                'step': step,
                'start_nodes': set(),
                'selected_tables': [],
                'next_layer_nodes': set(),
                'status': 'skipped'
            })
            continue
        
        logger.info(f"✅ 第 {i} 层确定了 {len(layer_start_nodes)} 个起始节点")
        
        if task_logger:
            task_logger.info(f"✅ 第 {i} 层起始节点: {list(layer_start_nodes)}")
        
        # 2.3 🚀 PageRank子图搜索替代传统方法
        logger.info(f"🔍 执行第 {i} 层的PageRank子图搜索...")
        
        if task_logger:
            task_logger.info(f"🎯 第{i}层: 开始PageRank子图搜索")
        
        # 构建相关数据库的工作图字典
        relevant_working_graphs = {}
        for db_name in relevant_databases:
            if db_name in working_graphs and working_graphs[db_name].nodes():
                relevant_working_graphs[db_name] = working_graphs[db_name]
        
        # 使用PageRank方法进行子图搜索
        layer_selected_tables = expand_via_pagerank_subgraph(
            step, relevant_working_graphs, current_layer_nodes,
            table_descriptions, rerank_components, 
            embedding_model, task, use_description,
            use_coverage_bonus, coverage_beta,
            task_logger, processed_tables,
            example_root=example_root, current_example_id=current_example_id,
            enable_batch_rerank=enable_batch_rerank, batch_size=batch_size,
            conditional_manager=conditional_manager,  # 🆕 传递conditional manager
            query_context=step_query_context,  # 🆕 使用子查询特定的context
            workload_weight=workload_weight,  # 🆕 传递workload权重
            edge_workload_weight=edge_workload_weight,  # 🔬 λ: 边增强权重
            node_workload_weight=node_workload_weight,  # 🔬 γ: 节点先验权重
            enable_graph_topology=enable_graph_topology
        )
                        
        # 更新下一层节点：选择当前层相关表作为下一层起始点
        next_layer_nodes = set()
        for table_result in layer_selected_tables:
            if table_result["answer"] == "Y":
                next_layer_nodes.add(table_result["table name"])
                all_selected_names.add(table_result["table name"])
                        
        # PageRank搜索已完成，layer_selected_tables 和 next_layer_nodes 已设置
        
        # 2.4 记录当前层的结果
        layer_result = {
            'layer': i,
            'step': step,
            'start_nodes': layer_start_nodes.copy(),
            'selected_tables': layer_selected_tables.copy(),
            'next_layer_nodes': next_layer_nodes.copy(),
            'status': 'completed' if layer_selected_tables else 'empty'
        }
        layer_history.append(layer_result)
        
        # 更新全局结果
        all_selected_tables.extend(layer_selected_tables)
        for table in layer_selected_tables:
            processed_tables.add(table['table name'])
            all_selected_names.add(table['table name'])
        
        # 更新下一层的起始节点
        current_layer_nodes = next_layer_nodes
        
        # 日志记录
        if task_logger:
            task_logger.info(f"✅ 第 {i} 层完成:")
            task_logger.info(f"  📊 选中表格: {len(layer_selected_tables)} 个")
            task_logger.info(f"  🔗 下一层节点: {len(next_layer_nodes)} 个")
            task_logger.info(f"  📋 状态: {layer_result['status']}")
            if layer_selected_tables:
                selected_names = [t['table name'] for t in layer_selected_tables]
                task_logger.info(f"  📝 选中表名: {selected_names[:3]}{'...' if len(selected_names) > 3 else ''}")
        
        logger.info(f"✅ 第 {i} 层完成: 选中 {len(layer_selected_tables)} 个表，准备 {len(next_layer_nodes)} 个节点用于下一层")
    
    # 记录层级搜索总结
    if task_logger:
        task_logger.info("=" * 80)
        task_logger.info("🎉 层级化子查询搜索完成")
        task_logger.info("=" * 80)
        for layer_result in layer_history:
            task_logger.info(f"第 {layer_result['layer']} 层: {layer_result['status']} - "
                           f"{len(layer_result['selected_tables'])} 个表, "
                           f"{len(layer_result['next_layer_nodes'])} 个下层节点")
        task_logger.info(f"🎯 总结果: {len(all_selected_tables)} 个表被选中")
    
    logger.info(f"🎉 层级化子查询分解搜索完成，总共选中 {len(all_selected_tables)} 个表")
    logger.info(f"📋 选中的表: {[t['table name'] for t in all_selected_tables]}")
    logger.info(f"🔄 层级搜索历史: {len(layer_history)} 层，其中 {sum(1 for l in layer_history if l['status'] == 'completed')} 层成功")
    
    return all_selected_tables


def expand_for_single_subquery_layered(start_node, subquery, rerank_components, graph,
                                      table_descriptions, processed_tables, use_description,
                                      max_failures_per_layer, current_table_set, expansion_search_count,
                                      example_root, current_example_id, global_selected_names,
                                      task_logger=None, layer_num=1, embedding_model=None, subquery_embedding=None, task=None):
    """
    为单个子查询进行层级化扩展搜索，返回选中的表和下一层的节点
    
    Args:
        start_node: 起始节点
        subquery: 当前子查询
        rerank_components: LLM组件
        graph: 当前数据库图
        table_descriptions: 表描述字典
        processed_tables: 已处理表集合
        use_description: 是否使用描述
        max_failures_per_layer: 每层最大连续失败次数
        current_table_set: 当前表集合
        expansion_search_count: 扩展搜索计数
        example_root: 样本根目录
        current_example_id: 当前样本ID
        global_selected_names: 全局已选择的表名集合
        task_logger: 任务日志记录器
        layer_num: 当前层数
        
    Returns:
        Tuple[List[Dict], Set[str]]: (选中的表列表, 下一层节点集合)
    """
    if task_logger:
        task_logger.info(f"🔍 第 {layer_num} 层从节点 {start_node} 开始层级化扩展")
    
    selected_tables = []
    next_layer_nodes = set()
    consecutive_failures = 0
    
    # 🚀 优化：检查起始节点处理逻辑
    # 预选的Top-K表可以作为起始节点进行扩展，但不重复评估自身
    is_preselected = start_node in global_selected_names
    if start_node in processed_tables and not is_preselected:
        if task_logger:
            task_logger.info(f"⏭️ 起始节点 {start_node} 已被处理，跳过")
        return selected_tables, next_layer_nodes
    elif is_preselected:
        if task_logger:
            task_logger.info(f"🎯 起始节点 {start_node} 是预选表，跳过自身评估，直接进行邻居扩展")
    
    # 1. 🚀 优化：起始节点评估逻辑
    try:
        # 获取表schema
        table_schema = None
        
        # 从图节点属性中获取schema_str
        node_data = graph.nodes.get(start_node, {})
        if 'schema_str' in node_data and node_data['schema_str']:
            table_schema = node_data['schema_str']
        elif start_node in table_descriptions:
            # 从table_descriptions获取
            table_schema = table_descriptions[start_node]
        
        if not table_schema:
            if task_logger:
                task_logger.warning(f"⚠️ 无法获取节点 {start_node} 的schema，节点属性: {list(node_data.keys())}")
            return selected_tables, next_layer_nodes
        
        # 🚀 关键修改：预选表跳过自身评估，直接进行邻居扩展
        if is_preselected:
            if task_logger:
                task_logger.info(f"🎯 预选表 {start_node} 跳过自身评估，直接进行邻居扩展")
            # 预选表已经在结果中，不需要重复添加，直接跳到邻居扩展
            start_node_relevant = True
        else:
            # 非预选表需要进行相关性评估
            # 直接使用LLM判断相关性，不进行embedding预筛选
            # 构建包含完整task和当前子查询的判断输入
            task_with_subquery = f"""Original Task: {task}

Current Subquery Step: {subquery}"""
            
            is_relevant = rerank_single_table(
                task_with_subquery, table_schema, rerank_components,
                parent_info=f"Layer {layer_num} start node", instruction_suffix=""
            )
            
            start_node_relevant = is_relevant
            
            if is_relevant:
                # 添加到选中表（使用标准输出格式）
                table_info = {
                    "think": "",
                    "answer": "Y",
                    "columns": [],
                    "table name": start_node,
                    "score": 0.9,
                    'schema': table_schema,
                    'layer': layer_num,
                    'role': 'start_node'
                }
                selected_tables.append(table_info)
                processed_tables.add(start_node)
                global_selected_names.add(start_node)
                
                if task_logger:
                    task_logger.info(f"✅ 第 {layer_num} 层起始节点 {start_node} 相关，已选中")
        
        # 2. 进行邻居扩展（无论起始节点是否相关，都尝试扩展）
        if start_node_relevant:
            
            # 获取邻居节点作为下一层的候选
            neighbors = list(graph.neighbors(start_node))
            valid_neighbors = []
            
            for neighbor in neighbors:
                if (neighbor not in processed_tables and 
                    neighbor not in global_selected_names and
                    (current_table_set is None or neighbor in current_table_set)):
                    valid_neighbors.append(neighbor)
            
            # 按边权重排序邻居节点
            neighbor_weights = []
            for neighbor in valid_neighbors:
                edge_data = graph.get_edge_data(start_node, neighbor)
                weight = edge_data.get('weight', 0.5) if edge_data else 0.5
                neighbor_weights.append((neighbor, weight))
            
            # 按权重降序排序，选择top-k作为下一层节点
            neighbor_weights.sort(key=lambda x: x[1], reverse=True)
            top_neighbors = [n[0] for n in neighbor_weights[:3]]  # 最多选择3个邻居
            next_layer_nodes.update(top_neighbors)
            
            if task_logger:
                task_logger.info(f"🔗 第 {layer_num} 层从 {start_node} 发现 {len(valid_neighbors)} 个邻居，"
                               f"选择top-{len(top_neighbors)} 作为下一层节点: {top_neighbors}")
            
        else:
            # 🚀 记录不相关的表
            irrelevant_table_info = {
                "think": "",
                "answer": "N",
                "columns": [],
                "table name": start_node,
                "score": 0.1,
                'schema': table_schema,
                'layer': layer_num,
                'role': 'start_node_rejected'
            }
            selected_tables.append(irrelevant_table_info)  # 也添加到结果中
            processed_tables.add(start_node)  # 标记为已处理
            
            consecutive_failures += 1
            if task_logger:
                task_logger.info(f"❌ 第 {layer_num} 层起始节点 {start_node} 不相关 (失败次数: {consecutive_failures})")
            
            if consecutive_failures >= max_failures_per_layer:
                if task_logger:
                    task_logger.warning(f"⚠️ 第 {layer_num} 层连续失败 {consecutive_failures} 次，停止扩展")
                return selected_tables, next_layer_nodes
    
    except Exception as e:
        logger.error(f"第 {layer_num} 层扩展出错: {e}")
        if task_logger:
            task_logger.error(f"第 {layer_num} 层扩展出错: {e}")
    
    return selected_tables, next_layer_nodes


def expand_via_minhash(parent_table_name, parent_info, task, rerank_components, 
                      database_graphs, table_descriptions, processed_tables, 
                      use_description, max_consecutive_failures, all_embeddings=None,
                      current_table_set=None, expansion_search_count=None,
                      current_level=1, max_level=3, parent_chain=None,
                      global_candidate_cache=None, duplicate_candidates_avoided=None,
                      successful_paths=None, example_root=None, current_example_id=None):
    """
    基于MinHash相似度扩展到相邻节点，支持多层级递归扩展（保持原有接口兼容性）
    
    Args:
        current_level: 当前扩展层级
        max_level: 最大扩展层级
        parent_chain: 父节点链，包含所有上级节点信息
        successful_paths: Dict[str, List[List[Dict]]] - 记录每个根父表的成功路径
    """
    logger.info(f"🌳 Level {current_level} expansion from parent table: {parent_table_name}")
    
    # 检查是否达到最大层级限制
    if current_level > max_level:
        logger.info(f"🛑 Max level {max_level} reached, stopping recursion")
        return []
    
    # 初始化父节点链
    if parent_chain is None:
        parent_chain = []
    
    # 🔥 路径追踪：确定根父表
    root_parent = parent_chain[0]['table_name'] if parent_chain else parent_table_name
    if successful_paths is not None and root_parent not in successful_paths:
        successful_paths[root_parent] = []
    
    # 添加当前父节点到链中
    current_parent_chain = parent_chain + [{
        'table_name': parent_table_name,
        'info': parent_info,
        'level': current_level - 1
    }]
    
    expanded_tables = []
    
    # 寻找包含该表的数据库图
    logger.debug(f"    🔍 Searching for graph containing {parent_table_name}...")
    parent_graph = None
    for db_name, graph in database_graphs.items():
        if parent_table_name in graph.nodes():
            parent_graph = graph
            logger.debug(f"    ✅ Found in database graph: {db_name}")
            break
    
    if parent_graph is None:
        logger.info(f"    ❌ No graph found containing table {parent_table_name}")
        return expanded_tables
    
    # 获取所有邻居节点，预先过滤已处理和超出范围的表
    logger.debug(f"    🔗 Finding neighbors of {parent_table_name}...")
    all_neighbors = list(parent_graph.neighbors(parent_table_name))
    logger.debug(f"    📊 Total neighbors in graph: {len(all_neighbors)}")
    
    # 🚀 优化：预先过滤，避免重复处理
    # 1. 过滤已处理的表
    unprocessed_neighbors = [n for n in all_neighbors if n not in processed_tables]
    processed_count = len(all_neighbors) - len(unprocessed_neighbors)
    
    # 2. 过滤超出范围的表
    if current_table_set is not None:
        valid_neighbors = [n for n in unprocessed_neighbors if n in current_table_set]
        out_of_scope_count = len(unprocessed_neighbors) - len(valid_neighbors)
    else:
        valid_neighbors = unprocessed_neighbors
        out_of_scope_count = 0
    
    logger.debug(f"    ⏭️  Filtered out {processed_count} already processed neighbors")
    if out_of_scope_count > 0:
        logger.info(f"    🔒 Filtered out {out_of_scope_count} neighbors outside current sample scope")
    
    # 3. 构建候选表列表（只对有效的邻居进行边数据获取）
    neighbors = []
    for neighbor in valid_neighbors:
        edge_data = parent_graph.get_edge_data(parent_table_name, neighbor)
        minhash_sim = edge_data.get('mh_sim', 0.0) if edge_data else 0.0
        
        neighbors.append({
            'table_name': neighbor,
            'minhash_similarity': minhash_sim,
            'edge_data': edge_data
        })
    
    # 🚀 优化：进一步过滤全局候选表缓存中已存在的表
    if global_candidate_cache is not None:
        original_count = len(neighbors)
        neighbors = [n for n in neighbors if n['table_name'] not in global_candidate_cache]
        cache_filtered_count = original_count - len(neighbors)
        
        if cache_filtered_count > 0:
            logger.debug(f"    🗂️  Filtered out {cache_filtered_count} neighbors already in global candidate cache")
            if duplicate_candidates_avoided is not None:
                duplicate_candidates_avoided[0] += cache_filtered_count
        
        # 将当前候选表添加到全局缓存
        for neighbor in neighbors:
            global_candidate_cache.add(neighbor['table_name'])
    
    # 按MinHash相似度降序排序
    neighbors.sort(key=lambda x: x['minhash_similarity'], reverse=True)
    
    if not neighbors:
        logger.info(f"    🚫 No unprocessed neighbors found for {parent_table_name}")
        return expanded_tables
    
    logger.info(f"    🎯 Found {len(neighbors)} candidate neighbors for expansion")
    
    # 显示top候选者
    logger.info(f"    🏆 Top candidates by MinHash similarity:")
    for i, neighbor in enumerate(neighbors[:3]):
        logger.info(f"      {i+1}. {neighbor['table_name']} (MinHash: {neighbor['minhash_similarity']:.3f})")
    if len(neighbors) > 3:
        logger.info(f"      ... and {len(neighbors) - 3} more candidates")
    
    # 准备多层级父节点信息用于子节点判断
    logger.debug(f"    📋 Preparing multi-level parent context for child evaluation...")
    parent_desc = build_multi_parent_context(current_parent_chain, table_descriptions, use_description)
    logger.debug(f"    📝 Built context with {len(current_parent_chain)} parent levels")
    
    consecutive_failures = 0
    successful_expansions = 0
    
    # 🔥 路径追踪：记录当前层级的成功路径
    current_level_paths = []
    
    logger.info(f"    🤖 Starting neighbor evaluation (max consecutive failures: {max_consecutive_failures})")
    
    # 依次判断邻居节点
    for idx, neighbor_info in enumerate(neighbors, 1):
        neighbor_name = neighbor_info['table_name']
        minhash_sim = neighbor_info['minhash_similarity']
        edge_data = neighbor_info.get('edge_data', {})
        
        logger.info(f"      🔍 [{idx}/{len(neighbors)}] Evaluating: {neighbor_name}")
        logger.info(f"        📊 MinHash similarity: {minhash_sim:.3f}")
        
        # 显示边的其他信息
        if edge_data:
            desc_sim = edge_data.get('desc_sim', 0.0)
            reason = edge_data.get('reason', 'Unknown')
            logger.debug(f"        🔗 Edge info - Desc sim: {desc_sim:.3f}, Reason: {reason}")
        
        # 构建邻居表的schema信息
        # 需要传递example_root参数，这个参数需要从上层函数传递下来
        neighbor_schema = get_complete_neighbor_schema(neighbor_name, table_descriptions, use_description, 
                                                     example_root if 'example_root' in locals() else None, 
                                                     current_example_id if 'current_example_id' in locals() else None)
        has_neighbor_desc = use_description and neighbor_name in table_descriptions and table_descriptions[neighbor_name]
        if not neighbor_schema:
            # 如果无法获取完整schema，回退到简化版本
            neighbor_schema = f"Table full name: {neighbor_name}"
            if has_neighbor_desc:
                neighbor_desc = table_descriptions[neighbor_name]
                neighbor_schema = f"{neighbor_schema}\nTable Description: {neighbor_desc}"
                logger.debug(f"        📝 Enhanced neighbor with description")
            else:
                logger.debug(f"        📄 Using neighbor schema only")
        else:
            logger.debug(f"        📋 Using complete neighbor schema with columns")
        
        # 使用父节点信息进行rerank判断
        if expansion_search_count is not None:
            expansion_search_count[0] += 1  # 增加扩展搜索计数
        logger.info(f"        🤖 Running rerank with parent context... (expansion search #{expansion_search_count[0] if expansion_search_count else 'N/A'})")
        is_relevant = rerank_single_table(
            task, neighbor_schema, rerank_components,
            parent_info=parent_desc,
            instruction_suffix="consider how this table complements the parent table in addressing the task"
        )
        
        if is_relevant:
            logger.info(f"        ✅ EXPANDED: {neighbor_name} → relevant via {parent_table_name}")
            expanded_table = {
                "think": "",
                "answer": "Y",
                "columns": [],
                "table name": neighbor_name,
                "score": 0.9,
                "similarity": 0.0,  # 未直接计算语义相似度
                "expansion_level": current_level,
                "parent_table": parent_table_name,
                "parent_chain": current_parent_chain,  # 保存完整的父节点链
                "minhash_similarity": minhash_sim,
                "has_description": has_neighbor_desc
            }
            expanded_tables.append(expanded_table)
            consecutive_failures = 0
            successful_expansions += 1
            
            # 🔥 路径追踪：记录成功的单步路径
            if successful_paths is not None:
                step_path = {
                    'from': parent_table_name,
                    'to': neighbor_name,
                    'level': current_level,
                    'minhash_sim': minhash_sim,
                    'edge_data': edge_data,
                    'parent_chain': [p['table_name'] for p in current_parent_chain]
                }
                current_level_paths.append(step_path)
            
            logger.info(f"        🎉 Total successful expansions so far: {successful_expansions}")
            
            # 递归扩展：如果当前层级未达到最大限制，继续扩展
            if current_level < max_level:
                logger.info(f"        🔄 Recursive expansion to level {current_level + 1}")
                
                # 构建邻居的增强信息
                neighbor_enhanced_info = {
                    'table_name': neighbor_name,
                    'enhanced_schema': neighbor_schema,
                    'level': current_level
                }
                
                # 递归扩展
                sub_expanded = expand_via_minhash(
                    neighbor_name, neighbor_enhanced_info, task, rerank_components,
                    database_graphs, table_descriptions, processed_tables,
                    use_description, max_consecutive_failures, all_embeddings,
                    current_table_set, expansion_search_count,
                    current_level + 1, max_level, current_parent_chain,
                    global_candidate_cache, duplicate_candidates_avoided,
                    successful_paths, example_root, current_example_id
                )
                
                # 合并递归扩展的结果
                expanded_tables.extend(sub_expanded)
                logger.info(f"        🌳 Recursive expansion added {len(sub_expanded)} more tables")
        else:
            consecutive_failures += 1
            logger.info(f"        ❌ REJECTED: {neighbor_name} → not relevant")
            logger.info(f"        🔄 Consecutive failures: {consecutive_failures}/{max_consecutive_failures}")
            
            if consecutive_failures >= max_consecutive_failures:
                remaining = len(neighbors) - idx
                logger.info(f"        🛑 EARLY STOPPING after {consecutive_failures} consecutive failures")
                logger.info(f"        ⏭️  Skipping {remaining} remaining candidates")
                break
    
    # 🔥 路径追踪：将当前层级的成功路径记录到根父表下
    if successful_paths is not None and current_level_paths:
        # 构建完整路径：从根到当前层级的完整链
        for step in current_level_paths:
            # 构建从根父表到当前节点的完整路径
            full_path = []
            
            # 添加父节点链中的所有步骤
            for i, parent_step in enumerate(current_parent_chain[1:], 1):
                if i < len(current_parent_chain):
                    prev_parent = current_parent_chain[i-1]['table_name']
                    curr_parent = parent_step['table_name']
                    full_path.append({
                        'from': prev_parent,
                        'to': curr_parent,
                        'level': i,
                        'step_type': 'parent_chain'
                    })
            
            # 添加当前步骤
            full_path.append(step)
            
            # 记录到根父表下
            successful_paths[root_parent].append(full_path)
    
    logger.info(f"    🎯 MinHash expansion completed for {parent_table_name}:")
    logger.info(f"      ✅ Successful expansions: {successful_expansions}")
    logger.info(f"      📊 Success rate: {successful_expansions}/{min(idx, len(neighbors))} ({(successful_expansions/min(idx, len(neighbors))*100):.1f}%)")
    
    return expanded_tables


def save_expansion_paths_to_file(example_id, task, successful_paths, output_dir):
    """
    将成功的扩展路径保存到文件中，以任务为基本单位
    
    Args:
        example_id: str - 样本ID
        task: str - 任务描述
        successful_paths: Dict[str, List[List[Dict]]] - 成功路径数据
        output_dir: str - 输出目录
    """
    if not successful_paths:
        logger.debug(f"No successful paths to save for {example_id}")
        return
    
    try:
        # 确保输出目录存在
        os.makedirs(output_dir, exist_ok=True)
        
        # 构建输出文件路径
        output_file = os.path.join(output_dir, f"{example_id}_expansion_paths.json")
        
        # 准备要保存的数据
        task_data = {
            'example_id': example_id,
            'task': task,
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'expansion_paths': {},
            'statistics': {
                'total_root_parents': len(successful_paths),
                'total_paths': sum(len(paths) for paths in successful_paths.values()),
                'root_parents': list(successful_paths.keys())
            }
        }
        
        # 转换路径数据为可序列化格式
        for root_parent, paths in successful_paths.items():
            parent_data = {
                'root_table': root_parent,
                'total_paths': len(paths),
                'paths': []
            }
            
            for path_idx, path in enumerate(paths, 1):
                path_data = {
                    'path_id': path_idx,
                    'length': len(path),
                    'steps': []
                }
                
                # 构建路径摘要
                if path:
                    path_tables = [path[0]['from']] + [step['to'] for step in path]
                    path_data['path_summary'] = ' → '.join(path_tables)
                    
                    # 计算平均MinHash相似度
                    minhash_sims = [step['minhash_sim'] for step in path if 'minhash_sim' in step]
                    if minhash_sims:
                        path_data['avg_minhash_similarity'] = sum(minhash_sims) / len(minhash_sims)
                # 添加每个步骤的详细信息
                for step_idx, step in enumerate(path, 1):
                    step_data = {
                        'step': step_idx,
                        'from': step['from'],
                        'to': step['to'],
                        'level': step['level'],
                        'minhash_similarity': step.get('minhash_sim', 0.0),
                        'step_type': step.get('step_type', 'expansion')
                    }
                    
                    # 添加边的额外信息
                    if 'edge_data' in step and step['edge_data']:
                        edge_info = step['edge_data']
                        step_data['edge_info'] = {
                            'desc_sim': edge_info.get('desc_sim', 0.0),
                            'reason': edge_info.get('reason', 'Unknown')
                        }
                    
                    path_data['steps'].append(step_data)
                
                parent_data['paths'].append(path_data)
            
            task_data['expansion_paths'][root_parent] = parent_data
        
        # 保存到文件
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(task_data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"✅ Expansion paths saved for {example_id}: {output_file}")
        logger.info(f"    📊 {task_data['statistics']['total_root_parents']} root parents, {task_data['statistics']['total_paths']} total paths")
        
    except Exception as e:
        logger.error(f"❌ Error saving expansion paths for {example_id}: {e}")


def analyze_expansion_paths_summary(output_dir):
    """
    分析输出目录中所有的扩展路径文件，生成统计摘要
    
    Args:
        output_dir: str - 包含扩展路径文件的目录
    """
    try:
        if not os.path.exists(output_dir):
            logger.warning(f"Output directory does not exist: {output_dir}")
            return
        
        path_files = [f for f in os.listdir(output_dir) if f.endswith('_expansion_paths.json')]
        
        if not path_files:
            logger.info("No expansion path files found for analysis")
            return
        
        logger.info(f"📊 Analyzing {len(path_files)} expansion path files...")
        
        total_stats = {
            'total_examples': len(path_files),
            'total_root_parents': 0,
            'total_paths': 0,
            'examples': []
        }
        
        for path_file in path_files:
            file_path = os.path.join(output_dir, path_file)
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                example_stats = {
                    'example_id': data['example_id'],
                    'root_parents': data['statistics']['total_root_parents'],
                    'total_paths': data['statistics']['total_paths'],
                    'task_preview': data['task'][:100] + '...' if len(data['task']) > 100 else data['task']
                }
                
                total_stats['total_root_parents'] += example_stats['root_parents']
                total_stats['total_paths'] += example_stats['total_paths']
                total_stats['examples'].append(example_stats)
                
            except Exception as e:
                logger.error(f"Error reading {path_file}: {e}")
        
        # 生成摘要报告
        logger.info("📊 === EXPANSION PATHS SUMMARY ===")
        logger.info(f"📈 Total examples analyzed: {total_stats['total_examples']}")
        logger.info(f"📈 Total root parents: {total_stats['total_root_parents']}")
        logger.info(f"📈 Total expansion paths: {total_stats['total_paths']}")
        
        if total_stats['total_examples'] > 0:
            avg_parents = total_stats['total_root_parents'] / total_stats['total_examples']
            avg_paths = total_stats['total_paths'] / total_stats['total_examples']
            logger.info(f"📊 Average root parents per example: {avg_parents:.1f}")
            logger.info(f"📊 Average paths per example: {avg_paths:.1f}")
        
        # 显示前几个例子的详情
        logger.info("\n🔍 Sample Examples:")
        for i, example in enumerate(total_stats['examples'][:5], 1):
            logger.info(f"  {i}. {example['example_id']}: {example['root_parents']} parents, {example['total_paths']} paths")
            logger.info(f"     Task: {example['task_preview']}")
        
        if len(total_stats['examples']) > 5:
            logger.info(f"  ... and {len(total_stats['examples']) - 5} more examples")
        
        # 保存摘要到文件
        summary_file = os.path.join(output_dir, "expansion_paths_summary.json")
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(total_stats, f, indent=2, ensure_ascii=False)
        
        logger.info(f"📄 Summary saved to: {summary_file}")
        logger.info("📊 === END SUMMARY ===")
        
    except Exception as e:
        logger.error(f"❌ Error analyzing expansion paths: {e}")


def build_multi_parent_context(parent_chain, table_descriptions, use_description=True):
    """
    构建包含所有父节点信息的上下文
    
    Args:
        parent_chain: 父节点链列表
        table_descriptions: 表描述字典
        use_description: 是否使用表描述
    
    Returns:
        str: 格式化的多父节点上下文
    """
    if not parent_chain:
        return ""
    
    context_parts = []
    
    for i, parent in enumerate(parent_chain):
        table_name = parent['table_name']
        level = parent['level']
        
        # 获取父表的描述信息
        if use_description and table_name in table_descriptions:
            desc_raw = table_descriptions[table_name]
            desc = desc_raw.get("description", str(desc_raw)) if isinstance(desc_raw, dict) else str(desc_raw)
            parent_info = f"Level {level} Parent Table: {table_name}\nDescription: {desc}"
        else:
            # 使用schema信息
            schema_info = parent['info'].get('enhanced_schema', '')
            if schema_info:
                parent_info = f"Level {level} Parent Table: {table_name}\nSchema: {schema_info[:300]}..."
            else:
                parent_info = f"Level {level} Parent Table: {table_name}"
        
        context_parts.append(parent_info)
    
    # 构建层次化的上下文
    if context_parts:
        context = "=== PARENT TABLE CHAIN ===\n" + "\n\n".join(context_parts) + "\n\n=== CURRENT CANDIDATE ===\n"
    else:
        context = ""
    
    return context


def test_search_space_isolation(example_root: str, database_graphs_dir: str = "database_graphs"):
    """
    测试搜索空间独立性
    验证不同样本的schema linking不会互相干扰
    """
    logger.info("🧪 Testing search space isolation...")
    
    # 获取可用的样本
    available_samples = [d for d in os.listdir(example_root) 
                        if os.path.isdir(os.path.join(example_root, d))]
    
    if len(available_samples) < 2:
        logger.warning("⚠️  Need at least 2 samples to test isolation")
        return
    
    logger.info(f"📊 Testing with {len(available_samples)} samples: {available_samples[:5]}")
    
    # 加载数据库图
    database_graphs = {}
    if os.path.exists(database_graphs_dir):
        for graph_file in os.listdir(database_graphs_dir):
            if graph_file.endswith('_schema_graph.gpickle'):
                graph_path = os.path.join(database_graphs_dir, graph_file)
                database_name = graph_file.replace('_schema_graph.gpickle', '')
                try:
                    graph = load_graph(graph_path)
                    database_graphs[database_name] = graph
                except Exception as e:
                    logger.error(f"Error loading graph {graph_path}: {e}")
    
    logger.info(f"📊 Loaded {len(database_graphs)} database graphs")
    
    # 测试每个样本的表范围
    isolation_results = {}
    
    for sample_id in available_samples[:3]:  # 测试前3个样本
        logger.info(f"🔍 Testing sample: {sample_id}")
        
        # 获取该样本的表信息
        try:
            sample_tables = get_table_info_from_directory(example_root, sample_id)
            if not sample_tables:
                logger.warning(f"⚠️  No tables found for sample {sample_id}")
                continue
            
            # 提取表名
            table_names = [extract_table_name(tb) for tb in sample_tables]
            logger.info(f"  📋 Sample {sample_id} has {len(table_names)} tables")
            
            # 测试搜索空间限制
            current_table_set = set(table_names)
            
            # 模拟图扩展，检查是否会超出范围
            expansion_tests = []
            for db_name, graph in database_graphs.items():
                for table_name in table_names:
                    if table_name in graph.nodes():
                        neighbors = list(graph.neighbors(table_name))
                        in_scope_neighbors = [n for n in neighbors if n in current_table_set]
                        out_of_scope_neighbors = [n for n in neighbors if n not in current_table_set]
                        
                        expansion_tests.append({
                            'table': table_name,
                            'database': db_name,
                            'total_neighbors': len(neighbors),
                            'in_scope': len(in_scope_neighbors),
                            'out_of_scope': len(out_of_scope_neighbors)
                        })
            
            isolation_results[sample_id] = {
                'table_count': len(table_names),
                'expansion_tests': expansion_tests
            }
            
            logger.info(f"  ✅ Sample {sample_id}: {len(expansion_tests)} expansion tests completed")
            
        except Exception as e:
            logger.error(f"❌ Error testing sample {sample_id}: {e}")
    
    # 输出测试结果
    logger.info("🎯 Search Space Isolation Test Results:")
    for sample_id, results in isolation_results.items():
        expansion_tests = results['expansion_tests']
        if expansion_tests:
            total_neighbors = sum(test['total_neighbors'] for test in expansion_tests)
            total_in_scope = sum(test['in_scope'] for test in expansion_tests)
            total_out_of_scope = sum(test['out_of_scope'] for test in expansion_tests)
            
            logger.info(f"  📊 {sample_id}:")
            logger.info(f"    🔢 Tables: {results['table_count']}")
            logger.info(f"    🔗 Total graph neighbors: {total_neighbors}")
            logger.info(f"    ✅ In-scope neighbors: {total_in_scope}")
            logger.info(f"    🚫 Out-of-scope neighbors: {total_out_of_scope}")
            
            if total_out_of_scope > 0:
                isolation_ratio = total_in_scope / (total_in_scope + total_out_of_scope) * 100
                logger.info(f"    🔒 Isolation ratio: {isolation_ratio:.1f}% (higher is better)")
            else:
                logger.info(f"    🔒 Perfect isolation: No cross-sample neighbors")
    
    logger.info("✅ Search space isolation test completed")
    return isolation_results


def ask_model_sl_(tbs, task, rerank_components, score_threshold=0.5, external="", table_descriptions=None, use_description=True):
    """
    原始的Schema Linking算法，保持向后兼容
    """
    linked = []
    instruction = "Given a database task, judge whether the table schema is relevant to the task."
    scores = []
    
    if table_descriptions is None:
        table_descriptions = {}
    
    # First calculate all scores
    for tb in tbs:
        # Extract table name to get description
        try:
            table_name = re.search(r'^Table full name:\s*(.+)$', tb, re.MULTILINE).group(1)
        except Exception:
            table_name = "unknown"
        
        # Enhance table info with description if available and enabled
        enhanced_tb = tb
        if use_description and table_name in table_descriptions and table_descriptions[table_name]:
            desc_raw = table_descriptions[table_name]
            desc = desc_raw.get("description", str(desc_raw)) if isinstance(desc_raw, dict) else str(desc_raw)
            enhanced_tb = f"{tb}\n\nTable Description: {desc}"
            print(f"[DEBUG] Enhanced table {table_name} with description: {desc[:100]}...")
        elif not use_description:
            print(f"[DEBUG] Description disabled for table {table_name}")
        
        pairs = [(task, enhanced_tb)]
        inputs = process_inputs(
            pairs, instruction,
            rerank_components['max_length'],
            rerank_components['suffix_tokens'],
            rerank_components['tokenizer']
        )
        score = compute_logits(
            rerank_components['model'],
            inputs,
            rerank_components['sampling_params'],
            rerank_components['true_token'],
            rerank_components['false_token']
        )[0]
        scores.append(score)
    
    avg_score = sum(scores) / len(scores) if scores else 0.0

    # Then iterate through each table to make decisions
    for tb, score in zip(tbs, scores):
        answer = "Y" if (score > score_threshold or score > avg_score) else "N"
        try:
            table_name = re.search(r'^Table full name:\s*(.+)$', tb, re.MULTILINE).group(1)
        except Exception:
            table_name = "unknown"
        data = {
            "think": "",
            "answer": answer,
            "columns": [],
            "table name": table_name,
            "score": float(score)
        }
        print(f"[DEBUG] Table: {table_name} | Score: {score:.4f} | Threshold: {score_threshold} | Avg: {avg_score:.4f} | Answer: {answer}")
        linked.append(data)
    return linked

def compute_metrics_sl(file_pth, db_path):
    with open(file_pth) as f:
        data = json.load(f)
    count = 0
    precision_all = []
    recall_all = []

    # column-level metrics (only computed when gold has 'gold_columns')
    col_precision_all = []
    col_recall_all    = []
    has_col_gold      = False  # set True once we confirm gold has column info

    # 新增统计变量
    total_judged_tables = 0  # 总共判断了多少表
    total_selected_tables = 0  # 总共选择了多少表
    total_gold_tables = 0  # 总共gold表数
    task_stats = []  # 每个任务的统计

    # build a fast lookup dict from gold list
    gold_dict = {ex['instance_id']: ex for ex in gold}

    for example, tbs in data.items():
        ex = gold_dict.get(example)
        if ex is None:
            continue
        gold_table = set(ex["gold_tables"])

        # gold columns: stored as "table.column" or bare "column"
        gold_cols_raw = ex.get("gold_columns", [])
        if gold_cols_raw:
            has_col_gold = True
        # normalize: lower-case, strip quotes
        gold_cols = {c.lower().strip().replace('"','').replace('`','') for c in gold_cols_raw}

        if True or os.path.getsize(os.path.join(db_path, example, "prompts.txt")) > THRESHOLD:
            count += 1
            pred = []

            # 统计当前任务的表数量
            judged_count  = len(tbs)
            selected_count = 0
            gold_count    = len(gold_table)

            # predicted columns: collect from 'columns' field of selected tables
            pred_cols = set()

            for tb in tbs:
                if "answer" in tb:
                    if tb["answer"] == "Y":
                        pred.append(tb["table name"])
                        selected_count += 1
                        # each selected table entry may carry a 'columns' list
                        tbl_short = tb["table name"].split(".")[-1].lower()
                        for col in tb.get("columns", []):
                            col_norm = col.lower().strip().replace('"','').replace('`','')
                            # store as "table.column" to mirror gold format
                            pred_cols.add(f"{tbl_short}.{col_norm}")
                            pred_cols.add(col_norm)   # also bare name for loose match
                else:
                    print(tb)
                    pred.append(tb)
                    selected_count += 1

            # ── table-level metrics ──
            total_judged_tables  += judged_count
            total_selected_tables += selected_count
            total_gold_tables    += gold_count

            task_stats.append({
                'task_id': example,
                'judged_tables': judged_count,
                'selected_tables': selected_count,
                'gold_tables': gold_count
            })

            precision, recall = compute_precision_recall(clear_name(pred), clear_name(gold_table))
            print(f"Res: {precision}, {recall}, {example}")

            if recall < 1:
                print(f"Failed: {precision}, {recall}, {example}")
            precision_all.append(precision)
            recall_all.append(recall)

            # ── column-level metrics (when gold_columns present) ──
            if gold_cols:
                # try "table.column" match first; fall back to bare column name
                tp_cols = len(pred_cols & gold_cols)
                col_p = tp_cols / len(pred_cols) if pred_cols else 0.0
                col_r = tp_cols / len(gold_cols) if gold_cols else 0.0
                col_precision_all.append(col_p)
                col_recall_all.append(col_r)

    # 计算平均统计
    avg_judged   = total_judged_tables  / count if count > 0 else 0
    avg_selected = total_selected_tables / count if count > 0 else 0
    avg_gold     = total_gold_tables    / count if count > 0 else 0
    print(f"Count: {count}, mean recall: {np.mean(recall_all)}, mean precision: {np.mean(precision_all)}, num recall < 1: {np.sum(np.array(recall_all) < 1)}")
    print(f"\n=== 表数量统计 ===")
    print(f"总任务数: {count}")
    print(f"平均每个任务判断表数: {avg_judged:.2f}")
    print(f"平均每个任务选择表数: {avg_selected:.2f}")
    print(f"平均每个任务gold表数: {avg_gold:.2f}")
    print(f"选择表数/Gold表数比例: {avg_selected/avg_gold:.2f}")
    
    # 统计分布
    judged_dist = {'1-10': 0, '11-50': 0, '51-100': 0, '100+': 0}
    selected_dist = {'1-5': 0, '6-10': 0, '11-20': 0, '20+': 0}
    gold_dist = {'1-3': 0, '4-10': 0, '11-20': 0, '20+': 0}
    
    for stat in task_stats:
        # 判断表数分布
        if stat['judged_tables'] <= 10:
            judged_dist['1-10'] += 1
        elif stat['judged_tables'] <= 50:
            judged_dist['11-50'] += 1
        elif stat['judged_tables'] <= 100:
            judged_dist['51-100'] += 1
        else:
            judged_dist['100+'] += 1
            
        # 选择表数分布
        if stat['selected_tables'] <= 5:
            selected_dist['1-5'] += 1
        elif stat['selected_tables'] <= 10:
            selected_dist['6-10'] += 1
        elif stat['selected_tables'] <= 20:
            selected_dist['11-20'] += 1
        else:
            selected_dist['20+'] += 1
            
        # Gold表数分布
        if stat['gold_tables'] <= 3:
            gold_dist['1-3'] += 1
        elif stat['gold_tables'] <= 10:
            gold_dist['4-10'] += 1
        elif stat['gold_tables'] <= 20:
            gold_dist['11-20'] += 1
        else:
            gold_dist['20+'] += 1
    
    print(f"\n判断表数分布:")
    for range_name, count in judged_dist.items():
        print(f"  {range_name}表: {count}个任务")
    
    print(f"\n选择表数分布:")
    for range_name, count in selected_dist.items():
        print(f"  {range_name}表: {count}个任务")
    
    print(f"\nGold表数分布:")
    for range_name, count in gold_dist.items():
        print(f"  {range_name}表: {count}个任务")

    print(f"\n=== 性能指标 ===")
    tbl_f1 = [2*p*r/(p+r) if (p+r) > 0 else 0
              for p, r in zip(precision_all, recall_all)]
    print(f"  Table-level  Mean Precision : {np.mean(precision_all):.4f}")
    print(f"  Table-level  Mean Recall    : {np.mean(recall_all):.4f}")
    print(f"  Table-level  Mean F1        : {np.mean(tbl_f1):.4f}")
    print(f"  Num recall < 1             : {np.sum(np.array(recall_all) < 1)}")

    # Column-level metrics — only shown when gold file contains 'gold_columns'
    if has_col_gold and col_recall_all:
        print(f"\n=== Column-level 性能指标 (gold_columns 存在的样本: {len(col_recall_all)}) ===")
        print(f"  Column-level Mean Precision : {np.mean(col_precision_all):.4f}")
        print(f"  Column-level Mean Recall    : {np.mean(col_recall_all):.4f}")
        col_f1 = [2*p*r/(p+r) if (p+r) > 0 else 0
                  for p, r in zip(col_precision_all, col_recall_all)]
        print(f"  Column-level Mean F1        : {np.mean(col_f1):.4f}")
        print(f"  Num col-recall < 1          : {np.sum(np.array(col_recall_all) < 1)}")
    elif has_col_gold:
        print("\n[Note] gold_columns found but no predicted columns in output (add 'columns' field to predictions)")

import sqlite3
import pandas as pd
from datasketch import MinHash
from collections import Counter
import re
import networkx as nx

def extract_profiling(db_path, table_name, sample_size=10000):
    """
    抽取指定表的字段统计信息（Profiling）
    :param db_path: SQLite数据库路径
    :param table_name: 表名
    :param sample_size: 抽样记录数（大表用抽样）
    :return: dict，按字段组织的统计信息
    """
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute(f"PRAGMA table_info({table_name})")
    columns = [row[1] for row in cursor.fetchall()]
    profiling = {}
    for col in columns:
        query = f"SELECT {col} FROM {table_name} WHERE {col} IS NOT NULL LIMIT {sample_size}"
        try:
            df = pd.read_sql(query, conn)
        except Exception as e:
            print(f"[Profiling] Error reading {table_name}.{col}: {e}")
            continue
        values = df[col].astype(str).tolist()
        total = len(values)
        unique = len(set(values))
        null_ratio = 1 - (total / max(1, sample_size))
        try:
            numeric_vals = [float(v) for v in values if v.replace('.', '', 1).isdigit()]
            min_val = min(numeric_vals) if numeric_vals else None
            max_val = max(numeric_vals) if numeric_vals else None
        except:
            min_val = max_val = None
        top_values = Counter(values).most_common(10)
        lengths = [len(v) for v in values]
        avg_length = sum(lengths) / len(lengths) if lengths else 0
        patterns = {
            "14_digit": r"^\d{14}$",
            "date": r"\d{4}-\d{2}-\d{2}",
            "json": r"^\{.*\}$"
        }
        format_type = next((k for k, pattern in patterns.items() if any(re.match(pattern, v) for v in values)), "text")
        
        # 修改：MinHash基于列名而不是列值
        minhash = MinHash(num_perm=128)
        minhash.update(col.encode('utf8'))  # 只使用列名
        
        profiling[col] = {
            "null_ratio": round(null_ratio, 3),
            "unique_count": unique,
            "min_value": min_val,
            "max_value": max_val,
            "avg_length": round(avg_length, 1),
            "format": format_type,
            "top_values": [v[0] for v in top_values],
            "minhash": list(minhash.digest())
        }
    conn.close()
    return profiling

def build_schema_graph(db_path, minhash_threshold=0.8):
    """
    构建schema graph，节点为表，边基于字段minhash相似度
    :param db_path: SQLite数据库路径
    :param minhash_threshold: 建边的minhash相似度阈值
    :return: networkx.Graph 对象
    """
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
    tables = [row[0] for row in cursor.fetchall()]
    table_profiles = {}
    for table in tables:
        print(f"Profiling table: {table}")
        table_profiles[table] = extract_profiling(db_path, table)
    G = nx.Graph()
    for table, profile in table_profiles.items():
        G.add_node(table, profiling=profile)
    for i, t1 in enumerate(tables):
        for t2 in tables[i+1:]:
            cols1 = table_profiles[t1]
            cols2 = table_profiles[t2]
            for c1, p1 in cols1.items():
                for c2, p2 in cols2.items():
                    try:
                        m1 = MinHash(num_perm=128)
                        m1.digest(p1["minhash"])
                        m2 = MinHash(num_perm=128)
                        m2.digest(p2["minhash"])
                        sim = m1.jaccard(m2)
                    except Exception as e:
                        continue
                    if sim >= minhash_threshold:
                        G.add_edge(t1, t2, reason=f"minhash({c1},{c2})={sim:.2f}")
    return G

def reduce_columns(sql: str, subset_columns: set[str]) -> str:
    table_match = re.search(r'create\s+(?:or\s+replace\s+)?table\s+`?([^\s(]+)`?', sql, re.IGNORECASE)
    assert table_match, sql
    table_name = table_match.group(1)
    columns_block_match = re.search(r'\((.*?)\)\s*(PARTITION|CLUSTER|OPTIONS|;|$)', sql, re.DOTALL | re.IGNORECASE)
    if not columns_block_match:
        raise ValueError("Cannot extract columns block.")
    columns_block = columns_block_match.group(1)
    lines = columns_block.splitlines()
    filtered_lines = []
    for line in lines:
        line = line.strip().rstrip(',')
        if not line:
            continue
        parts = line.split()
        if len(parts) < 2:
            continue
        col_name = parts[0].strip('`",')
        if col_name in subset_columns:
            filtered_lines.append(f"  {line},")
    if filtered_lines:
        filtered_lines[-1] = filtered_lines[-1].rstrip(',')
    new_sql = f'CREATE TABLE {table_name} (\n' + '\n'.join(filtered_lines) + '\n);'
    return new_sql

def reduce_ddl(example_path, dictionaries, linked_json, reduce_col=False):
    print("Doing schema linking")
    for eg_id in tqdm(dictionaries):
        api = get_api_name(eg_id)
        ddl_paths = search_file(os.path.join(example_path, eg_id), "DDL.csv")
        if os.path.getsize(os.path.join(example_path, eg_id, "prompts.txt")) < THRESHOLD or eg_id in DEPS_DEV_V1:
            continue
        with open(linked_json) as f:
            sl = json.load(f)
        table_names = []
        columns = {}
        for ex_id, tbs in sl.items():
            if ex_id == eg_id:
                for tb in tbs:
                    if "answer" in tb:
                        if tb["answer"] == "Y":
                            table_names.append(tb["table name"])
                            columns[tb["table name"]] = tb['columns']
                    else:
                        raise NotImplementedError
                        print(tb)
                        table_names.append(tb)
        if not table_names:
            print("Empty result in table_names", eg_id)
            continue
        print("Doing sl for", eg_id)
        table_names_no_digit = [remove_digits(i) for i in table_names]
        temp_file_paths = []
        for ddl_path in ddl_paths:
            temp_file = ddl_path.replace("DDL.csv", "DDL_sl.csv")
            temp_file_paths.append(temp_file)
            with open(ddl_path, "r", newline="", encoding="utf-8", errors="ignore") as infile, \
                open(temp_file, "w", newline="", encoding="utf-8", errors="ignore") as outfile:
                reader = csv.reader(infile)
                writer = csv.writer(outfile)
                header = next(reader)
                writer.writerow(header)
                row_count = 0
                row_count_rm = 0
                total_count = 0
                row_list_all = []
                row_list = []
                for row in reader:
                    # 调试：打印行内容以了解实际格式
                    if len(row) == 0:
                        continue  # 跳过空行
                    
                    # 找到包含 CREATE 语句的列
                    create_col_idx = None
                    for i, col in enumerate(row):
                        if col and col.upper().startswith("CREATE"):
                            create_col_idx = i
                            break
                    
                    if create_col_idx is None:
                        print(f"Warning: No CREATE statement found in row: {row}")
                        continue
                    
                    # 使用找到的 CREATE 语句列
                    ddl_statement = row[create_col_idx]
                    table_name = row[0].strip()  # 表名通常在第一列
                    if "." in table_name:
                        table_name = table_name.split(".")[-1]
                    json_pth = ddl_path.replace("DDL.csv", table_name+".json")
                    if os.path.exists(json_pth):
                        with open(json_pth) as f:
                            table_fullname = json.load(f)["table_fullname"]
                    else:
                        print(f"{ex_id}: {json_pth} doesn't exist")
                        continue
                    if any(remove_digits(table_fullname) in item for item in table_names_no_digit):
                        row_count_rm += 1
                        row_list_all.append(row)
                    if any(table_fullname == item for item in table_names):
                        row_count += 1
                        if reduce_col:
                            assert table_fullname in columns, print(table_names, table_fullname)
                            # 如果 columns 为空列表，表示使用所有列，不修改 DDL
                            if columns[table_fullname]:  # 只有当 columns 不为空时才调用 reduce_columns
                                row[create_col_idx] = reduce_columns(row[create_col_idx], columns[table_fullname])
                            # 如果 columns 为空，保持原始 DDL 不变
                        row_list.append(row)
                    total_count += 1
                print(f"{eg_id}: tables before linking: {total_count}, tables after linking: {row_count}, tables rm digits after linking: {row_count_rm}")
                if 0 < row_count < 10 or row_count_rm > 1000 or reduce_col:
                    writer.writerows(row_list)
                elif row_count_rm:
                    print("RM digits", len(row_list))
                    writer.writerows(row_list_all)
        if all(is_csv_empty(i) for i in temp_file_paths):
            print(f"{eg_id}: All empty DDL_sl.csv, remove, table_names", table_names)
            for i in temp_file_paths:
                os.remove(i)
    compress_ddl(example_path, add_description=True, add_sample_rows=True, rm_digits=True, schema_linked=True, clear_long_eg_des=True, reduce_col=reduce_col)

ask_prompt = """
You are doing table level schema linking. Given a table with schema information and the task, you should think step by step and decide whether this table is related to the task. 
You should answer Y/N only. If the answer is Y, you should add columns that you think is related in python list format.

Please answer only in json code block like:
```json
{{
    "think": "think step by step to decide",
    "answer": "Y or N only",
    "columns": [col_name1, col_name2]
}}
```

Table info: {0}
Task: {1}
{2}
"""

def get_table_info_from_directory(example_path, example_id):
    """
    直接从样本目录读取所有表的信息，参考 reconstruct_data.py 的处理逻辑
    :param example_path: examples_lite 路径
    :param example_id: 样本ID，如 bq001
    :return: 表信息列表
    """
    import pandas as pd
    
    tables_info = []
    sample_dir = os.path.join(example_path, example_id)
    
    if not os.path.exists(sample_dir):
        return tables_info
    
    # 遍历样本目录下的所有子目录 (项目名)
    for project_name in os.listdir(sample_dir):
        if project_name == "spider":
            continue
            
        project_path = os.path.join(sample_dir, project_name)
        if not os.path.isdir(project_path):
            continue
            
        # 遍历项目下的数据库目录
        for db_name in os.listdir(project_path):
            db_path = os.path.join(project_path, db_name)
            if not os.path.isdir(db_path):
                continue
                
            # 查找 DDL.csv 文件
            ddl_path = os.path.join(db_path, "DDL.csv")
            if not os.path.exists(ddl_path):
                continue
                
            try:
                # 使用 pandas 读取 DDL.csv，获取表名列表
                ddl_file = pd.read_csv(ddl_path)
                if 'table_name' not in ddl_file.columns:
                    continue
                    
                table_name_list = ddl_file['table_name'].tolist()
                
                # 处理每个表
                for table_name in table_name_list:
                    if pd.isna(table_name):
                        continue
                        
                    table_name = str(table_name).strip()
                    if not table_name:
                        continue
                    
                    # 尝试多种文件名匹配方式（参考 reconstruct_data.py 的逻辑）
                    json_files_to_try = [
                        os.path.join(db_path, f"{table_name}.json"),
                        os.path.join(db_path, f"{db_name}.{table_name}.json")
                    ]
                    
                    table_json = None
                    for json_file in json_files_to_try:
                        if os.path.exists(json_file):
                            try:
                                with open(json_file, 'r', encoding='utf-8') as f:
                                    table_json = json.load(f)
                                break
                            except Exception as e:
                                continue
                    
                    if table_json is None:
                        continue
                    
                    # 构建表信息字符串，严格按照 reconstruct_data.py 的格式
                    table_info = f"Table full name: {table_json.get('table_fullname', table_name)}\n"
                    
                    # 处理列信息
                    column_names = table_json.get('column_names', [])
                    column_types = table_json.get('column_types', [])
                    descriptions = table_json.get('description', [])
                    
                    for j in range(len(column_names)):
                        col_type = column_types[j] if j < len(column_types) else "UNKNOWN"
                        
                        # 添加描述信息（如果有）
                        description_text = ""
                        if j < len(descriptions) and descriptions[j]:
                            description_text = f" Description: {descriptions[j]}"
                        
                        table_info += f"Column name: {column_names[j]} Type: {col_type}{description_text}\n"
                    
                    # 添加样本行（如果有）
                    if 'sample_rows' in table_json and table_json['sample_rows']:
                        sample_rows = table_json['sample_rows']
                        table_info += f"Sample rows:\n{sample_rows}\n"
                    
                    # 处理相似表信息（参考 reconstruct_data.py 的 representatives 逻辑）
                    # 这部分暂时简化，如需要可以添加
                    
                    tables_info.append(table_info.strip())
                            
            except Exception as e:
                logger.info(f"Error reading DDL for {example_id}/{project_name}/{db_name}: {e}")
                continue
    
    return tables_info

def get_external_knowledge(example_path, example_id):
    """
    获取外部知识信息（文档等）
    """
    external_knowledge = ""
    sample_dir = os.path.join(example_path, example_id)
    
    # 查找 .md 文档文件
    for item in os.listdir(sample_dir):
        if item.endswith('.md'):
            try:
                with open(os.path.join(sample_dir, item), 'r', encoding='utf-8') as f:
                    content = f.read()
                    # 取前1000字符作为外部知识
                    external_knowledge += content[:1000] + "\n"
            except:
                continue
                
    return external_knowledge

def save_all_table_profiling(db_path, output_dir, sample_size=10000):
    """
    对db_path下所有表做profiling，并将结果存为json
    """
    os.makedirs(output_dir, exist_ok=True)
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
    tables = [row[0] for row in cursor.fetchall()]
    conn.close()
    total_tables = len(tables)
    print(f"[Profiling] Total tables to process: {total_tables}")
    for idx, table in enumerate(tables, 1):
        print(f"[Profiling] ({idx}/{total_tables}) Profiling Table: {table}")
        try:
            profiling = extract_profiling(db_path, table, sample_size=sample_size)
            out_path = os.path.join(output_dir, f"{table}_profiling.json")
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(profiling, f, indent=2, ensure_ascii=False)
            print(f"[Profiling] Saved to {out_path}")
        except Exception as e:
            print(f"[Profiling][Error] Table: {table} - {e}")

def extract_profiling_sample_rows(sample_rows: List[Dict], column_names: List[str], sample_size: int = 10000):
    """根据样本行（json list of dict）做profiling，逻辑与 csv/sqlite 保持一致"""
    from collections import defaultdict
    collected: Dict[str, List[str]] = defaultdict(list)
    for row in sample_rows[:sample_size]:
        for col in column_names:
            if col in row and row[col] is not None:
                collected[col].append(str(row[col]))
    profiling = {}
    for col, values in collected.items():
        total = len(values)
        unique = len(set(values))
        null_ratio = 1 - (total / max(1, sample_size))
        try:
            numeric_vals = [float(v) for v in values if v.replace('.', '', 1).isdigit()]
            min_val = min(numeric_vals) if numeric_vals else None
            max_val = max(numeric_vals) if numeric_vals else None
        except Exception:
            min_val = max_val = None
        top_values = Counter(values).most_common(10)
        lengths = [len(v) for v in values]
        avg_length = sum(lengths) / len(lengths) if lengths else 0
        patterns = {
            "14_digit": r"^\d{14}$",
            "date": r"\d{4}-\d{2}-\d{2}",
            "json": r"^\{.*\}$"
        }
        format_type = next((k for k, pattern in patterns.items() if any(re.match(pattern, v) for v in values)), "text")
        
        # 修改：MinHash基于列名而不是列值
        minhash = MinHash(num_perm=128)
        minhash.update(col.encode('utf8'))  # 只使用列名
        
        profiling[col] = {
            "null_ratio": round(null_ratio, 3),
            "unique_count": unique,
            "min_value": min_val,
            "max_value": max_val,
            "avg_length": round(avg_length, 1),
            "format": format_type,
            "top_values": [v[0] for v in top_values],
            "minhash": [int(x) for x in minhash.digest()]
        }
    return profiling


def save_all_table_profiling_from_json(example_path, output_dir, sample_size: int = 10000):
    """
    遍历样本目录结构，查找每个表的json文件，若包含sample_rows则做profiling，否则保存静态结构信息
    """
    os.makedirs(output_dir, exist_ok=True)
    total_tables = 0
    for example_id in os.listdir(example_path):
        sample_dir = os.path.join(example_path, example_id)
        if not os.path.isdir(sample_dir):
            continue
        for item in os.listdir(sample_dir):
            item_path = os.path.join(sample_dir, item)
            if not os.path.isdir(item_path) or item in ['spider', 'output', '__pycache__']:
                continue
            for schema_item in os.listdir(item_path):
                schema_path = os.path.join(item_path, schema_item)
                if not os.path.isdir(schema_path):
                    continue
                for file in os.listdir(schema_path):
                    if file.endswith('.json') and not file.startswith('.'):
                        json_path = os.path.join(schema_path, file)
                        try:
                            with open(json_path, 'r', encoding='utf-8') as jf:
                                table_json = json.load(jf)
                            if 'sample_rows' in table_json and table_json['sample_rows']:
                                profiling = extract_profiling_sample_rows(table_json['sample_rows'], table_json.get('column_names', []), sample_size)
                            else:
                                profiling = {
                                    "table_fullname": table_json.get("table_fullname"),
                                    "column_names": table_json.get("column_names", []),
                                    "column_types": table_json.get("column_types", [])
                                }
                            out_path = os.path.join(output_dir, f"{example_id}_{item}_{schema_item}_{file.replace('.json', '_profiling.json')}")
                            with open(out_path, "w", encoding="utf-8") as f:
                                json.dump(profiling, f, indent=2, ensure_ascii=False)
                            print(f"[Profiling] Saved to {out_path}")
                            total_tables += 1
                        except Exception as e:
                            print(f"[Profiling][Error] {json_path} - {e}")
    print(f"[Profiling] Total tables processed from json: {total_tables}")

# ===================== CSV Profiling =====================

def extract_profiling_csv(csv_path: str, sample_size: int = 10000):
    """
    对单个 csv 文件的每一列进行 profiling，与 extract_profiling 中 sqlite 逻辑保持一致。
    :param csv_path: csv 文件绝对路径
    :param sample_size: 采样行数（防止大文件过慢）
    :return: dict 结构 {column: profiling_info}
    """
    try:
        df = pd.read_csv(csv_path, nrows=sample_size)
    except Exception as e:
        # 尝试使用 \t 分隔符再次读取
        try:
            df = pd.read_csv(csv_path, nrows=sample_size, sep='\t')
        except Exception:
            raise e
    profiling = {}
    for col in df.columns:
        values = df[col].dropna().astype(str).tolist()
        total = len(values)
        unique = len(set(values))
        null_ratio = 1 - (total / max(1, sample_size))
        try:
            numeric_vals = [float(v) for v in values if v.replace('.', '', 1).isdigit()]
            min_val = min(numeric_vals) if numeric_vals else None
            max_val = max(numeric_vals) if numeric_vals else None
        except Exception:
            min_val = max_val = None
        top_values = Counter(values).most_common(10)
        lengths = [len(v) for v in values]
        avg_length = sum(lengths) / len(lengths) if lengths else 0
        patterns = {
            "14_digit": r"^\d{14}$",
            "date": r"\d{4}-\d{2}-\d{2}",
            "json": r"^\{.*\}$"
        }
        format_type = next((k for k, pattern in patterns.items() if any(re.match(pattern, v) for v in values)), "text")
        
        # 修改：MinHash基于列名而不是列值
        minhash = MinHash(num_perm=128)
        minhash.update(col.encode('utf8'))  # 只使用列名
        
        profiling[col] = {
            "null_ratio": round(null_ratio, 3),
            "unique_count": unique,
            "min_value": min_val,
            "max_value": max_val,
            "avg_length": round(avg_length, 1),
            "format": format_type,
            "top_values": [v[0] for v in top_values],
            "minhash": [int(x) for x in minhash.digest()]
        }
    return profiling


def save_all_table_profiling_from_csv(example_path: str, output_dir: str, sample_size: int = 10000):
    """
    遍历样本目录，查找 csv 数据文件，对每张表执行 extract_profiling_csv，并保存 json。
    规则假设：schema 目录下存在与 table json 同名或相似（忽略大小写）的 csv 文件；
    若未找到匹配 csv 则跳过。
    """
    os.makedirs(output_dir, exist_ok=True)
    total_tables = 0
    for example_id in os.listdir(example_path):
        sample_dir = os.path.join(example_path, example_id)
        if not os.path.isdir(sample_dir):
            continue
        for project_name in os.listdir(sample_dir):
            project_path = os.path.join(sample_dir, project_name)
            if not os.path.isdir(project_path) or project_name in ['spider', 'output', '__pycache__']:
                continue
            for schema_name in os.listdir(project_path):
                schema_path = os.path.join(project_path, schema_name)
                if not os.path.isdir(schema_path):
                    continue
                # 先构建一个 csv 文件列表供后续匹配
                csv_files = [f for f in os.listdir(schema_path) if f.lower().endswith('.csv')]
                if not csv_files:
                    continue
                for json_file in os.listdir(schema_path):
                    if not json_file.endswith('.json') or json_file.startswith('.'):
                        continue
                    json_path = os.path.join(schema_path, json_file)
                    try:
                        with open(json_path, 'r', encoding='utf-8') as jf:
                            table_meta = json.load(jf)
                    except Exception as e:
                        print(f"[Profiling][Error] Read json {json_path} - {e}")
                        continue
                    table_fullname = table_meta.get('table_fullname') or json_file.replace('.json', '')
                    # 根据 table_fullname / json 文件名 尝试匹配 csv
                    cand_csv_names = [
                        f"{table_fullname}.csv",
                        f"{json_file.replace('.json', '.csv')}",
                        f"{table_fullname.split('.')[-1]}.csv"
                    ]
                    csv_path = None
                    for cand in cand_csv_names:
                        if cand in csv_files:
                            csv_path = os.path.join(schema_path, cand)
                            break
                    # 若仍未匹配，尝试大小写忽略匹配
                    if csv_path is None:
                        lower_map = {f.lower(): f for f in csv_files}
                        for cand in cand_csv_names:
                            if cand.lower() in lower_map:
                                csv_path = os.path.join(schema_path, lower_map[cand.lower()])
                                break
                    if csv_path is None:
                        # 未找到对应 csv，跳过
                        continue
                    # 做 profiling
                    try:
                        profiling = extract_profiling_csv(csv_path, sample_size=sample_size)
                        out_fname = f"{example_id}_{project_name}_{schema_name}_{table_fullname.replace('.', '_')}_profiling.json"
                        out_path = os.path.join(output_dir, out_fname)
                        with open(out_path, 'w', encoding='utf-8') as f:
                            json.dump(profiling, f, indent=2, ensure_ascii=False)
                        print(f"[Profiling] Saved to {out_path}")
                        total_tables += 1
                    except Exception as e:
                        print(f"[Profiling][Error] csv {csv_path} - {e}")
    print(f"[Profiling] Total tables processed with csv: {total_tables}")

# ================ Description Generation & Graph Build ================
import itertools
from tqdm import tqdm
import time, traceback

# -------------------- GPTChat 基于 ask_model_sl 风格的描述生成 --------------------

DESC_PROMPT = (
    "You are a database expert. Given the following table schema, generate a concise English description (80-120 words) summarizing the core content and purpose of the table.\n"
    "Please output only a JSON code block in the following format: ```json\n{{\n  \"description\": \"...\"\n}}```\n"
    "\nTable schema:\n{schema}\n"
)

# DESC_PROMPT = (
#     "你是数据库专家，请根据给定表结构，生成一句 80-120 字的中文描述，总结该表的核心内容和用途。\n"
#     "请仅输出 JSON 代码块，格式如下：```json\n{{\n  \"description\": \"...\"\n}}```\n"
#     "\n表结构如下：\n{schema}\n"
# )



def ask_model_desc_(schema_str: str, chat_session: GPTChat, timeout: int = 60, max_length: int = 131072) -> str:
    """
    Mimic ask_model_sl_: 3 retries + total timeout.
    Return None for any exception and print traceback.
    """
    try:
        chat_session.init_messages()
        prompt = DESC_PROMPT.format(schema=schema_str)
        
        # Check input length limit
        if len(prompt) > max_length:
            logger.warning(f"Input length {len(prompt)} exceeds limit {max_length}, truncating")
            # Truncate schema_str part while keeping prompt template
            template_part = DESC_PROMPT.split("{schema}")[0] + DESC_PROMPT.split("{schema}")[1]
            available_length = max_length - len(template_part) - 100  # Leave some margin
            truncated_schema = schema_str[:available_length] + "...(truncated)"
            prompt = DESC_PROMPT.format(schema=truncated_schema)
            logger.info(f"Truncated input length: {len(prompt)}")
        
        deadline = time.time() + timeout
        logger.debug(f"Starting description generation, timeout: {timeout}s, prompt length: {len(prompt)} chars")

        for attempt in range(1, 4):          # Max 3 attempts
            try:
                logger.debug(f"Attempt {attempt} calling model...")
                start_time = time.time()
                block = None
                
                block = chat_session.get_model_response(prompt, "json")[0]
                
                # Check if response is None or empty
                if block is None or not block.strip():
                    logger.warning(f"Attempt {attempt} received empty response, possibly Bad Request")
                    raise ValueError("Received empty response or None")
                
                elapsed = time.time() - start_time
                logger.debug(f"Attempt {attempt} received response, elapsed: {elapsed:.2f}s")
                logger.debug(f"Raw response: {block[:200]}...")
                
                result = json.loads(block)["description"].strip()
                logger.debug(f"Attempt {attempt} success: {result[:100]}...")
                return result
                
            except Exception as e:
                elapsed = time.time() - start_time if 'start_time' in locals() else 0
                logger.warning(f"Attempt {attempt} failed (elapsed {elapsed:.2f}s): {e}")
                
                # If Bad Request related error, skip directly
                if "400" in str(e) or "Bad Request" in str(e) or (block is None):
                    logger.error(f"Detected Bad Request or empty response, skipping this table")
                    return None
                
                if time.time() > deadline:
                    logger.warning(f"Attempt {attempt} timeout, stopping retries")
                    break    # Exit loop on timeout
                    
                prompt = (
                    "Previous output cannot be parsed, please only output JSON code block like:"
                    "{\"description\":\"...\"}\n\n"
                    f"{schema_str}"
                )

        # All attempts failed
        logger.error("All attempts failed, returning None")
        return None
        
    except Exception as e:
        logger.error(f"ask_model_desc_ unexpected error: {e}")
        logger.error(traceback.format_exc())
        return None


def build_schema_str(table_json: Dict, tname_fallback: str) -> str:
    tname = table_json.get('table_fullname', tname_fallback)
    col_names = table_json.get('column_names', [])
    col_types = table_json.get('column_types', [])
    schema_info = f"Table full name: {tname}\n"
    for n, t in zip(col_names, col_types):
        schema_info += f"Column name: {n} Type: {t}\n"
    if table_json.get('sample_rows'):
        schema_info += f"Sample rows:\n{table_json['sample_rows']}\n"
    return schema_info



def build_schema_graph_from_profiling(profiling_dir: str, output_path: str,
                                       desc_sim_th: float = 0.7, minhash_th: float = 0.8,
                                       example_root: str = None, fk_relations: List[Dict] = None):
    """基于 profiling json 构建表级 schema graph，支持外键关系增强"""
    model = get_embedding_model()
    
    # 1. 加载表描述数据（从example_root读取，而不是profiling文件）
    table_descriptions = {}
    if example_root and os.path.exists(example_root):
        logger.info(f"Loading table descriptions from {example_root}")
        for example_id in os.listdir(example_root):
            example_dir = os.path.join(example_root, example_id)
            if not os.path.isdir(example_dir):
                                    continue

            desc_file = os.path.join(example_dir, "table_descriptions.json")
            if os.path.exists(desc_file):
                try:
                    with open(desc_file, 'r', encoding='utf-8') as f:
                        desc_data = json.load(f)
                    
                                        # 合并到全局字典中
                    for table_name, description in desc_data.items():
                        table_descriptions[table_name] = description
                        
                except Exception as e:
                    logger.warning(f"Error loading descriptions from {desc_file}: {e}")

        logger.info(f"Loaded descriptions for {len(table_descriptions)} tables")
    else:
        logger.warning("No example_root provided, building graph without table descriptions")

    # 检查外键关系
    if fk_relations:
        logger.info(f"Using {len(fk_relations)} foreign key relationships for graph construction")
    else:
        logger.info("No foreign key relationships provided")

    table_info = {}
    # 2. 收集节点信息（从profiling文件读取MinHash，从table_descriptions读取描述）
    for fp in os.listdir(profiling_dir):
        if not fp.endswith('.json'):
            continue
        full_path = os.path.join(profiling_dir, fp)
        with open(full_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        tname = data.get('table_fullname') or fp.replace('_profiling.json', '')
        
        # 从table_descriptions获取描述，而不是从profiling文件
        desc = table_descriptions.get(tname, '')
        if desc:
            # 构建完整的schema描述（类似于build_database_specific_graphs的做法）
            schema_str = f"Table full name: {tname}\n"
            col_names = data.get('column_names', [])
            col_types = data.get('column_types', [])
            for n, t in zip(col_names, col_types):
                schema_str += f"Column name: {n} Type: {t}\n"
            enhanced_description = f"{schema_str}\n\nTable Description: {desc}"
            emb = model.encode(desc, normalize_embeddings=True)
            #emb = model.encode(enhanced_description, normalize_embeddings=True)
        else:
            emb = None
        # 聚合列级 minhash
        mh_table = MinHash(num_perm=128)
        if any(isinstance(v, dict) and 'minhash' in v for v in (data.values() if isinstance(data, dict) else [])):
            # 若 data 直接是列 dict
            col_items = [v for v in data.values() if isinstance(v, dict) and 'minhash' in v]
        else:
            col_items = [v for v in data.get('columns', {}).values()] if isinstance(data, dict) else []
        
        # 修改：MinHash基于列名而不是从profiling文件读取
        col_names = data.get('column_names', [])
        for col_name in col_names:
            mh_table.update(col_name.encode('utf8'))
        
        # 注释掉原来的从profiling文件读取MinHash的逻辑
        # for col in col_items:
        #     col_mh = MinHash(num_perm=128)
        #     col_mh.digest([int(x) for x in col['minhash']])
        #     for i in range(128):
        #         mh_table.hashvalues[i] = min(mh_table.hashvalues[i], col_mh.hashvalues[i])
        table_info[tname] = {'emb': emb, 'mh': mh_table}
    print(f"[Graph] Total tables loaded: {len(table_info)}")
    G = nx.Graph()
    for t in table_info:
        G.add_node(t)
    names = list(table_info.keys())
    
    edges_added = 0
    fk_edges_added = 0
    
    for i in range(len(names)):
        for j in range(i+1, len(names)):
            t1, t2 = names[i], names[j]
            
            # 检查是否存在外键关系
            has_fk_relation = False
            if fk_relations:
                has_fk_relation = has_foreign_key_relationship(t1, t2, fk_relations)
            
            emb_sim = 0.0
            if table_info[t1]['emb'] is not None and table_info[t2]['emb'] is not None:
                emb_sim = float(QwenEmbeddingModel.cos_sim(table_info[t1]['emb'], table_info[t2]['emb']))
            mh_sim = table_info[t1]['mh'].jaccard(table_info[t2]['mh']) if table_info[t1]['mh'] and table_info[t2]['mh'] else 0.0
            
            # 判断是否添加边
            should_add_edge = False
            edge_reasons = []
            edge_weight = 1.0
            
            # 优先考虑外键关系
            if has_fk_relation:
                should_add_edge = True
                edge_weight = 1.0  # 外键关系权重为1
                edge_reasons.append("foreign_key=1.0")
                fk_edges_added += 1
                print(f"  🔗 FK edge: {t1} -> {t2} (weight: 1.0)")
            
            # 如果没有外键关系，使用原有的相似度判断
            elif emb_sim >= desc_sim_th:
                should_add_edge = True
                edge_weight = emb_sim
                edge_reasons.append(f"desc_sim={emb_sim:.3f}")
            
            elif mh_sim >= minhash_th:
                should_add_edge = True
                edge_weight = mh_sim
                edge_reasons.append(f"minhash={mh_sim:.3f}")
            
            if should_add_edge:
                G.add_edge(t1, t2, 
                          desc_sim=round(emb_sim,3), 
                          mh_sim=round(mh_sim,3),
                          fk_relation=has_fk_relation,
                          weight=edge_weight,
                          reason=", ".join(edge_reasons))
                edges_added += 1
    
    print(f"[Graph] Graph saved to {output_path} | Nodes: {G.number_of_nodes()} Edges: {G.number_of_edges()}")
    if fk_edges_added > 0:
        print(f"  📎 Foreign key edges: {fk_edges_added}")
    
    save_graph(G, output_path)

# ================= Table Description Generation Only =================


def get_table_info_from_sqlite(sqlite_path: str, example_id: str) -> List[str]:
    """
    从SQLite文件中提取表信息，转换为schema_str格式
    :param sqlite_path: SQLite文件路径
    :param example_id: 样本ID（用于表名前缀）
    :return: 表信息字符串列表
    """
    import sqlite3
    
    tables_info = []
    
    try:
        connection = sqlite3.connect(sqlite_path)
        cursor = connection.cursor()
        
        # 获取所有表
        cursor.execute("SELECT name, sql FROM sqlite_master WHERE type='table'")
        tables = cursor.fetchall()
        
        logger.info(f"Found {len(tables)} tables in SQLite file: {sqlite_path}")
        
        for table_name, create_sql in tables:
            if table_name.startswith('sqlite_'):  # 跳过系统表
                continue
                
            try:
                # 获取列信息
                cursor.execute(f"PRAGMA table_info({table_name})")
                columns_info = cursor.fetchall()
                
                column_names = []
                column_types = []
                for col in columns_info:
                    column_names.append(col[1])  # 列名
                    column_types.append(col[2])  # 列类型
                
                # 获取样本行
                sample_rows = []
                try:
                    cursor.execute(f"SELECT * FROM {table_name} LIMIT 3")
                    sample_rows = cursor.fetchall()
                except Exception as e:
                    logger.warning(f"Error fetching sample rows for {table_name}: {e}")
                
                # 构建表信息字符串
                table_info = f"Table full name: {table_name}\n"
                
                for i, col_name in enumerate(column_names):
                    col_type = column_types[i] if i < len(column_types) else "UNKNOWN"
                    table_info += f"Column name: {col_name} Type: {col_type}\n"
                
                # 添加样本行
                if sample_rows:
                    table_info += f"Sample rows:\n{str(sample_rows)}\n"
                
                tables_info.append(table_info.strip())
                logger.debug(f"Processed table {table_name} with {len(column_names)} columns")
                
            except Exception as e:
                logger.error(f"Error processing table {table_name}: {e}")
                continue
        
        connection.close()
        
    except Exception as e:
        logger.error(f"Error reading SQLite file {sqlite_path}: {e}")
    
    return tables_info


def generate_table_descriptions(example_root: str, model_name: str = "Qwen3-235B-A22B-Instruct-2507-FP8", desc_output: str = "data/table_descriptions.json"):
    """Traverse all table schema json files and sqlite files under example_root, generate descriptions for each table, save by example_id."""
    
    dictionaries = [d for d in os.listdir(example_root) if os.path.isdir(os.path.join(example_root, d))]
    
    logger.info(f"Starting table description generation, model: {model_name}")
    logger.info(f"Processing directory: {example_root}")
    logger.info(f"Total {len(dictionaries)} examples to process")
    
    chat_global = GPTChat(model=model_name, temperature=0)
    processed_count = 0
    failed_count = 0
    
    for example_id in tqdm(dictionaries, desc="Processing examples"):
        try:
            logger.info(f"Starting to process example: {example_id}")
            
            # 处理local样本（SQLite文件）
            if example_id.startswith("local"):
                sample_dir = os.path.join(example_root, example_id)
                sqlite_files = [f for f in os.listdir(sample_dir) if f.endswith('.sqlite')]
                
                if not sqlite_files:
                    logger.warning(f"Local example {example_id} has no SQLite files, skipping")
                    continue
                
                # 通常local样本只有一个sqlite文件
                sqlite_file = sqlite_files[0]
                sqlite_path = os.path.join(sample_dir, sqlite_file)
                
                logger.info(f"Processing local example {example_id} with SQLite file: {sqlite_file}")
                tbs = get_table_info_from_sqlite(sqlite_path, example_id)
                
            else:
                # 处理普通样本（JSON文件）
                tbs = get_table_info_from_directory(example_root, example_id)
            
            if not tbs:
                logger.warning(f"Example {example_id} no table info found, skipping")
                continue
                
            logger.info(f"Example {example_id} contains {len(tbs)} tables")
            
            # Generate descriptions for all tables in current example_id
            example_desc_dict = {}
            table_success = 0
            table_failed = 0
            
            for i, schema_str in enumerate(tbs):
                try:
                    tname = schema_str.split('\n', 1)[0].replace('Table full name:', '').strip()
                    
                    logger.info(f"Processing table: {example_id}/{tname} ({i+1}/{len(tbs)})")
                    desc = ask_model_desc_(schema_str, chat_global)
                    
                    # Check if None returned (Bad Request or other errors)
                    if desc is None:
                        logger.warning(f"Skipping table {example_id}/{tname}: model returned None")
                        table_failed += 1
                        continue
                    
                    logger.info(f"Description generated successfully: {example_id}/{tname} -> {desc[:50]}...")
                    example_desc_dict[tname] = desc
                    table_success += 1
                    
                except Exception as e:
                    logger.error(f"Failed to process table: {example_id}/table_{i} - {e}")
                    import traceback
                    logger.error(traceback.format_exc())
                    table_failed += 1
                    continue
            
            # Save to the path corresponding to this example_id
            example_dir = os.path.join(example_root, example_id)
            desc_file = os.path.join(example_dir, "table_descriptions.json")
            
            os.makedirs(example_dir, exist_ok=True)
            with open(desc_file, 'w', encoding='utf-8') as f:
                json.dump(example_desc_dict, f, indent=2, ensure_ascii=False)
                
            logger.info(f"Example {example_id} completed: {table_success} success, {table_failed} failed, saved to {desc_file}")
            processed_count += 1
            
        except Exception as e:
            logger.error(f"Failed to process example: {example_id} - {e}")
            import traceback
            logger.error(traceback.format_exc())
            failed_count += 1
            continue
    
    logger.info(f"Description generation completed! Successfully processed {processed_count} examples, failed {failed_count}")



# ================= Database-Grouped Schema Graph Construction =================

def collect_tables_by_database(example_root: str):
    """
    遍历所有样本，按数据库分组收集表信息，支持任务级别的精确隔离
    对于bq001这样的任务，只收集对应的具体数据库路径，如bq001.bigquery-public-data.google_analytics_sample
    支持普通样本（JSON文件）和local样本（SQLite文件）
    :param example_root: 样本根目录
    :return: Dict[task_specific_database_name, List[table_info_with_metadata]]
    """
    database_groups = {}
    dictionaries = [d for d in os.listdir(example_root) if os.path.isdir(os.path.join(example_root, d))]
    
    logger.info(f"Collecting tables from {len(dictionaries)} samples with task isolation")
    
    for example_id in tqdm(dictionaries, desc="Processing samples with task isolation"):
        sample_dir = os.path.join(example_root, example_id)
        if not os.path.exists(sample_dir):
            continue
        
        # 处理local样本（SQLite文件）
        if example_id.startswith("local"):
            sqlite_files = [f for f in os.listdir(sample_dir) if f.endswith('.sqlite')]
            if not sqlite_files:
                logger.warning(f"Local example {example_id} has no SQLite files, skipping")
                continue
            
            # 通常local样本只有一个sqlite文件
            sqlite_file = sqlite_files[0]
            sqlite_path = os.path.join(sample_dir, sqlite_file)
            # 🎯 关键修改：使用example_id作为前缀，确保任务隔离
            database_name = f"{example_id}_{sqlite_file.replace('.sqlite', '')}"
            
            logger.debug(f"Processing local example {example_id} with database: {database_name}")
            
            # 初始化数据库组
            if database_name not in database_groups:
                database_groups[database_name] = []
            
            try:
                import sqlite3
                connection = sqlite3.connect(sqlite_path)
                cursor = connection.cursor()
                
                # 获取所有表
                cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
                tables = cursor.fetchall()
                
                for (table_name,) in tables:
                    if table_name.startswith('sqlite_'):  # 跳过系统表
                        continue
                    
                    try:
                        # 获取列信息
                        cursor.execute(f"PRAGMA table_info({table_name})")
                        columns_info = cursor.fetchall()
                        
                        column_names = [col[1] for col in columns_info]
                        column_types = [col[2] for col in columns_info]
                        
                        # 获取样本行
                        sample_rows = ""
                        try:
                            cursor.execute(f"SELECT * FROM {table_name} LIMIT 3")
                            rows = cursor.fetchall()
                            sample_rows = str(rows) if rows else ""
                        except Exception as e:
                            logger.warning(f"Error fetching sample rows for {table_name}: {e}")
                        
                        # 构建表信息，包含元数据
                        table_info = {
                            'table_fullname': table_name,
                            'column_names': column_names,
                            'column_types': column_types,
                            'sample_rows': sample_rows,
                            'metadata': {
                                'example_id': example_id,
                                'database_name': database_name,
                                'schema_name': 'main',  # SQLite默认schema
                                'json_path': sqlite_path,
                                'schema_str': build_schema_str({
                                    'table_fullname': table_name,
                                    'column_names': column_names,
                                    'column_types': column_types,
                                    'sample_rows': sample_rows
                                }, table_name)
                            }
                        }
                        
                        database_groups[database_name].append(table_info)
                        logger.debug(f"Added table {table_name} from local example {example_id}")
                        
                    except Exception as e:
                        logger.error(f"Error processing table {table_name} in {example_id}: {e}")
                        continue
                
                connection.close()
                
            except Exception as e:
                logger.error(f"Error reading SQLite file {sqlite_path}: {e}")
                continue
        
        else:
            # 处理普通样本（JSON文件）- 🎯 关键修改：创建任务特定的数据库路径
            # 遍历样本目录下的所有数据库/项目目录
            for database_name in os.listdir(sample_dir):
                database_path = os.path.join(sample_dir, database_name)
                if not os.path.isdir(database_path):
                    continue
                    
                # 跳过特殊目录
                if database_name in ['spider', 'output', '__pycache__']:
                    continue
                    
                # 遍历数据库下的schema目录
                for schema_name in os.listdir(database_path):
                    schema_path = os.path.join(database_path, schema_name)
                    if not os.path.isdir(schema_path):
                        continue
                    
                    # 🎯 关键修改：创建任务特定的数据库标识符
                    # 格式：example_id.database_name.schema_name
                    task_specific_db_name = f"{example_id}.{database_name}.{schema_name}"
                    
                    # 初始化任务特定的数据库组
                    if task_specific_db_name not in database_groups:
                        database_groups[task_specific_db_name] = []
                        
                    # 查找DDL.csv文件
                    ddl_path = os.path.join(schema_path, "DDL.csv")
                    if not os.path.exists(ddl_path):
                        continue
                        
                    # 读取表信息
                    try:
                        with open(ddl_path, "r", newline="", encoding="utf-8", errors="ignore") as f:
                            reader = csv.reader(f)
                            header = next(reader, None)
                            
                            for row in reader:
                                if len(row) >= 2:
                                    table_name = row[0]
                                    
                                    # 从表名提取简化名称
                                    if "." in table_name:
                                        table_name = table_name.split(".")[-1]
                                        
                                    # 尝试读取对应的JSON文件
                                    json_files = [
                                        os.path.join(schema_path, f"{table_name}.json"),
                                        os.path.join(schema_path, f"{schema_name}.{table_name}.json")
                                    ]
                                    
                                    table_json = None
                                    for json_file in json_files:
                                        if os.path.exists(json_file):
                                            try:
                                                with open(json_file, 'r', encoding='utf-8') as jf:
                                                    table_json = json.load(jf)
                                                break
                                            except:
                                                continue
                                    
                                    if table_json:
                                        # 构建表信息，包含完整的元数据
                                        table_info = {
                                            'table_fullname': table_json.get('table_fullname', table_name),
                                            'column_names': table_json.get('column_names', []),
                                            'column_types': table_json.get('column_types', []),
                                            'sample_rows': table_json.get('sample_rows', ''),
                                            'metadata': {
                                                'example_id': example_id,
                                                'database_name': database_name,
                                                'schema_name': schema_name,
                                                'task_specific_db_name': task_specific_db_name,
                                                'json_path': json_file if 'json_file' in locals() else '',
                                                'schema_str': build_schema_str(table_json, table_name)
                                            }
                                        }
                                        database_groups[task_specific_db_name].append(table_info)
                                        
                    except Exception as e:
                        logger.error(f"Error reading DDL for {example_id}/{database_name}/{schema_name}: {e}")
                        continue
    
    # 统计信息
    for db_name, tables in database_groups.items():
        logger.info(f"Task-isolated database '{db_name}': {len(tables)} tables collected")
    
    logger.info(f"Total task-isolated databases found: {len(database_groups)}")
    return database_groups

def build_database_specific_graphs(database_groups: Dict[str, List[Dict]], 
                                 output_dir: str = "database_graphs",
                                 desc_sim_th: float = 0.7, 
                                 minhash_th: float = 0.8,
                                 use_descriptions: bool = True,
                                 profiling_dir: str = None,
                                 example_root: str = None,
                                 fk_relations: Dict[str, List[Dict]] = None,
                                 enable_ind_aind: bool = True,
                                 aind_threshold: float = 0.95,
                                 ind_sample_limit: int = 50000) -> Dict[str, nx.Graph]:
    """
    为每个数据库独立构建Schema Graph，支持外键关系增强（包括IND/AIND）

    Args:
        database_groups: 按数据库分组的表信息  
        output_dir: 图文件输出目录
        desc_sim_th: 描述相似度阈值
        minhash_th: MinHash相似度阈值
        use_descriptions: 是否使用表描述
        profiling_dir: profiling JSON文件目录（用于读取预计算的MinHash）
        example_root: 样本根目录（用于读取table_descriptions.json）
        fk_relations: 外键关系字典 {database_name: [foreign_key_constraints]}
        enable_ind_aind: 是否启用IND/AIND隐式外键检测（默认True）
        aind_threshold: AIND的置信度阈值（默认0.95）
        ind_sample_limit: IND/AIND检测时每列最大采样行数（默认50000）
    :return: Dict[database_name, Graph]
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # 加载语义模型（如果使用描述）
    model = get_embedding_model() if use_descriptions else None
    
    # 1. 预加载profiling数据（如果提供了profiling_dir）
    profiling_data = {}
    if profiling_dir and os.path.exists(profiling_dir):
        logger.info(f"Loading profiling data from {profiling_dir}")
        for prof_file in os.listdir(profiling_dir):
            if prof_file.endswith('_profiling.json'):
                prof_path = os.path.join(profiling_dir, prof_file)
                try:
                    with open(prof_path, 'r', encoding='utf-8') as f:
                        prof_data = json.load(f)
                    
                    # 提取表名（从文件名或数据中）
                    table_fullname = prof_data.get('table_fullname')
                    if not table_fullname:
                        # 从文件名推断：example_database_schema_table_profiling.json
                        parts = prof_file.replace('_profiling.json', '').split('_')
                        if len(parts) >= 4:
                            table_fullname = f"{parts[1]}.{parts[2]}.{'_'.join(parts[3:])}"
                    
                    if table_fullname:
                        profiling_data[table_fullname] = prof_data
                        
                except Exception as e:
                    logger.warning(f"Error loading profiling file {prof_file}: {e}")
        
        logger.info(f"Loaded profiling data for {len(profiling_data)} tables")
    else:
        logger.warning("No profiling directory provided, will compute MinHash on-the-fly")
    
    # 2. 预加载表描述数据和语义向量（如果提供了example_root且使用描述）
    table_descriptions = {}
    all_embeddings = {}  # 预计算的语义向量
    
    if use_descriptions and example_root and os.path.exists(example_root):
        logger.info(f"Loading table descriptions and embeddings from {example_root}")
        for example_id in os.listdir(example_root):
            example_dir = os.path.join(example_root, example_id)
            if not os.path.isdir(example_dir):
                continue
                
            # 加载表描述
            desc_file = os.path.join(example_dir, "table_descriptions.json")
            if os.path.exists(desc_file):
                try:
                    with open(desc_file, 'r', encoding='utf-8') as f:
                        desc_data = json.load(f)
                    
                    # 合并到全局字典中
                    for table_name, description in desc_data.items():
                        table_descriptions[table_name] = description
                        
                except Exception as e:
                    logger.warning(f"Error loading descriptions from {desc_file}: {e}")
            
            # 尝试加载预计算的语义向量
            embeddings = load_table_embeddings(example_id, example_root)
            if embeddings:
                all_embeddings.update(embeddings)
                logger.debug(f"Loaded {len(embeddings)} pre-computed embeddings from {example_id}")
        
        logger.info(f"Loaded descriptions for {len(table_descriptions)} tables")
        logger.info(f"Loaded pre-computed embeddings for {len(all_embeddings)} tables")
    elif use_descriptions:
        logger.warning("No example_root provided for loading table descriptions, will use schema strings")
    
    database_graphs = {}
    
    for database_name, tables in database_groups.items():
        if len(tables) < 1:  # 至少需要1个表才能构建图
            logger.warning(f"Database '{database_name}' has no tables, skipping graph construction")
            continue
            
        logger.info(f"Building schema graph for database '{database_name}' with {len(tables)} tables")
        
        # 获取当前数据库的外键关系
        current_fk_relations = fk_relations.get(database_name, []) if fk_relations else []
        if current_fk_relations:
            logger.info(f"  📎 Found {len(current_fk_relations)} foreign key relationships for {database_name}")
        
        # 创建图
        G = nx.Graph()
        table_info = {}
        
        # 3. 添加节点并准备表级信息
        for i, table in enumerate(tables):
            table_name = table['table_fullname']
            
            # 添加节点
            G.add_node(table_name, **table['metadata'])
            
            # 准备表级MinHash和描述embedding
            table_minhash = None
            table_embedding = None
            
            # A. 尝试从profiling数据读取MinHash
            if table_name in profiling_data:
                try:
                    prof_data = profiling_data[table_name]
                    # 聚合列级MinHash到表级
                    table_minhash = MinHash(num_perm=128)
                    
                    # 检查profiling数据结构
                    col_data_found = False
                    for col_name, col_info in prof_data.items():
                        if isinstance(col_info, dict) and 'minhash' in col_info:
                            col_mh = MinHash(num_perm=128)
                            # 恢复MinHash对象
                            minhash_values = col_info['minhash']
                            if isinstance(minhash_values, list) and len(minhash_values) == 128:
                                col_mh.hashvalues = np.array(minhash_values, dtype=np.uint64)
                                # 聚合到表级MinHash（取最小值）
                                for j in range(128):
                                    table_minhash.hashvalues[j] = min(table_minhash.hashvalues[j], col_mh.hashvalues[j])
                                col_data_found = True
                    
                    if not col_data_found:
                        logger.debug(f"No column profiling data found for {table_name}")
                        table_minhash = None
                        
                except Exception as e:
                    logger.warning(f"Error processing profiling data for {table_name}: {e}")
                    table_minhash = None
            
            # B. 如果没有profiling数据，现场计算MinHash
            if table_minhash is None:
                table_minhash = MinHash(num_perm=128)
                
                # 修改：MinHash基于列名而不是样本行
                for col_name in table.get('column_names', []):
                    table_minhash.update(col_name.encode('utf8'))
            
            # C. 获取或计算语义向量
            if use_descriptions and model:
                # 优先使用预计算的语义向量
                if table_name in all_embeddings:
                    table_embedding = all_embeddings[table_name]
                    logger.debug(f"Using pre-computed embedding for {table_name}")
                else:
                    # 没有预计算向量，现场计算
                    # 基础schema信息
                    base_schema = table['metadata']['schema_str']
                    
                    # 增强：将table_descriptions与schema信息结合（如果可用）
                    if table_name in table_descriptions and table_descriptions[table_name]:
                        desc_raw = table_descriptions[table_name]
                        desc = desc_raw.get("description", str(desc_raw)) if isinstance(desc_raw, dict) else str(desc_raw)
                        # 采用与rerank相同的格式
                        enhanced_description = f"{base_schema}\n\nTable Description: {desc}"
                        logger.debug(f"Enhanced table {table_name} with description: {desc[:50]}...")
                    else:
                        # 如果没有预生成的描述，只使用schema信息
                        enhanced_description = base_schema
                        logger.debug(f"Using schema string only for {table_name}")
                    
                    try:
                        table_embedding = model.encode(enhanced_description, normalize_embeddings=True)
                        logger.debug(f"Computed embedding on-the-fly for {table_name}")
                    except Exception as e:
                        logger.warning(f"Error computing embedding for {table_name}: {e}")
            
            table_info[table_name] = {
                'minhash': table_minhash,
                'embedding': table_embedding
            }
        
        # 4. 建立数据库连接（用于IND/AIND隐式外键检测，仅支持SQLite本地数据库）
        db_connection = None
        if enable_ind_aind and example_root and len(tables) > 0:
            first_table_metadata = tables[0].get('metadata', {})
            example_id = first_table_metadata.get('example_id', '')
            if example_id.startswith('local'):
                db_path = first_table_metadata.get('json_path')
                if db_path and os.path.exists(db_path):
                    try:
                        import sqlite3
                        db_connection = sqlite3.connect(db_path)
                        logger.info(f"  🔌 Connected to database for IND/AIND detection: {db_path}")
                    except Exception as e:
                        logger.warning(f"  ⚠️  Failed to connect to database for IND/AIND: {e}")

        # 4.5 准备表的列信息（用于IND/AIND检测）
        tables_metadata = {}
        for table in tables:
            table_name = table['table_fullname']
            tables_metadata[table_name] = {
                'table_name': table_name,
                'columns': table.get('column_names', []),
                'metadata': table.get('metadata', {})
            }

        # 5. 计算边
        table_names = list(table_info.keys())
        edges_added = 0
        fk_edges_added = 0
        ind_edges_added = 0
        aind_edges_added = 0

        for i in range(len(table_names)):
            for j in range(i+1, len(table_names)):
                t1, t2 = table_names[i], table_names[j]
                
                # 检查是否存在显式外键关系
                has_fk_relation = has_foreign_key_relationship(t1, t2, current_fk_relations)
                
                # 计算MinHash相似度
                mh_sim = 0.0
                try:
                    if (table_info[t1]['minhash'] is not None and 
                        table_info[t2]['minhash'] is not None):
                        mh_sim = table_info[t1]['minhash'].jaccard(table_info[t2]['minhash'])
                except Exception as e:
                    logger.debug(f"Error computing MinHash similarity between {t1} and {t2}: {e}")
                
                # 计算描述相似度
                desc_sim = 0.0
                if (use_descriptions and 
                    table_info[t1]['embedding'] is not None and 
                    table_info[t2]['embedding'] is not None):
                    try:
                        desc_sim = float(QwenEmbeddingModel.cos_sim(table_info[t1]['embedding'], table_info[t2]['embedding']))
                    except Exception as e:
                        logger.debug(f"Error computing description similarity between {t1} and {t2}: {e}")
                
                # 判断是否添加边
                should_add_edge = False
                edge_reasons = []
                edge_weight = 1.0
                fk_type = None        # 'explicit' | 'IND' | 'AIND' | None
                ind_info = None

                # 【策略1】优先考虑显式外键关系
                if has_fk_relation:
                    should_add_edge = True
                    edge_weight = 1.0
                    edge_reasons.append("foreign_key=1.0")
                    fk_type = 'explicit'
                    fk_edges_added += 1
                    logger.debug(f"  🔗 Explicit FK edge: {t1} -> {t2}")

                # 【策略2】没有显式FK时，检查IND/AIND隐式外键
                elif enable_ind_aind and db_connection:
                    ind_result = check_IND_AIND_relationship(
                        tables_metadata.get(t1, {}),
                        tables_metadata.get(t2, {}),
                        db_connection,
                        aind_threshold=aind_threshold,
                        sample_limit=ind_sample_limit
                    )
                    if ind_result['has_relationship']:
                        should_add_edge = True
                        edge_weight = 1.0  # IND/AIND 视同外键，权重为 1
                        fk_type = ind_result['relationship_type']
                        ind_info = ind_result
                        confidence = ind_result['confidence']
                        edge_reasons.append(f"{fk_type}={confidence:.3f}")
                        if fk_type == 'IND':
                            ind_edges_added += 1
                            logger.debug(f"  🔗 IND edge: {t1} -> {t2} (conf: {confidence:.3f})")
                        else:
                            aind_edges_added += 1
                            logger.debug(f"  🔗 AIND edge: {t1} -> {t2} (conf: {confidence:.3f})")

                # 【策略3】无 FK-like 关系时，使用 MinHash / 描述相似度
                if not should_add_edge:
                    if mh_sim >= minhash_th:
                        should_add_edge = True
                        edge_weight = mh_sim
                        edge_reasons.append(f"minhash={mh_sim:.3f}")
                    elif desc_sim >= desc_sim_th:
                        should_add_edge = True
                        edge_weight = desc_sim
                        edge_reasons.append(f"desc_sim={desc_sim:.3f}")
                
                if should_add_edge:
                    edge_attrs = {
                        'mh_sim': round(mh_sim, 3),
                        'desc_sim': round(desc_sim, 3),
                        'fk_relation': (fk_type is not None),
                        'fk_type': fk_type,  # 'explicit' | 'IND' | 'AIND' | None
                        'weight': edge_weight,
                        'reason': ", ".join(edge_reasons)
                    }
                    if ind_info:
                        edge_attrs['ind_confidence'] = round(ind_info['confidence'], 3)
                        edge_attrs['ind_direction'] = ind_info['direction']
                        edge_attrs['ind_column_pair'] = ind_info['column_pair']
                    G.add_edge(t1, t2, **edge_attrs)
                    edges_added += 1

        # 关闭数据库连接
        if db_connection:
            try:
                db_connection.close()
            except Exception:
                pass

        logger.info(f"Database '{database_name}': {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
        if fk_edges_added > 0:
            logger.info(f"  📎 Explicit FK edges: {fk_edges_added}")
        if ind_edges_added > 0:
            logger.info(f"  🔗 IND edges (strict inclusion): {ind_edges_added}")
        if aind_edges_added > 0:
            logger.info(f"  🔗 AIND edges (approx inclusion, τ≥{aind_threshold}): {aind_edges_added}")
        total_fk_like = fk_edges_added + ind_edges_added + aind_edges_added
        if total_fk_like > 0:
            logger.info(f"  ✅ Total FK-like edges: {total_fk_like} "
                        f"({fk_edges_added} explicit + {ind_edges_added} IND + {aind_edges_added} AIND)")
        
        # 6. 保存图
        graph_path = os.path.join(output_dir, f"{database_name}_schema_graph.gpickle")
        save_graph(G, graph_path)
        
        # 6. 保存图的摘要信息
        summary_path = os.path.join(output_dir, f"{database_name}_graph_summary.json")
        summary = {
            'database_name': database_name,
            'nodes_count': G.number_of_nodes(),
            'edges_count': G.number_of_edges(),
            'tables': list(G.nodes()),
            'edges': [(u, v, data) for u, v, data in G.edges(data=True)],
            'connected_components': len(list(nx.connected_components(G))),
            'graph_density': nx.density(G),
            'data_sources': {
                'profiling_used': len([t for t in table_names if profiling_data.get(t)]),
                'descriptions_used': len([t for t in table_names if table_descriptions.get(t)]),
                'total_tables': len(table_names)
            },
            'foreign_key_info': {
                'total_fk_relations': len(current_fk_relations),
                'fk_edges_in_graph': fk_edges_added,
                'ind_edges_in_graph': ind_edges_added,
                'aind_edges_in_graph': aind_edges_added,
                'fk_relations': current_fk_relations
            }
        }
        
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        
        database_graphs[database_name] = G
        logger.info(f"Graph saved: {graph_path}")
        logger.info(f"Summary saved: {summary_path}")
        logger.info(f"Data sources - Profiling: {summary['data_sources']['profiling_used']}/{len(table_names)}, "
                   f"Descriptions: {summary['data_sources']['descriptions_used']}/{len(table_names)}")
    
    return database_graphs

def analyze_database_graphs(output_dir: str = "database_graphs"):
    """
    分析已构建的数据库图，生成统计报告
    """
    if not os.path.exists(output_dir):
        logger.error(f"Output directory {output_dir} does not exist")
        return
    
    summary_files = [f for f in os.listdir(output_dir) if f.endswith('_graph_summary.json')]
    
    if not summary_files:
        logger.error("No graph summary files found")
        return
    
    total_stats = {
        'total_databases': len(summary_files),
        'total_tables': 0,
        'total_edges': 0,
        'database_details': []
    }
    
    for summary_file in summary_files:
        summary_path = os.path.join(output_dir, summary_file)
        try:
            with open(summary_path, 'r', encoding='utf-8') as f:
                summary = json.load(f)
            
            total_stats['total_tables'] += summary['nodes_count']
            total_stats['total_edges'] += summary['edges_count']
            total_stats['database_details'].append({
                'database': summary['database_name'],
                'tables': summary['nodes_count'],
                'edges': summary['edges_count'],
                'density': summary['graph_density'],
                'components': summary['connected_components']
            })
            
        except Exception as e:
            logger.error(f"Error reading {summary_file}: {e}")
    
    # 输出统计报告
    logger.info("="*50)
    logger.info("DATABASE SCHEMA GRAPHS ANALYSIS")
    logger.info("="*50)
    logger.info(f"Total databases: {total_stats['total_databases']}")
    logger.info(f"Total tables: {total_stats['total_tables']}")
    logger.info(f"Total edges: {total_stats['total_edges']}")
    logger.info(f"Average tables per database: {total_stats['total_tables']/total_stats['total_databases']:.1f}")
    logger.info(f"Average edges per database: {total_stats['total_edges']/total_stats['total_databases']:.1f}")
    
    logger.info("\nPer-Database Details:")
    for db_detail in sorted(total_stats['database_details'], key=lambda x: x['tables'], reverse=True):
        logger.info(f"  {db_detail['database']}: {db_detail['tables']} tables, "
                   f"{db_detail['edges']} edges, density={db_detail['density']:.3f}, "
                   f"components={db_detail['components']}")
    
    # 保存完整报告
    report_path = os.path.join(output_dir, "analysis_report.json")
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(total_stats, f, indent=2, ensure_ascii=False)
    
    logger.info(f"\nDetailed report saved to: {report_path}")

# ================ prompts.txt Generation ================

def generate_all_prompts_txt(example_root: str, force_rebuild: bool = False):
    """
    为所有样本生成/重建 prompts.txt 文件
    :param example_root: 样本根目录
    :param force_rebuild: 是否强制重建（即使已存在）
    """
    if not os.path.exists(example_root):
        logger.error(f"Example root directory not found: {example_root}")
        return
    
    dictionaries = [d for d in os.listdir(example_root) 
                   if os.path.isdir(os.path.join(example_root, d))]
    
    logger.info(f"Starting prompts.txt generation for {len(dictionaries)} examples")
    logger.info(f"Example root: {example_root}")
    logger.info(f"Force rebuild: {force_rebuild}")
    
    success_count = 0
    failed_count = 0
    skipped_count = 0
    
    for example_id in tqdm(dictionaries, desc="Generating prompts.txt"):
        try:
            sample_dir = os.path.join(example_root, example_id)
            prompts_path = os.path.join(sample_dir, "prompts.txt")
            
            # 检查是否已存在（除非强制重建）
            if os.path.exists(prompts_path) and not force_rebuild:
                logger.debug(f"prompts.txt already exists for {example_id}, skipping")
                skipped_count += 1
                continue
            
            # 获取所有表信息
            tables_info = get_table_info_from_directory(example_root, example_id)
            if not tables_info:
                logger.warning(f"No tables found for {example_id}")
                failed_count += 1
                continue
            
            # 构建内容
            content_parts = []
            for table_info in tables_info:
                content_parts.append(table_info.strip())
            
            content = "\n" + ("-" * 50 + "\n").join(content_parts)
            
            # 获取外部知识信息
            external_knowledge = get_external_knowledge(example_root, example_id)
            if external_knowledge:
                content += f"\n{'-'*50}\nExternal knowledge that might be helpful: \n{external_knowledge}"
            
            # 备份现有文件
            if os.path.exists(prompts_path):
                backup_path = prompts_path + ".bak"
                if os.path.exists(backup_path):
                    os.remove(backup_path)  # 删除旧备份
                os.rename(prompts_path, backup_path)
                logger.debug(f"Backed up existing prompts.txt to {backup_path}")
            
            # 写入新文件
            with open(prompts_path, 'w', encoding='utf-8') as f:
                f.write(content.strip())
            
            logger.info(f"Generated prompts.txt for {example_id}: {len(tables_info)} tables")
            success_count += 1
            
        except Exception as e:
            logger.error(f"Error generating prompts.txt for {example_id}: {e}")
            failed_count += 1
            continue
    
    logger.info(f"prompts.txt generation completed:")
    logger.info(f"  - Success: {success_count}")
    logger.info(f"  - Failed: {failed_count}")
    logger.info(f"  - Skipped: {skipped_count}")


def get_complete_neighbor_schema(neighbor_name, table_descriptions, use_description, example_root, current_example_id):
    """
    获取邻居表的完整schema信息，包含列信息和样本行
    :param neighbor_name: 邻居表名
    :param table_descriptions: 表描述字典
    :param use_description: 是否使用描述
    :param example_root: 样本根目录
    :param current_example_id: 当前样本ID
    :return: 完整的schema字符串或None
    """
    if not example_root or not current_example_id:
        return None
    
    try:
        # 尝试从当前样本目录中获取完整的表信息
        sample_dir = os.path.join(example_root, current_example_id)
        if not os.path.exists(sample_dir):
            return None
        
        # 遍历样本目录查找表的JSON文件
        for project_name in os.listdir(sample_dir):
            if project_name == "spider":
                continue
                
            project_path = os.path.join(sample_dir, project_name)
            if not os.path.isdir(project_path):
                continue
                
            # 遍历项目下的数据库目录
            for db_name in os.listdir(project_path):
                db_path = os.path.join(project_path, db_name)
                if not os.path.isdir(db_path):
                    continue
                
                # 尝试多种文件名匹配方式
                json_files_to_try = [
                    os.path.join(db_path, f"{neighbor_name}.json"),
                    os.path.join(db_path, f"{db_name}.{neighbor_name}.json"),
                    os.path.join(db_path, f"{neighbor_name.split('.')[-1]}.json")
                ]
                
                for json_file in json_files_to_try:
                    if os.path.exists(json_file):
                        try:
                            with open(json_file, 'r', encoding='utf-8') as f:
                                table_json = json.load(f)
                            
                            # 检查表名匹配
                            table_fullname = table_json.get('table_fullname', '')
                            if table_fullname == neighbor_name or neighbor_name.endswith(table_fullname):
                                # 构建完整的schema信息
                                schema_info = f"Table full name: {table_fullname or neighbor_name}\n"
                                
                                # 处理列信息
                                column_names = table_json.get('column_names', [])
                                column_types = table_json.get('column_types', [])
                                descriptions = table_json.get('description', [])
                                
                                for j in range(len(column_names)):
                                    col_type = column_types[j] if j < len(column_types) else "UNKNOWN"
                                    
                                    # 添加列级描述信息（如果有）
                                    description_text = ""
                                    if j < len(descriptions) and descriptions[j]:
                                        description_text = f" Description: {descriptions[j]}"
                                    
                                    schema_info += f"Column name: {column_names[j]} Type: {col_type}{description_text}\n"
                                
                                # 添加样本行（如果有，限制数量避免过长）
                                if 'sample_rows' in table_json and table_json['sample_rows']:
                                    sample_rows = table_json['sample_rows']
                                    # 限制样本行数量和长度
                                    if isinstance(sample_rows, list):
                                        sample_rows = sample_rows[:2]  # 只取前2行
                                    schema_info += f"Sample rows:\n{sample_rows}\n"
                                
                                # 添加表级描述（如果有且启用）
                                if use_description and neighbor_name in table_descriptions:
                                    desc = table_descriptions[neighbor_name]
                                    if desc:
                                        schema_info += f"\nTable Description: {desc}\n"
                                
                                return schema_info.strip()
                                
                        except Exception as e:
                            logger.debug(f"Error reading JSON file {json_file}: {e}")
                            continue
        
        return None
        
    except Exception as e:
        logger.debug(f"Error getting complete neighbor schema for {neighbor_name}: {e}")
        return None

# ================= 外键关系检测 =================

def extract_foreign_key_constraints(ddl_content: str) -> List[Dict]:
    """
    从DDL内容中提取外键约束信息
    :param ddl_content: DDL语句内容
    :return: 外键约束列表，每个元素包含 {fk_table, fk_column, pk_table, pk_column}
    """
    foreign_keys = []
    
    # 匹配FOREIGN KEY约束的正则表达式
    # 支持多种格式：
    # 1. CONSTRAINT name FOREIGN KEY (col) REFERENCES table(col)
    # 2. FOREIGN KEY (col) REFERENCES table(col)
    # 3. CONSTRAINT name FOREIGN KEY (col1, col2) REFERENCES table(col1, col2)
    
    # 单列外键
    single_col_pattern = r'(?:CONSTRAINT\s+\w+\s+)?FOREIGN\s+KEY\s*\(\s*`?(\w+)`?\s*\)\s+REFERENCES\s+`?(\w+)`?\s*\(\s*`?(\w+)`?\s*\)'
    
    # 多列外键（简化处理，只取第一列）
    multi_col_pattern = r'(?:CONSTRAINT\s+\w+\s+)?FOREIGN\s+KEY\s*\(\s*`?(\w+)`?(?:\s*,\s*`?\w+`?)*\s*\)\s+REFERENCES\s+`?(\w+)`?\s*\(\s*`?(\w+)`?(?:\s*,\s*`?\w+`?)*\s*\)'
    
    # 查找所有外键约束
    for match in re.finditer(single_col_pattern, ddl_content, re.IGNORECASE):
        fk_column = match.group(1)
        pk_table = match.group(2)
        pk_column = match.group(3)
        
        # 从DDL内容中提取当前表名
        table_match = re.search(r'CREATE\s+(?:OR\s+REPLACE\s+)?TABLE\s+`?([^\s(]+)`?', ddl_content, re.IGNORECASE)
        if table_match:
            fk_table = table_match.group(1)
            foreign_keys.append({
                'fk_table': fk_table,
                'fk_column': fk_column,
                'pk_table': pk_table,
                'pk_column': pk_column
            })
    
    # 如果没有找到单列外键，尝试多列外键
    if not foreign_keys:
        for match in re.finditer(multi_col_pattern, ddl_content, re.IGNORECASE):
            fk_column = match.group(1)
            pk_table = match.group(2)
            pk_column = match.group(3)
            
            table_match = re.search(r'CREATE\s+(?:OR\s+REPLACE\s+)?TABLE\s+`?([^\s(]+)`?', ddl_content, re.IGNORECASE)
            if table_match:
                fk_table = table_match.group(1)
                foreign_keys.append({
                    'fk_table': fk_table,
                    'fk_column': fk_column,
                    'pk_table': pk_table,
                    'pk_column': pk_column
                })
    
    return foreign_keys


def collect_foreign_key_relationships(example_root: str) -> Dict[str, List[Dict]]:
    """
    收集所有样本中的外键关系
    :param example_root: 样本根目录
    :return: 按数据库分组的 {database_name: [foreign_key_constraints]}
    """
    database_fk_relations = {}
    
    dictionaries = [d for d in os.listdir(example_root) 
                   if os.path.isdir(os.path.join(example_root, d))]
    
    logger.info(f"Collecting foreign key relationships from {len(dictionaries)} samples")
    
    for example_id in tqdm(dictionaries, desc="Collecting FK relationships"):
        sample_dir = os.path.join(example_root, example_id)
        if not os.path.exists(sample_dir):
            continue
        
        # 处理local样本（SQLite文件）
        if example_id.startswith("local"):
            sqlite_files = [f for f in os.listdir(sample_dir) if f.endswith('.sqlite')]
            if not sqlite_files:
                continue
            
            sqlite_file = sqlite_files[0]
            # 🎯 关键修改：使用与collect_tables_by_database一致的命名
            database_name = f"{example_id}_{sqlite_file.replace('.sqlite', '')}"
            
            if database_name not in database_fk_relations:
                database_fk_relations[database_name] = []
            
            # 从SQLite中提取外键信息
            try:
                import sqlite3
                sqlite_path = os.path.join(sample_dir, sqlite_file)
                connection = sqlite3.connect(sqlite_path)
                cursor = connection.cursor()
                
                # 方法1: 尝试PRAGMA foreign_key_list
                cursor.execute("PRAGMA foreign_key_list")
                fk_list = cursor.fetchall()
                
                if fk_list:
                    logger.debug(f"Found {len(fk_list)} FK constraints via PRAGMA for {database_name}")
                    for fk_info in fk_list:
                        # SQLite PRAGMA foreign_key_list返回: (id, seq, table, from, to, on_update, on_delete, match)
                        fk_table = fk_info[2]  # 外键表
                        fk_column = fk_info[3]  # 外键列
                        pk_table = fk_info[4]   # 主键表
                        pk_column = fk_info[5]  # 主键列
                        
                        database_fk_relations[database_name].append({
                            'fk_table': fk_table,
                            'fk_column': fk_column,
                            'pk_table': pk_table,
                            'pk_column': pk_column
                        })
                
                # 方法2: 如果PRAGMA没有结果，从schema中提取
                if not fk_list:
                    logger.debug(f"No FK constraints via PRAGMA for {database_name}, trying schema extraction")
                    
                    # 获取所有表的schema
                    cursor.execute("SELECT sql FROM sqlite_master WHERE type='table' AND sql LIKE '%FOREIGN KEY%'")
                    fk_schemas = cursor.fetchall()
                    
                    if fk_schemas:
                        logger.debug(f"Found {len(fk_schemas)} tables with FK in schema for {database_name}")
                        
                        for schema_row in fk_schemas:
                            schema_sql = schema_row[0]
                            if schema_sql:
                                # 使用正则表达式提取外键约束
                                foreign_keys = extract_foreign_key_constraints(schema_sql)
                                database_fk_relations[database_name].extend(foreign_keys)
                                
                                logger.debug(f"Extracted {len(foreign_keys)} FK constraints from schema")
                
                connection.close()
                
                if database_fk_relations[database_name]:
                    logger.info(f"Database '{database_name}': {len(database_fk_relations[database_name])} foreign key relationships collected")
                else:
                    logger.debug(f"Database '{database_name}': No foreign key relationships found")
                
            except Exception as e:
                logger.warning(f"Error extracting FK from SQLite {sqlite_file}: {e}")
                continue
        
        else:
            # 处理普通样本（JSON文件）
            # 遍历样本目录下的所有数据库/项目目录
            for database_name in os.listdir(sample_dir):
                database_path = os.path.join(sample_dir, database_name)
                if not os.path.isdir(database_path):
                    continue
                    
                # 跳过特殊目录
                if database_name in ['spider', 'output', '__pycache__']:
                    continue
                    
                # 初始化数据库组
                if database_name not in database_fk_relations:
                    database_fk_relations[database_name] = []
                    
                # 遍历数据库下的schema目录
                for schema_name in os.listdir(database_path):
                    schema_path = os.path.join(database_path, schema_name)
                    if not os.path.isdir(schema_path):
                        continue
                        
                    # 查找DDL.csv文件
                    ddl_path = os.path.join(schema_path, "DDL.csv")
                    if not os.path.exists(ddl_path):
                        continue
                        
                    # 读取DDL内容并提取外键约束
                    try:
                        with open(ddl_path, "r", newline="", encoding="utf-8", errors="ignore") as f:
                            reader = csv.reader(f)
                            header = next(reader, None)
                            
                            for row in reader:
                                if len(row) >= 2:
                                    # 找到包含 CREATE 语句的列
                                    create_col_idx = None
                                    for i, col in enumerate(row):
                                        if col and col.upper().startswith("CREATE"):
                                            create_col_idx = i
                                            break
                                    
                                    if create_col_idx is not None:
                                        ddl_statement = row[create_col_idx]
                                        foreign_keys = extract_foreign_key_constraints(ddl_statement)
                                        if foreign_keys:
                                            database_fk_relations[database_name].extend(foreign_keys)
                                            logger.debug(f"Found {len(foreign_keys)} FK constraints in DDL for {database_name}")
                                        
                    except Exception as e:
                        logger.warning(f"Error reading DDL for {example_id}/{database_name}/{schema_name}: {e}")
                        continue
    
    # 统计信息
    total_fk_relations = 0
    databases_with_fk = 0
    
    for db_name, fk_list in database_fk_relations.items():
        if fk_list:
            databases_with_fk += 1
            total_fk_relations += len(fk_list)
            logger.info(f"Database '{db_name}': {len(fk_list)} foreign key relationships collected")
        else:
            logger.debug(f"Database '{db_name}': No foreign key relationships found")
    
    logger.info(f"Total databases with FK relationships: {databases_with_fk}")
    logger.info(f"Total foreign key relationships collected: {total_fk_relations}")
    
    return database_fk_relations


def has_foreign_key_relationship(table1: str, table2: str, fk_relations: List[Dict]) -> bool:
    """
    检查两个表之间是否存在外键关系
    :param table1: 第一个表名
    :param table2: 第二个表名
    :param fk_relations: 外键关系列表
    :return: 是否存在外键关系
    """
    for fk in fk_relations:
        # 检查table1 -> table2的外键关系
        if (fk['fk_table'] == table1 and fk['pk_table'] == table2):
            return True
        # 检查table2 -> table1的外键关系
        if (fk['fk_table'] == table2 and fk['pk_table'] == table1):
            return True
    return False


def get_column_unique_values(db_connection, table_name: str, column_name: str,
                             sample_limit: int = 50000) -> set:
    """
    获取指定列的唯一非空值集合
    :param db_connection: 数据库连接对象
    :param table_name: 表名
    :param column_name: 列名
    :param sample_limit: 采样限制（避免大表查询过慢）
    :return: 唯一值的集合
    """
    try:
        cursor = db_connection.cursor()
        query = f"SELECT DISTINCT `{column_name}` FROM `{table_name}` WHERE `{column_name}` IS NOT NULL LIMIT {sample_limit}"
        cursor.execute(query)
        values = set(str(row[0]) for row in cursor.fetchall())
        return values
    except Exception as e:
        logger.debug(f"Error getting unique values for {table_name}.{column_name}: {e}")
        return set()


def compute_inclusion_dependency(values_A: set, values_B: set) -> tuple:
    """
    计算包含依赖 A ⊆ B 的置信度
    :return: (is_IND, confidence)
             - is_IND: 是否严格包含 (conf=1.0)
             - confidence: 置信度 |A∩B| / |A|
    """
    if not values_A:
        return False, 0.0
    intersection = values_A & values_B
    confidence = len(intersection) / len(values_A)
    is_IND = (confidence == 1.0)
    return is_IND, confidence


def check_IND_AIND_relationship(table1_info: Dict, table2_info: Dict,
                                db_connection,
                                aind_threshold: float = 0.95,
                                sample_limit: int = 50000) -> Dict:
    """
    检查两个表之间是否存在 IND（严格包含依赖）或 AIND（近似包含依赖）关系。
    用于在没有显式外键声明时发现隐式 JOIN 关系。

    :param table1_info: {'table_name': str, 'columns': List[str]}
    :param table2_info: {'table_name': str, 'columns': List[str]}
    :param db_connection: 数据库连接（SQLite）
    :param aind_threshold: AIND 的置信度阈值（默认 0.95）
    :param sample_limit: 每列最大采样行数
    :return: {
        'has_relationship': bool,
        'relationship_type': 'IND' | 'AIND' | None,
        'confidence': float,
        'direction': 't1->t2' | 't2->t1' | None,
        'column_pair': (col_A, col_B) | None
    }
    """
    result = {
        'has_relationship': False,
        'relationship_type': None,
        'confidence': 0.0,
        'direction': None,
        'column_pair': None
    }

    if not db_connection:
        return result

    try:
        table1_name = table1_info.get('table_name', '')
        table2_name = table2_info.get('table_name', '')
        table1_columns = table1_info.get('columns', [])
        table2_columns = table2_info.get('columns', [])

        if not table1_columns or not table2_columns:
            return result

        best_confidence = 0.0
        best_relation = None

        # 策略1：优先检查同名列（效率高，且语义上更可信）
        common_columns = set(table1_columns[:10]) & set(table2_columns[:10])

        for col_name in common_columns:
            values1 = get_column_unique_values(db_connection, table1_name, col_name, sample_limit)
            values2 = get_column_unique_values(db_connection, table2_name, col_name, sample_limit)

            if not values1 or not values2:
                continue

            is_IND_12, conf_12 = compute_inclusion_dependency(values1, values2)
            is_IND_21, conf_21 = compute_inclusion_dependency(values2, values1)

            if conf_12 >= aind_threshold and conf_12 >= conf_21:
                boosted_conf = min(1.0, conf_12 + 0.05)  # 同名列额外 +0.05 bonus
                if boosted_conf > best_confidence:
                    best_confidence = boosted_conf
                    best_relation = {
                        'relationship_type': 'IND' if is_IND_12 else 'AIND',
                        'confidence': conf_12,
                        'direction': 't1->t2',
                        'column_pair': (col_name, col_name)
                    }

            if conf_21 >= aind_threshold and conf_21 > conf_12:
                boosted_conf = min(1.0, conf_21 + 0.05)
                if boosted_conf > best_confidence:
                    best_confidence = boosted_conf
                    best_relation = {
                        'relationship_type': 'IND' if is_IND_21 else 'AIND',
                        'confidence': conf_21,
                        'direction': 't2->t1',
                        'column_pair': (col_name, col_name)
                    }

        # 策略2：若同名列无结果，检查所有列对（最多各取前10列）
        if not best_relation:
            for col1 in table1_columns[:10]:
                values1 = get_column_unique_values(db_connection, table1_name, col1, sample_limit)
                if not values1:
                    continue

                for col2 in table2_columns[:10]:
                    if col1 == col2 and col1 in common_columns:
                        continue

                    values2 = get_column_unique_values(db_connection, table2_name, col2, sample_limit)
                    if not values2:
                        continue

                    is_IND_12, conf_12 = compute_inclusion_dependency(values1, values2)
                    if conf_12 >= aind_threshold and conf_12 > best_confidence:
                        best_confidence = conf_12
                        best_relation = {
                            'relationship_type': 'IND' if is_IND_12 else 'AIND',
                            'confidence': conf_12,
                            'direction': 't1->t2',
                            'column_pair': (col1, col2)
                        }

                    is_IND_21, conf_21 = compute_inclusion_dependency(values2, values1)
                    if conf_21 >= aind_threshold and conf_21 > best_confidence:
                        best_confidence = conf_21
                        best_relation = {
                            'relationship_type': 'IND' if is_IND_21 else 'AIND',
                            'confidence': conf_21,
                            'direction': 't2->t1',
                            'column_pair': (col2, col1)
                        }

        if best_relation:
            result.update(best_relation)
            result['has_relationship'] = True

    except Exception as e:
        logger.debug(f"Error checking IND/AIND between tables: {e}")

    return result


# ================= 子查询分解和动态权重更新模块 =================

def decompose_task_with_analysis(task: str, rerank_components, table_names: List[str] = None, task_logger=None) -> List[Dict]:
    """
    🆕 一次性LLM调用完成任务分解 + 查询特征分析
    
    Args:
        task: 原始自然语言任务
        rerank_components: LLM组件，用于任务分解
        table_names: 所有可用的表名列表
        task_logger: 任务日志记录器
    
    Returns:
        List[Dict]: 分解后的子查询及其分析结果
        每个元素包含:
        {
            'index': int,
            'subquery': str,
            'has_aggregation': bool,
            'aggregation_types': List[str],
            'has_join': bool,
            'has_groupby': bool,
            'has_orderby': bool,
            'comparison_ops': List[str],
            'has_time_filter': bool,
            'entity_types': List[str],
            'query_complexity': str
        }
    """
    logger.info(f"🎯 开始任务分解+分析 (一次LLM调用): {task[:100]}...")
    
    if task_logger:
        task_logger.info("🎯 开始任务分解+查询特征分析 (增强版本)")
        task_logger.info(f"📝 原始任务: {task}")
    
    try:
        # 构建增强的任务分解提示
        table_names_str = ""
        if table_names:
            table_names_str = f"\n\nAvailable table names in the database:\n{', '.join(table_names[:20])}{'...' if len(table_names) > 20 else ''}"
        
        decomposition_prompt = f"""You are a schema linking assistant for database query analysis.

Your task has TWO parts:
1. Decompose the natural language query into DISTINCT table search queries (1-5 queries)
2. For EACH search query, analyze its characteristics

=== PART 1: Task Decomposition ===
Identify KEY ENTITY TYPES and TABLE CATEGORIES needed:
- Focus on TABLE CATEGORIES, not individual tables
- Each query describes a CATEGORY/FAMILY of tables (e.g., "expression tables", "mutation tables")
- One query may match MULTIPLE tables of the same type
- Do NOT describe computation logic or sequential steps
- Do NOT include specific table names or column names

=== PART 2: Query Characteristics Analysis ===
For EACH search query, analyze:
1. **Aggregation**: Does it need SUM/AVG/COUNT/MAX/MIN? List which types.
2. **Join**: Does it need to join multiple tables?
3. **Grouping**: Does it need GROUP BY?
4. **Ordering**: Does it need ORDER BY?
5. **Filtering**: What comparison operators? (=/>/</>=/<=/LIKE/BETWEEN/IN)
6. **Time Filter**: Does it filter by date/time?
7. **Entity Types**: What entities are involved? (customer/product/order/gene/sample/mutation/etc.)
8. **Complexity**: simple/medium/complex

=== OUTPUT FORMAT (JSON ONLY) ===
Output a JSON array with 1-5 objects. Each object:
{{
  "index": <number>,
  "subquery": "<description of table category>",
  "has_aggregation": <true/false>,
  "aggregation_types": [<"SUM"/"AVG"/"COUNT"/"MAX"/"MIN">],
  "has_join": <true/false>,
  "has_groupby": <true/false>,
  "has_orderby": <true/false>,
  "comparison_ops": [<"="/">"/"<"/">="/"<="/"LIKE"/"BETWEEN"/"IN">],
  "has_time_filter": <true/false>,
  "entity_types": [<entity names>],
  "query_complexity": <"simple"/"medium"/"complex">
}}

=== EXAMPLE 1 ===
Input: "Assess whether different genetic variants affect TP53 expression levels in BRCA samples. Provide sample count and mutation types."

Output:
```json
[
  {{
    "index": 1,
    "subquery": "Gene expression tables storing mRNA/transcript abundance measurements with gene identifiers and sample IDs",
    "has_aggregation": false,
    "aggregation_types": [],
    "has_join": false,
    "has_groupby": false,
    "has_orderby": false,
    "comparison_ops": ["="],
    "has_time_filter": false,
    "entity_types": ["gene", "expression", "sample"],
    "query_complexity": "simple"
  }},
  {{
    "index": 2,
    "subquery": "Genetic variant/mutation tables containing somatic or germline mutation records, variant types, and affected genes",
    "has_aggregation": false,
    "aggregation_types": [],
    "has_join": false,
    "has_groupby": false,
    "has_orderby": false,
    "comparison_ops": ["="],
    "has_time_filter": false,
    "entity_types": ["mutation", "variant", "gene"],
    "query_complexity": "simple"
  }},
  {{
    "index": 3,
    "subquery": "Sample/clinical metadata tables with patient demographics and cohort annotations for the BRCA study",
    "has_aggregation": true,
    "aggregation_types": ["COUNT"],
    "has_join": false,
    "has_groupby": true,
    "has_orderby": false,
    "comparison_ops": ["="],
    "has_time_filter": false,
    "entity_types": ["sample", "patient", "clinical"],
    "query_complexity": "medium"
  }}
]
```

=== EXAMPLE 2 ===
Input: "Which traffic source generated the highest product revenue in H1 2017?"

Output:
```json
[
  {{
    "index": 1,
    "subquery": "Transaction/revenue tables capturing sales events with revenue amounts, timestamps, and traffic source attribution",
    "has_aggregation": true,
    "aggregation_types": ["SUM", "MAX"],
    "has_join": false,
    "has_groupby": true,
    "has_orderby": true,
    "comparison_ops": [">=", "<="],
    "has_time_filter": true,
    "entity_types": ["transaction", "revenue", "traffic_source"],
    "query_complexity": "complex"
  }},
  {{
    "index": 2,
    "subquery": "Product catalog/dimension tables storing product attributes and categories",
    "has_aggregation": false,
    "aggregation_types": [],
    "has_join": true,
    "has_groupby": false,
    "has_orderby": false,
    "comparison_ops": ["="],
    "has_time_filter": false,
    "entity_types": ["product", "category"],
    "query_complexity": "simple"
  }}
]
```

=== EXAMPLE 3 ===
Input: "What is the total number of employees?"

Output:
```json
[
  {{
    "index": 1,
    "subquery": "Employee/HR tables containing staff records and employment status",
    "has_aggregation": true,
    "aggregation_types": ["COUNT"],
    "has_join": false,
    "has_groupby": false,
    "has_orderby": false,
    "comparison_ops": [],
    "has_time_filter": false,
    "entity_types": ["employee", "staff"],
    "query_complexity": "simple"
  }}
]
```

{table_names_str}

Now analyze this query:
{task}

CRITICAL: Output ONLY the JSON array, start with "[" and end with "]". No explanations, no thinking process."""
        
        if task_logger:
            task_logger.info("🤖 调用LLM进行一次性分解+分析")
        
        # 使用GPTChat进行任务分解+分析
        model = rerank_components['model']
        
        logger.info("🤖 任务分解+分析: 正在调用LLM...")
        model.init_messages()
        response = model.get_model_response_txt(decomposition_prompt)
        logger.info("✅ 任务分解+分析: LLM调用完成")
        
        if task_logger:
            task_logger.info(f"🤖 LLM响应长度: {len(str(response))} 字符")
        
        # 解析JSON响应
        subqueries_with_analysis = _parse_decomposition_json(response, task, task_logger)
        
        if subqueries_with_analysis:
            logger.info(f"✅ 解析成功: {len(subqueries_with_analysis)} 个子查询及其分析")
            
            if task_logger:
                task_logger.info(f"✅ 任务分解+分析完成，共 {len(subqueries_with_analysis)} 个步骤:")
                for item in subqueries_with_analysis:
                    task_logger.info(f"  步骤 {item['index']}: {item['subquery'][:60]}...")
                    task_logger.info(f"    特征: 聚合={item['has_aggregation']}, "
                                   f"连接={item['has_join']}, "
                                   f"分组={item['has_groupby']}, "
                                   f"复杂度={item['query_complexity']}")
            
            return subqueries_with_analysis
        else:
            # JSON解析失败，降级到基本模式
            logger.warning("⚠️ JSON解析失败，降级到基本分解模式")
            return _fallback_to_basic_decomposition(response, task, task_logger)
        
    except Exception as e:
        logger.error(f"❌ 任务分解+分析失败: {e}")
        if task_logger:
            task_logger.error(f"❌ 任务分解+分析失败: {e}")
        
        # 降级到原始任务
        return [{
            'index': 1,
            'subquery': task,
            'has_aggregation': False,
            'aggregation_types': [],
            'has_join': False,
            'has_groupby': False,
            'has_orderby': False,
            'comparison_ops': [],
            'has_time_filter': False,
            'entity_types': [],
            'query_complexity': 'simple'
        }]


def decompose_task_into_subqueries(task: str, rerank_components, table_names: List[str] = None, task_logger=None) -> List[str]:
    """
    将复杂任务分解为恰好三个逻辑步骤（原有版本，保持向后兼容）
    
    Args:
        task: 原始自然语言任务
        rerank_components: LLM组件，用于任务分解
        table_names: 所有可用的表名列表
        task_logger: 任务日志记录器
    
    Returns:
        List[str]: 分解后的三个步骤列表
    """
    logger.info(f"🎯 开始任务分解: {task[:100]}...")
    
    if task_logger:
        task_logger.info("🎯 开始任务分解")
        task_logger.info(f"📝 原始任务: {task}")
    
    try:
        # 构建任务分解的提示 - 按照三步逻辑分解策略
        table_names_str = ""
        if table_names:
            table_names_str = f"\n\nAvailable table names in the database:\n{', '.join(table_names[:20])}{'...' if len(table_names) > 20 else ''}"
        
        decomposition_prompt = f"""You are a schema linking assistant for database query analysis.

Your task: decompose a natural language query into DISTINCT table search queries by identifying the KEY ENTITY TYPES and DATA OBJECTS involved. Each search query should describe a TABLE CATEGORY or TABLE FAMILY that stores one type of entity or measurement.

CRITICAL - Focus on TABLE CATEGORIES, not individual tables:
- Identify: What are the CORE ENTITIES? (e.g., customers, products, transactions, genes, patients, mutations)
- Identify: What MEASUREMENTS or ATTRIBUTES are needed? (e.g., revenue, expression levels, counts)
- Identify: What RELATIONSHIPS connect entities? (e.g., patient-sample links, product-category mappings)
- Each query should describe a CATEGORY/FAMILY of tables (e.g., "expression tables", "mutation tables", "metadata tables")
- One query may match MULTIPLE tables of the same type (e.g., RNASEQ_GENE, RNASEQ_ISOFORM both belong to "expression tables")
- Do NOT describe computation logic or sequential steps (avoid "filter then aggregate", "identify then calculate")
- Do NOT include specific table names, column names, or SQL syntax

OUTPUT REQUIREMENTS (STRICTLY FOLLOW):
- Generate EXACTLY 1-5 search queries ONLY, one per DISTINCT table category
- Output ONLY the numbered list of table categories, NO explanations, NO thinking process
- Each line must be a single, complete description of a table category
- Do NOT output reasoning, analysis, or step-by-step thinking

Example Input (entity-rich):
"Assess whether different genetic variants affect TP53 expression levels in BRCA samples using mutation data. Provide sample count, mutation types, and F-statistic."

Example Output (3 table categories):
1. Gene expression tables storing mRNA/transcript abundance measurements with gene identifiers and sample IDs (may include multiple expression platforms/formats).
2. Genetic variant/mutation tables containing somatic or germline mutation records, variant types, and affected genes per sample.
3. Sample/clinical metadata tables with patient demographics, tissue types, and cohort annotations for the BRCA study.

Example Input (business analytics):
"Which traffic source generated the highest product revenue in H1 2017, and what were the max daily/weekly/monthly revenues?"

Example Output (2 table categories):
1. Transaction/revenue tables capturing sales events with product IDs, revenue amounts, timestamps, and traffic source attribution (may span multiple time granularities).
2. Product catalog/dimension tables storing product attributes, categories, and SKU information.

Example Input (simple):
"What is the total number of employees?"

Example Output (1 table category):
1. Employee/HR tables containing staff records, employment status, and headcount information.

{table_names_str}

Now, identify the KEY ENTITY TYPES in this query and generate search queries (1-5) describing TABLE CATEGORIES that store each entity type:
{task}

CRITICAL OUTPUT FORMAT:
- Output EXACTLY 1-5 lines ONLY
- Each line: a numbered description of ONE table category
- Format: "1. [description]", "2. [description]", etc.
- NO thinking process, NO explanations, NO additional text
- Start directly with "1. " on the first line"""
        
        if task_logger:
            task_logger.info("🤖 调用LLM进行任务分解")
            task_logger.info(f"📋 LLM输入提示: {decomposition_prompt[:300]}...")
        
        # 使用GPTChat进行任务分解
        model = rerank_components['model']
        
        # 初始化GPTChat对话
        logger.info("🤖 任务分解: 正在调用LLM进行任务分解...")
        model.init_messages()
        response = model.get_model_response_txt(decomposition_prompt)
        logger.info("✅ 任务分解: LLM调用完成")
        
        if task_logger:
            # 过滤掉 <think> 标签后再记录
            response_filtered = remove_think_tags(str(response))
            task_logger.info(f"🤖 LLM响应: {response_filtered}")
        
        # 解析响应 - 灵活处理1-5个分解步骤
        subqueries = []
        
        # 🚀 过滤掉thinking标签内容
        response_text = str(response).strip()
        # 移除 <think>...</think> 或 <thinking>...</thinking> 标签内容
        response_text = re.sub(r'<think>.*?</think>', '', response_text, flags=re.DOTALL)
        response_text = re.sub(r'<thinking>.*?</thinking>', '', response_text, flags=re.DOTALL)
        
        # 🚀 过滤掉明显的推理过程（常见开头词）
        thinking_indicators = [
            'okay', 'let\'s', 'first', 'so', 'the user', 'i need', 'looking at',
            'hmm', 'wait', 'maybe', 'alternatively', 'however'
        ]
        
        for line in response_text.split('\n'):
            line = line.strip()
            if not line or line.startswith('#') or line.startswith('Example') or line.startswith('Output'):
                continue
            
            # 检查是否是编号行
            if not re.match(r'^\d+\.', line):
                continue
            
            # 移除编号前缀 (1., 2., etc.)
            clean_line = re.sub(r'^\d+\.\s*', '', line)
            # 移除可能的markdown格式
            clean_line = re.sub(r'\*\*(.*?)\*\*', r'\1', clean_line)
            clean_line = clean_line.strip()
            
            # 🚀 过滤掉推理过程：检查开头是否是thinking indicators
            if clean_line and len(clean_line) > 10:
                # 检查前15个字符（小写）是否包含推理指示词
                line_start = clean_line[:30].lower()
                is_thinking = any(indicator in line_start for indicator in thinking_indicators)
                
                if not is_thinking:
                    # 这是一个有效的表类别描述
                    subqueries.append(clean_line)
                else:
                    # 这是推理过程，跳过
                    logger.debug(f"⚠️ 过滤掉推理步骤: {clean_line[:50]}...")
        
        # 自适应步骤数量：1-5个，过多则截断，过少则使用原任务兜底
        if len(subqueries) == 0:
            logger.warning(f"未解析到任何步骤，使用原任务作为单一搜索查询")
            subqueries = [task]
        elif len(subqueries) > 5:
            logger.warning(f"得到 {len(subqueries)} 个步骤（超过5个），截断为前5个")
            subqueries = subqueries[:5]
        # 1-5个步骤都是合理的，不做调整
        
        logger.info(f"✅ 任务分解完成，共 {len(subqueries)} 个步骤:")
        for i, step in enumerate(subqueries, 1):
            logger.info(f"  步骤{i}: {step}")
        
        if task_logger:
            task_logger.info(f"✅ 任务分解完成，共 {len(subqueries)} 个步骤:")
            for i, step in enumerate(subqueries, 1):
                task_logger.info(f"  步骤{i}: {step}")
        
        return subqueries
        
    except Exception as e:
        logger.error(f"任务分解失败: {e}")
        return [task]  # 回退到原始任务


def _parse_decomposition_json(response, task: str, task_logger=None) -> List[Dict]:
    """
    解析LLM返回的JSON格式分解结果
    
    Args:
        response: LLM响应
        task: 原始任务（用于降级）
        task_logger: 日志记录器
    
    Returns:
        List[Dict]: 解析后的子查询及分析，如果失败返回None
    """
    try:
        response_text = str(response).strip()
        
        # 移除可能的thinking标签
        response_text = re.sub(r'<think>.*?</think>', '', response_text, flags=re.DOTALL)
        response_text = re.sub(r'<thinking>.*?</thinking>', '', response_text, flags=re.DOTALL)
        
        # 尝试提取JSON（可能被包裹在markdown代码块中）
        json_match = re.search(r'```json\s*(.*?)\s*```', response_text, re.DOTALL)
        if json_match:
            json_str = json_match.group(1)
            logger.debug("✅ 从markdown代码块中提取JSON")
        else:
            # 尝试直接找到JSON数组
            json_match = re.search(r'\[\s*\{.*?\}\s*\]', response_text, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
                logger.debug("✅ 从响应中提取JSON数组")
            else:
                # 尝试整个响应
                json_str = response_text
                logger.debug("⚠️ 使用整个响应作为JSON")
        
        # 解析JSON
        import json
        subqueries_with_analysis = json.loads(json_str)
        
        # 验证格式
        if not isinstance(subqueries_with_analysis, list):
            logger.warning("⚠️ JSON不是数组格式")
            return None
        
        if len(subqueries_with_analysis) == 0:
            logger.warning("⚠️ JSON数组为空")
            return None
        
        # 验证每个元素的必需字段
        required_fields = ['index', 'subquery']
        for item in subqueries_with_analysis:
            if not isinstance(item, dict):
                logger.warning(f"⚠️ 数组元素不是对象: {type(item)}")
                return None
            
            for field in required_fields:
                if field not in item:
                    logger.warning(f"⚠️ 缺少必需字段: {field}")
                    return None
            
            # 确保所有分析字段存在（如果缺失则使用默认值）
            item.setdefault('has_aggregation', False)
            item.setdefault('aggregation_types', [])
            item.setdefault('has_join', False)
            item.setdefault('has_groupby', False)
            item.setdefault('has_orderby', False)
            item.setdefault('comparison_ops', [])
            item.setdefault('has_time_filter', False)
            item.setdefault('entity_types', [])
            item.setdefault('query_complexity', 'simple')
        
        logger.info(f"✅ JSON解析成功: {len(subqueries_with_analysis)} 个子查询")
        return subqueries_with_analysis
        
    except json.JSONDecodeError as e:
        logger.warning(f"⚠️ JSON解析失败: {e}")
        if task_logger:
            task_logger.warning(f"⚠️ JSON解析失败: {e}")
        return None
    except Exception as e:
        logger.error(f"❌ 解析过程出错: {e}")
        if task_logger:
            task_logger.error(f"❌ 解析过程出错: {e}")
        return None


def _fallback_to_basic_decomposition(response, task: str, task_logger=None) -> List[Dict]:
    """
    降级策略：如果JSON解析失败，提取文本子查询并使用空特征
    
    Args:
        response: LLM响应
        task: 原始任务
        task_logger: 日志记录器
    
    Returns:
        List[Dict]: 基本的子查询列表（带空特征）
    """
    logger.warning("⚠️ 使用降级策略：提取文本子查询 + 空特征")
    if task_logger:
        task_logger.warning("⚠️ 使用降级策略：提取文本子查询 + 空特征")
    
    subqueries = []
    response_text = str(response).strip()
    
    # 移除thinking标签
    response_text = re.sub(r'<think>.*?</think>', '', response_text, flags=re.DOTALL)
    response_text = re.sub(r'<thinking>.*?</thinking>', '', response_text, flags=re.DOTALL)
    
    # 尝试提取编号列表
    for line in response_text.split('\n'):
        line = line.strip()
        if not line:
            continue
        
        # 检查是否是编号行
        match = re.match(r'^\s*(\d+)\.\s*(.+)$', line)
        if match:
            index = int(match.group(1))
            subquery = match.group(2).strip()
            
            # 移除可能的markdown格式
            subquery = re.sub(r'\*\*(.*?)\*\*', r'\1', subquery)
            subquery = re.sub(r'```.*?```', '', subquery, flags=re.DOTALL)
            
            if len(subquery) > 10:  # 过滤太短的行
                subqueries.append({
                    'index': index,
                    'subquery': subquery,
                    'has_aggregation': False,
                    'aggregation_types': [],
                    'has_join': False,
                    'has_groupby': False,
                    'has_orderby': False,
                    'comparison_ops': [],
                    'has_time_filter': False,
                    'entity_types': [],
                    'query_complexity': 'simple'
                })
    
    # 如果没有提取到任何子查询，使用原始任务
    if len(subqueries) == 0:
        logger.warning("⚠️ 降级策略也失败，使用原始任务")
        if task_logger:
            task_logger.warning("⚠️ 降级策略也失败，使用原始任务")
        
        subqueries = [{
            'index': 1,
            'subquery': task,
            'has_aggregation': False,
            'aggregation_types': [],
            'has_join': False,
            'has_groupby': False,
            'has_orderby': False,
            'comparison_ops': [],
            'has_time_filter': False,
            'entity_types': [],
            'query_complexity': 'simple'
        }]
    
    logger.info(f"✅ 降级策略提取了 {len(subqueries)} 个子查询")
    if task_logger:
        task_logger.info(f"✅ 降级策略提取了 {len(subqueries)} 个子查询")
    
    return subqueries


def _identify_relevant_databases(working_graphs: Dict, current_table_set=None, example_root=None, current_example_id=None):
    """
    智能识别当前任务相关的数据库
    
    由于现在每个任务只加载相关的数据库图，这个函数已简化
    
    Args:
        working_graphs: 工作图字典（已经是任务特定的）
        current_table_set: 当前表集合
        example_root: 样本根目录
        current_example_id: 当前样本ID
    
    Returns:
        set: 相关数据库名称集合
    """
    # 🎯 优化：由于working_graphs已经是任务特定的，直接返回所有数据库
    # 这确保了任务隔离，每个任务只操作自己的数据库图
    relevant_databases = set(working_graphs.keys())
    
    logger.info(f"🎯 任务隔离模式：使用所有已加载的数据库图 {list(relevant_databases)}")
    
    return relevant_databases


def _load_precomputed_table_embeddings(working_graphs: Dict, example_root: str, task_id: str = None):
    """
    加载预计算的表embedding以提高搜索效率
    
    Args:
        working_graphs: 工作图字典
        example_root: 样本根目录路径
        task_id: 任务ID，用于定位正确的embedding文件
    """
    logger.info("🚀 加载预计算的表embedding...")
    total_tables = 0
    loaded_tables = 0
    
    for db_name, graph in working_graphs.items():
        if not hasattr(graph, '_table_embeddings_cache'):
            graph._table_embeddings_cache = {}
        
        cache = graph._table_embeddings_cache
        
        # 尝试加载对应数据库的embedding文件
        embedding_file = None
        if example_root:
            # 尝试从样本目录加载，优先使用task_id路径
            potential_paths = []
            if task_id:
                potential_paths.append(f"{example_root}/{task_id}/table_embeddings.npz")
            potential_paths.extend([
                f"{example_root}/{db_name}/table_embeddings.npz",
                f"{example_root}/table_embeddings.npz"
            ])
            for path in potential_paths:
                if os.path.exists(path):
                    embedding_file = path
                    break
        
        if embedding_file:
            try:
                embeddings_data = np.load(embedding_file)
                table_names = embeddings_data['table_names']
                embeddings = embeddings_data['embeddings']
                
                # 将embedding加载到缓存中
                for i, table_name in enumerate(table_names):
                    if table_name in graph.nodes():
                        cache[table_name] = embeddings[i]
                        loaded_tables += 1
                        total_tables += 1
                
                logger.info(f"✅ 从 {embedding_file} 加载了 {len(table_names)} 个表的embedding")
            except Exception as e:
                logger.warning(f"加载embedding文件失败 {embedding_file}: {e}")
        else:
            # 如果没有预计算的embedding文件，标记需要重新计算
            for node in graph.nodes():
                total_tables += 1
                cache[node] = None  # 标记为需要计算
    
    logger.info(f"✅ 表embedding加载完成: {loaded_tables}/{total_tables} 个表已加载")


def update_graph_weights_for_subquery_optimized(graph: nx.Graph, current_subquery: str, 
                                               table_descriptions: Dict, embedding_model=None,
                                               precomputed_subquery_embedding=None,
                                               node_task_similarities: Dict = None,
                                               alpha: float = 0.3, beta: float = 50.0,
                                               conditional_manager=None,
                                               query_context=None,
                                               edge_workload_weight: float = 0.1,
                                               node_workload_weight: float = 0.1,
                                               workload_weight: float = 0.1) -> nx.Graph:
    """
    基于当前子查询动态更新图中边的权重
    🆕 支持 workload-aware 权重调整
    
    改进策略：新权重 = α×w_orig + β×sim(起始节点, 总任务) * sim(候选节点, 子查询) + γ×workload_boost
    - α控制原始MinHash权重的保留比例（默认0.3，降低拓扑结构影响）
    - β控制语义相似度的放大系数（默认50.0，将0-1范围放大到与MinHash可比）
    - γ控制 workload boost 的权重（默认0.1）
    - 起始节点（已选节点）与总任务的相似度：保持全局一致性
    - 候选节点（待扩展节点）与子查询的相似度：聚焦当前步骤
    
    Args:
        graph: NetworkX图对象
        current_subquery: 当前处理的子查询
        table_descriptions: 表描述字典
        embedding_model: 嵌入模型（可选）
        precomputed_subquery_embedding: 预计算的子查询embedding（可选）
        node_task_similarities: 预计算的节点与总任务的相似度字典（在Top-K预选阶段已计算）
        alpha: MinHash权重系数（默认0.3）
        beta: 语义相似度放大系数（默认50.0）
        conditional_manager: ConditionalFunctionManager 实例（可选，用于 workload boost）
        query_context: QueryContext 实例（可选，用于 workload boost）
        workload_weight: workload boost 权重系数（默认0.1）
    
    Returns:
        更新权重后的图对象
    """
    logger.info(f"🔄 更新图权重，基于子查询: {current_subquery[:50]}...")
    
    try:
        # 如果没有嵌入模型，使用简单的文本匹配
        if embedding_model is None:
            logger.info("使用文本匹配更新权重")
            return _update_weights_by_text_matching(graph, current_subquery, table_descriptions, 
                                                   alpha=alpha, beta=beta)
        
        # 使用嵌入模型计算语义相似度
        logger.info("使用嵌入模型更新权重")
        
        # 使用预计算的子查询embedding，如果没有则重新计算
        if precomputed_subquery_embedding is not None:
            subquery_embedding = precomputed_subquery_embedding
            logger.info("✅ 权重更新: 使用预计算的子查询embedding")
        else:
            logger.info("🤖 权重更新: 正在计算子查询embedding...")
            subquery_embedding = embedding_model.encode(current_subquery, normalize_embeddings=True)
            logger.info("✅ 权重更新: 子查询embedding计算完成")
        
        # 🎯 计算节点与子查询的相似度
        node_subquery_similarities = {}
        
        # 🎯 使用预计算的节点与总任务的相似度
        if node_task_similarities is None:
            # 如果没有提供预计算的相似度，回退到旧方法（两个节点都用子查询）
            logger.warning("⚠️ 没有提供节点-总任务预计算相似度，回退到原有方法（两个节点都用子查询相似度）")
            node_task_sims = None  # 稍后会用子查询相似度替代
        else:
            # 使用预计算的总任务相似度（在函数外部已经计算好）
            logger.info(f"✅ 使用预计算的节点-总任务相似度 ({len(node_task_similarities)} 个节点)")
            node_task_sims = node_task_similarities
        
        # 检查是否有预计算的表embedding缓存
        if hasattr(graph, '_table_embeddings_cache'):
            table_embeddings_cache = graph._table_embeddings_cache
            logger.debug("使用缓存的表embedding")
        else:
            # 第一次计算时创建缓存
            table_embeddings_cache = {}
            logger.debug("创建表embedding缓存")
        
        # 遍历所有节点，只计算与子查询的相似度（总任务相似度已预计算）
        for node in graph.nodes():
            try:
                # 尝试从缓存获取表embedding
                if node in table_embeddings_cache and table_embeddings_cache[node] is not None:
                    table_embedding = table_embeddings_cache[node]
                    logger.debug(f"  ✅ 使用缓存的表embedding: {node}")
                else:
                    # 🚨 权重更新时不应该重新计算表embedding！
                    logger.warning(f"⚠️ 权重更新: 表 {node} 没有预计算embedding，跳过相似度计算")
                    node_subquery_similarities[node] = 0.0
                    continue
                
                # 计算与子查询的相似度（用于候选节点）
                subquery_sim = np.dot(subquery_embedding, table_embedding)
                node_subquery_similarities[node] = float(subquery_sim)
                logger.debug(f"  表 {node} 与子查询相似度: {subquery_sim:.3f}")
                
            except Exception as e:
                logger.debug(f"  计算表 {node} 相似度失败: {e}")
                node_subquery_similarities[node] = 0.0
        
        # 如果没有预计算的总任务相似度，回退到使用子查询相似度
        if node_task_sims is None:
            logger.info("⚠️ 回退方案：使用子查询相似度替代总任务相似度")
            node_task_sims = node_subquery_similarities.copy()
        
        # 保存缓存到图对象
        graph._table_embeddings_cache = table_embeddings_cache
        
        # 🚀 优化：禁用列级共现计算（性能瓶颈，收益有限）
        # 原因：Column Co-occurrence 计算代价极高（O(n²)），但 Table Co-occurrence 已捕捉大部分信息
        # table_to_columns = {}
        # if conditional_manager and query_context:
        #     all_nodes = list(graph.nodes())
        #     table_to_columns = build_table_to_columns_map(graph, all_nodes)
        #     logger.info(f"📋 提取列信息用于边增强: {len([t for t, cols in table_to_columns.items() if cols])}/{len(all_nodes)} 个表有列数据")
        
        # 🚀 更新边权重（加权组合版 + 🆕 workload boost）
        updated_edges = 0
        semantic_weights = []  # 收集语义权重用于统计
        workload_boosts = []  # 🆕 收集 workload boost 用于统计
        
        for u, v, edge_data in graph.edges(data=True):
            # 保存原始权重（如果还没保存）
            if 'original_weight' not in edge_data:
                edge_data['original_weight'] = edge_data.get('weight', 1.0)
            
            # 🎯 加权组合策略：新权重 = α×w_orig + β×sim(节点u, 总任务) * sim(节点v, 子查询)
            # α=0.3: 保留30%的MinHash拓扑信息（降低密集连接子图的优势）
            # β=50.0: 将语义相似度放大到与MinHash可比的量级（原MinHash~40，相似度0-1）
            # 
            # 注意：由于边是无向的，我们取平均值来处理u和v的对称性
            # 方法1: u作为起始节点，v作为候选节点
            similarity_product_1 = node_task_sims.get(u, 0.0) * node_subquery_similarities.get(v, 0.0)
            # 方法2: v作为起始节点，u作为候选节点  
            similarity_product_2 = node_task_sims.get(v, 0.0) * node_subquery_similarities.get(u, 0.0)
            # 取平均值来处理边的无向性
            similarity_product = (similarity_product_1 + similarity_product_2) / 2.0
            
            # 🔥 新权重公式：加权组合
            minhash_component = alpha * edge_data['original_weight']
            semantic_component = beta * similarity_product
            
            # 🆕 workload boost 组件（仅表级）
            # 🔧 架构修正：仅 Join + Table Cooccurrence 注入边权（离线 Enhanced Graph）
            #             Predicate + Aggregation 已移至 personalization（在线 Query Graph）
            # 🚀 优化：禁用列级共现计算（Column Co-occurrence），仅保留表级共现
            workload_component = 0.0
            if conditional_manager and query_context:
                # 表级 workload boost（Join + Table Cooccurrence）
                table_boost = conditional_manager.compute_edge_boost(u, v, query_context)
                
                if table_boost > 0:
                    # 🔧 应用权重系数（不再乘以 beta，防止尺度失控）
                    # table_boost 已经是 [0, ~1]（p95 归一化），与 semantic component 可比
                    workload_component = edge_workload_weight * table_boost  # 🔬 λ: 边增强权重
                    edge_data['workload_boost_raw'] = table_boost
                    edge_data['table_boost'] = table_boost
                    workload_boosts.append(workload_component)
            
            new_weight = minhash_component + semantic_component + workload_component
            
            edge_data['weight'] = new_weight
            edge_data['minhash_component'] = minhash_component
            edge_data['semantic_component'] = semantic_component
            edge_data['workload_component'] = workload_component  # 🆕
            edge_data['similarity_product'] = similarity_product
            edge_data['sim_prod_1'] = similarity_product_1  # 保存详细信息用于调试
            edge_data['sim_prod_2'] = similarity_product_2
            
            semantic_weights.append(semantic_component)
            updated_edges += 1
            
            logger.debug(f"  边 {u}-{v}:")
            logger.debug(f"    原MinHash权重: {edge_data['original_weight']:.3f}")
            logger.debug(f"    MinHash成分: {alpha}×{edge_data['original_weight']:.3f} = {minhash_component:.3f}")
            logger.debug(f"    语义成分: {beta}×{similarity_product:.4f} = {semantic_component:.3f}")
            if workload_component > 0:
                logger.debug(f"    🆕 Workload成分: {workload_weight}×{edge_data.get('workload_boost_raw', 0):.4f} = {workload_component:.3f}")
                if 'table_boost' in edge_data:
                    logger.debug(f"      - 表级boost (Join + Table Cooccur): {edge_data['table_boost']:.4f}")
            logger.debug(f"    新权重: {new_weight:.3f}")
        
        # 统计语义权重分布
        if semantic_weights:
            avg_semantic = np.mean(semantic_weights)
            max_semantic = np.max(semantic_weights)
            min_semantic = np.min(semantic_weights)
            logger.info(f"📊 语义权重统计: min={min_semantic:.3f}, max={max_semantic:.3f}, avg={avg_semantic:.3f}")
        
        # 🆕 统计 workload boost 分布
        if workload_boosts:
            avg_workload = np.mean(workload_boosts)
            max_workload = np.max(workload_boosts)
            min_workload = np.min(workload_boosts)
            logger.info(f"✨ Workload权重统计: {len(workload_boosts)} 条边有boost, "
                       f"min={min_workload:.3f}, max={max_workload:.3f}, avg={avg_workload:.3f}")
        
        logger.info(f"✅ 权重更新完成，更新了 {updated_edges} 条边")
        if conditional_manager and query_context:
            logger.info(f"📊 权重公式: w' = {alpha}×w_minhash + {beta}×sim_product + {workload_weight}×workload_boost (p95-normalized)")
        else:
            logger.info(f"📊 权重公式: w' = {alpha}×w_minhash + {beta}×sim_product")
        logger.info(f"🎯 策略: 降低MinHash影响({alpha*100:.0f}%)，强化语义相关性(×{beta})")
        return graph
    
    except Exception as e:
        logger.error(f"权重更新失败: {e}")
        return graph


def _update_weights_by_text_matching(graph: nx.Graph, subquery: str, table_descriptions: Dict, 
                                    alpha: float = 0.3, beta: float = 50.0) -> nx.Graph:
    """
    使用简单文本匹配更新权重的回退方法
    
    改进策略：新权重 = α×w_orig + β×score_product
    - α控制原始MinHash权重的保留比例（默认0.3）
    - β控制文本匹配分数的放大系数（默认50.0）
    """
    # 提取子查询中的关键词
    subquery_lower = subquery.lower()
    keywords = re.findall(r'\b\w+\b', subquery_lower)
    keywords = [kw for kw in keywords if len(kw) > 2]  # 过滤短词
    
    # 为每个节点计算匹配分数
    node_scores = {}
    for node in graph.nodes():
        score = 0.0
        node_text = _build_table_text_representation(node, table_descriptions).lower()
        
        for keyword in keywords:
            if keyword in node_text:
                score += 1.0
        
        # 归一化
        if keywords:
            score = score / len(keywords)
        node_scores[node] = score
    
    # 更新边权重（使用加权组合策略）
    updated_edges = 0
    text_match_weights = []
    
    for u, v, edge_data in graph.edges(data=True):
        if 'original_weight' not in edge_data:
            edge_data['original_weight'] = edge_data.get('weight', 1.0)
        
        # 🔥 新权重公式：加权组合（与embedding版本一致）
        u_score = node_scores.get(u, 0.0)
        v_score = node_scores.get(v, 0.0)
        score_product = u_score * v_score
        
        minhash_component = alpha * edge_data['original_weight']
        text_match_component = beta * score_product
        new_weight = minhash_component + text_match_component
        
        edge_data['weight'] = new_weight
        edge_data['minhash_component'] = minhash_component
        edge_data['text_match_component'] = text_match_component
        edge_data['text_match_score'] = score_product
        
        text_match_weights.append(text_match_component)
        updated_edges += 1
        
        logger.debug(f"  边 {u}-{v}:")
        logger.debug(f"    MinHash成分: {alpha}×{edge_data['original_weight']:.3f} = {minhash_component:.3f}")
        logger.debug(f"    文本匹配成分: {beta}×{score_product:.4f} = {text_match_component:.3f}")
        logger.debug(f"    新权重: {new_weight:.3f}")
    
    # 统计文本匹配权重分布
    if text_match_weights:
        avg_text = np.mean(text_match_weights)
        max_text = np.max(text_match_weights)
        min_text = np.min(text_match_weights)
        logger.info(f"📊 文本匹配权重统计: min={min_text:.3f}, max={max_text:.3f}, avg={avg_text:.3f}")
    
    logger.info(f"✅ 文本匹配权重更新完成，更新了 {updated_edges} 条边")
    
    return graph


def _build_table_text_representation(table_name: str, table_descriptions: Dict, 
                                     example_root=None, current_example_id=None, use_description=True) -> str:
    """构建表的文本表示，包含完整的列schema信息"""
    # 优先使用完整的schema信息（包含列信息）
    if example_root and current_example_id:
        complete_schema = get_complete_neighbor_schema(
            table_name, table_descriptions, use_description, 
            example_root, current_example_id
        )
        if complete_schema:
            return complete_schema
    
    # 回退到基本信息
    text_parts = [f"Table full name: {table_name}"]
    
    # 添加表描述
    if use_description and table_name in table_descriptions and table_descriptions[table_name]:
        text_parts.append(f"\nTable Description: {table_descriptions[table_name]}")
    
    return "\n".join(text_parts)


def select_best_start_node_for_subquery(subquery: str, graph: nx.Graph, 
                                       table_descriptions: Dict, embedding_model=None, top_k: int = 5, 
                                       precomputed_subquery_embedding=None) -> List[str]:
    """
    为子查询选择最佳起始节点列表，按相似度从高到低排序
    
    Args:
        subquery: 当前子查询
        graph: NetworkX图对象
        table_descriptions: 表描述字典
        embedding_model: 嵌入模型（可选）
        top_k: 返回的最大节点数
        precomputed_subquery_embedding: 预计算的子查询embedding（可选）
    
    Returns:
        按相似度排序的起始节点列表
    """
    logger.info(f"🎯 为子查询选择起始节点: {subquery[:50]}...")
    
    try:
        node_scores = []
        
        if embedding_model is None:
            # 使用文本匹配
            subquery_lower = subquery.lower()
            keywords = re.findall(r'\b\w+\b', subquery_lower)
            keywords = [kw for kw in keywords if len(kw) > 2]
            
            for node in graph.nodes():
                node_text = _build_table_text_representation(node, table_descriptions).lower()
                score = sum(1 for kw in keywords if kw in node_text)
                if keywords:
                    score = score / len(keywords)
                node_scores.append((node, score))
        else:
            # 使用嵌入模型
            if precomputed_subquery_embedding is not None:
                subquery_embedding = precomputed_subquery_embedding
                logger.info("✅ 起始节点选择: 使用预计算的子查询embedding")
            else:
                logger.info("🤖 起始节点选择: 正在计算子查询embedding...")
                subquery_embedding = embedding_model.encode(subquery, normalize_embeddings=True)
                logger.info("✅ 起始节点选择: 子查询embedding计算完成")
            
            for node in graph.nodes():
                node_text = _build_table_text_representation(node, table_descriptions)
                try:
                    node_embedding = embedding_model.encode(node_text, normalize_embeddings=True)
                    score = float(np.dot(subquery_embedding, node_embedding))
                    node_scores.append((node, score))
                except Exception as e:
                    logger.debug(f"计算节点 {node} 相似度失败: {e}")
                    node_scores.append((node, 0.0))
        
        if not node_scores:
            # 回退：选择度数最高的节点
            best_node = max(graph.nodes(), key=lambda n: graph.degree(n))
            return [best_node]
        
        # 按相似度排序，选择top_k
        node_scores.sort(key=lambda x: x[1], reverse=True)
        top_nodes = [node for node, score in node_scores[:top_k]]
        
        logger.info(f"✅ 选择了 {len(top_nodes)} 个起始节点候选")
        for i, (node, score) in enumerate(node_scores[:top_k]):
            logger.info(f"  {i+1}. {node} (分数: {score:.3f})")
        
        return top_nodes
    
    except Exception as e:
        logger.error(f"选择起始节点失败: {e}")
        # 回退：选择第一个节点
        first_node = list(graph.nodes())[0] if graph.nodes() else None
        return [first_node] if first_node else []


def check_table_connectivity(graph: nx.Graph, selected_tables: List[str]) -> Dict:
    """
    检查选中表的连通性
    
    Args:
        graph: 数据库图
        selected_tables: 选中的表列表
    
    Returns:
        连通性分析结果
    """
    if not selected_tables:
        return {'is_connected': True, 'components': [], 'isolated_tables': [], 'quality_score': 1.0}
    
    # 创建选中表的子图
    try:
        subgraph = graph.subgraph(selected_tables)
    except Exception as e:
        logger.debug(f"创建子图失败: {e}")
        return {'is_connected': False, 'components': [], 'isolated_tables': selected_tables, 'quality_score': 0.0}
    
    # 分析连通分量
    connected_components = list(nx.connected_components(subgraph))
    
    # 检查是否全连通
    is_fully_connected = len(connected_components) == 1
    
    # 找出孤立的表（度为0的节点）
    isolated_tables = [table for table in selected_tables if subgraph.degree(table) == 0]
    
    # 计算连通性质量得分
    quality_score = 0.0
    if selected_tables:
        if is_fully_connected:
            quality_score = 1.0
        else:
            # 部分分数：最大连通分量占比
            largest_component_size = max(len(comp) for comp in connected_components) if connected_components else 0
            quality_score = largest_component_size / len(selected_tables)
    
    result = {
        'is_connected': is_fully_connected,
        'num_components': len(connected_components),
        'components': [list(component) for component in connected_components],
        'isolated_tables': isolated_tables,
        'largest_component_size': max(len(comp) for comp in connected_components) if connected_components else 0,
        'quality_score': quality_score
    }
    
    logger.debug(f"连通性检查: {len(selected_tables)} 个表, {len(connected_components)} 个分量, 质量得分: {quality_score:.2f}")
    
    return result


def validate_and_refine_schema_with_sql(task, selected_tables, rerank_components,
                                       database_graphs, table_descriptions, use_description=True,
                                       all_embeddings=None, current_table_set=None,
                                       expansion_search_count=None, example_root=None,
                                       current_example_id=None, embedding_model=None,
                                       task_logger=None, max_iterations=5,
                                       sqlite_path=None, db_engine=None,
                                       enable_batch_rerank=False, batch_size=10,
                                       enable_graph_topology=True):
    """
    Step 3: Schema验证与SQL迭代优化

    根据选取的schema，让LLM判断是否满足任务需求，生成SQL验证，如果失败则迭代优化

    Returns:
        List[Dict]: 优化后的选中表列表
    """
    from sql import SqlEnv

    # 🆕 如果没有指定 db_engine，尝试从 current_example_id 推断
    if db_engine is None and current_example_id:
        db_engine = get_api_name(current_example_id)
        if task_logger:
            task_logger.info(f"🔧 自动检测数据库引擎: {db_engine}")
    elif db_engine is None:
        db_engine = "sqlite"  # 默认使用 sqlite
        if task_logger:
            task_logger.info(f"⚠️ 无法检测数据库引擎，使用默认值: {db_engine}")

    if task_logger:
        task_logger.info(f"🔍 开始Schema验证与SQL迭代优化（最多{max_iterations}次迭代）")
        task_logger.info(f"📊 当前选中表数量: {len(selected_tables)}")
        task_logger.info(f"💾 数据库引擎: {db_engine}")

    logger.info(f"🔍 Step 3: Schema验证与SQL迭代优化")
    logger.info(f"📊 当前选中 {len(selected_tables)} 个表")
    logger.info(f"💾 数据库引擎: {db_engine}")

    # 初始化SQL执行环境
    sql_env = SqlEnv()
    model = rerank_components['model']

    last_sql = None
    last_sql_error = None
    last_sql_feedback = None  # carry SQL result/error into next iteration

    try:
        # 迭代验证和优化（统一决策：让LLM在每轮选择动作：生成SQL / 生成子任务 / 直接完成）
        for iteration in range(1, max_iterations + 1):
            if task_logger:
                task_logger.info("=" * 80)
                task_logger.info(f"🔄 第 {iteration}/{max_iterations} 次迭代")
                task_logger.info("=" * 80)

            logger.info(f"🔄 迭代 {iteration}/{max_iterations}")

            # Step 3.1: 构建当前schema摘要
            if task_logger:
                task_logger.info(f"📋 当前Schema包含 {len(selected_tables)} 个表")
                table_names = [t.get('table name', 'unknown') for t in selected_tables]
                task_logger.info(f"📝 表列表: {table_names}")
                task_logger.info(f"🔍 开始构建Schema摘要...")

            schema_summary = _build_schema_summary_for_validation(
                selected_tables, table_descriptions, use_description,
                example_root, current_example_id, task_logger
            )

            if task_logger:
                missing_schema_count = schema_summary.count('[WARNING: Column schema information not available')
                if missing_schema_count > 0:
                    task_logger.warning(f"⚠️ 警告: {missing_schema_count} 个表缺少列schema信息！")
                    task_logger.warning(f"   LLM可能需要先查询这些表来发现其列结构")
                else:
                    task_logger.info(f"✅ 所有表都有完整的列schema信息")

            # 🆕 统一决策：让LLM选择动作
            if task_logger:
                task_logger.info("=" * 80)
                task_logger.info("📥 决策器输入信息:")
                task_logger.info("=" * 80)
                task_logger.info(f"🎯 任务: {task}")
                task_logger.info(f"💾 数据库: {db_engine}")
                task_logger.info(f"🔄 迭代: {iteration}/{max_iterations}")
                task_logger.info("")
                task_logger.info("📋 当前Schema摘要:")
                task_logger.info("-" * 80)
                schema_lines = schema_summary.split('\n')
                for line in schema_lines[:50]:
                    task_logger.info(line)
                if len(schema_lines) > 50:
                    task_logger.info(f"... (剩余 {len(schema_lines) - 50} 行)")
                task_logger.info("-" * 80)
                task_logger.info("")
                task_logger.info("📜 验证历史:")
                task_logger.info(f"  上一轮SQL: {last_sql if last_sql else '无'}")
                task_logger.info(f"  上一轮错误: {last_sql_error if last_sql_error else '无'}")
                task_logger.info(f"  上一轮反馈: {last_sql_feedback if last_sql_feedback else '无'}")
                task_logger.info("=" * 80)

            action, action_content = _plan_step3_action(
                task=task,
                schema_summary=schema_summary,
                model=model,
                db_engine=db_engine,
                iteration=iteration,
                max_iterations=max_iterations,
                last_sql=last_sql,
                last_sql_error=last_sql_error,
                last_sql_feedback=last_sql_feedback,
                task_logger=task_logger
            )

            if task_logger:
                task_logger.info("")
                task_logger.info("=" * 80)
                task_logger.info("📤 决策器输出结果:")
                task_logger.info("=" * 80)
                task_logger.info(f"🤖 决策动作: {action}")
                if action_content:
                    task_logger.info(f"📄 动作内容:")
                    task_logger.info("-" * 80)
                    task_logger.info(str(action_content))
                    task_logger.info("-" * 80)
                task_logger.info("=" * 80)

            if action == "SUBTASK":
                new_subtask = action_content
                if not new_subtask or str(new_subtask).upper().startswith("NO_SUBTASK"):
                    logger.info("🤖 决策为SUBTASK但无需新增表，继续下一轮")
                    continue

                if task_logger:
                    task_logger.info("")
                    task_logger.info("=" * 80)
                    task_logger.info("🔍 子任务图搜索:")
                    task_logger.info("=" * 80)
                    task_logger.info(f"🎯 子任务描述:")
                    task_logger.info("-" * 80)
                    task_logger.info(f"{new_subtask}")
                    task_logger.info("-" * 80)
                    task_logger.info(f"📊 当前Schema包含 {len(selected_tables)} 个表")
                    task_logger.info(f"🔍 开始图搜索寻找相关表...")

                new_tables = _search_tables_for_subtask(
                    new_subtask, database_graphs, table_descriptions, rerank_components,
                    all_embeddings, embedding_model, use_description,
                    example_root, current_example_id, task_logger,
                    enable_batch_rerank=enable_batch_rerank, batch_size=batch_size,
                    conditional_manager=None, query_context=None, workload_weight=0.1,
                    edge_workload_weight=0.0, node_workload_weight=0.0,  # 🔬 这里未启用 workload
                    enable_graph_topology=enable_graph_topology
                )

                if task_logger:
                    task_logger.info("")
                    task_logger.info("📊 图搜索结果:")
                    task_logger.info("-" * 80)
                    task_logger.info(f"🔍 找到 {len(new_tables)} 个候选表")
                    if new_tables:
                        task_logger.info("候选表列表:")
                        for idx, table in enumerate(new_tables, 1):
                            table_name = table.get('table name', 'unknown')
                            answer = table.get('answer', 'N')
                            task_logger.info(f"  {idx}. {table_name} ({answer})")

                existing_table_names = {t.get('table name') for t in selected_tables}
                new_added_count = 0
                added_table_names = []
                for new_table in new_tables:
                    new_table_name = new_table.get('table name')
                    if new_table_name and new_table_name not in existing_table_names:
                        selected_tables.append(new_table)
                        existing_table_names.add(new_table_name)
                        added_table_names.append(new_table_name)
                        new_added_count += 1

                if task_logger:
                    task_logger.info("-" * 80)
                    task_logger.info(f"📝 合并结果:")
                    task_logger.info(f"  ➕ 新增 {new_added_count} 个表到schema中")
                    if added_table_names:
                        task_logger.info(f"  新增的表:")
                        for table_name in added_table_names:
                            task_logger.info(f"    - {table_name}")
                    task_logger.info(f"  📊 当前schema共包含 {len(selected_tables)} 个表")
                    task_logger.info("=" * 80)

                logger.info(f"➕ 新增 {new_added_count} 个表，当前共 {len(selected_tables)} 个表")
                continue

            if action == "GENERATE_SQL":
                sql_query = _extract_sql_from_response(action_content) or str(action_content).strip()
                last_sql = sql_query

                if not sql_query:
                    logger.warning("❌ 决策生成SQL但未提取到SQL，继续下一轮")
                    last_sql_error = "sql_missing"
                    last_sql_feedback = "ERROR: sql_missing"
                    continue

                if task_logger:
                    task_logger.info("")
                    task_logger.info("=" * 80)
                    task_logger.info("🔍 SQL验证执行:")
                    task_logger.info("=" * 80)
                    task_logger.info(f"📝 待验证SQL:")
                    task_logger.info("-" * 80)
                    task_logger.info(f"{sql_query}")
                    task_logger.info("-" * 80)
                    task_logger.info(f"💾 数据库引擎: {db_engine}")
                    if db_engine == "sqlite":
                        task_logger.info(f"📂 SQLite路径: {sqlite_path}")
                    elif db_engine == "snowflake":
                        task_logger.info(f"🆔 Example ID: {current_example_id}")

                try:
                    if task_logger:
                        task_logger.info("⏳ 开始执行SQL查询...")

                    if db_engine == "sqlite":
                        if sqlite_path is None:
                            logger.error("❌ SQLite数据库路径为空，无法执行SQL验证")
                            if task_logger:
                                task_logger.error("❌ SQLite数据库路径为空，跳过SQL验证")
                            break
                        exec_result = sql_env.exec_sql_sqlite(sql_query, sqlite_path=sqlite_path)
                    elif db_engine == "bigquery":
                        exec_result = sql_env.exec_sql_bq(sql_query, save_path=None, max_len=30000)
                    elif db_engine == "snowflake":
                        if current_example_id is None:
                            logger.error("❌ Snowflake数据库需要example_id，无法执行SQL验证")
                            if task_logger:
                                task_logger.error("❌ Snowflake数据库需要example_id，跳过SQL验证")
                            break
                        exec_result = sql_env.exec_sql_sf(sql_query, save_path=None, max_len=30000, ex_id=current_example_id)
                    else:
                        logger.error(f"❌ 不支持的数据库引擎类型: {db_engine}")
                        if task_logger:
                            task_logger.error(f"❌ 不支持的数据库引擎类型: {db_engine}，跳过SQL验证")
                        break
                except Exception as e:
                    exec_result = f"##ERROR## SQL执行异常: {str(e)}"
                    logger.error(f"❌ SQL执行异常: {e}")
                    if task_logger:
                        task_logger.error(f"❌ SQL执行异常: {e}")

                if task_logger:
                    task_logger.info("")
                    task_logger.info("📊 SQL执行结果:")
                    task_logger.info("-" * 80)

                if "##ERROR##" in str(exec_result):
                    error_msg = str(exec_result).replace("##ERROR##", "").strip()
                    last_sql_error = error_msg
                    last_sql_feedback = f"ERROR: {error_msg}"
                    logger.warning(f"❌ SQL执行失败: {error_msg[:200]}")
                    if task_logger:
                        task_logger.error(f"❌ SQL执行失败!")
                        task_logger.error(f"错误信息:")
                        task_logger.error(f"{error_msg}")
                        task_logger.info("-" * 80)
                        task_logger.info(f"📝 反馈信息将传递到下一轮:")
                        task_logger.info(f"  - last_sql: {sql_query[:100]}...")
                        task_logger.info(f"  - last_sql_error: {error_msg[:200]}...")
                        task_logger.info(f"  - last_sql_feedback: {last_sql_feedback[:200]}...")
                        task_logger.info("=" * 80)
                    continue  # 下一轮
                else:
                    last_sql_feedback = f"RESULT_PREVIEW: {str(exec_result)[:500]}"
                    last_sql_error = None
                    logger.info(f"✅ SQL验证成功！")

                    # ✅ 关键：成功后直接结束Step3，避免重复迭代
                    if task_logger:
                        task_logger.info("✅ 已成功验证SQL，可判定当前schema可用，结束迭代")
                    return selected_tables

            if action in ("FINISH_SUFFICIENT", "FINISH_INSUFFICIENT"):
                logger.info(f"✅ 决策完成: {action}")
                if task_logger:
                    task_logger.info(f"✅ 决策完成: {action}")
                return selected_tables

            logger.warning(f"⚠️ 未知决策动作 {action}，返回当前schema")
            return selected_tables

        logger.info(f"🎉 Schema验证与优化完成（达到最大迭代次数），最终选中 {len(selected_tables)} 个表")
        return selected_tables

    finally:
        # 清理数据库连接（确保 early return 也能关闭）
        try:
            sql_env.close_db()
        except Exception:
            pass


def _build_schema_summary_for_validation(selected_tables, table_descriptions, use_description,
                                         example_root, current_example_id, task_logger=None):
    """构建用于验证的schema摘要"""
    schema_lines = []
    
    for table_info in selected_tables:
        table_name = table_info.get('table name', 'unknown')
        
        # 获取完整的表schema信息
        complete_schema = get_complete_neighbor_schema(
            table_name, table_descriptions, use_description,
            example_root, current_example_id
        )
        
        if complete_schema:
            schema_lines.append(complete_schema)
            if task_logger:
                task_logger.debug(f"✅ 成功获取表 {table_name} 的完整schema")
        else:
            # 回退到基本信息（包含WARNING）
            if task_logger:
                task_logger.warning(f"⚠️ 未找到表 {table_name} 的完整schema（JSON文件），回退到基本信息（仅表名和描述，无列信息）")
                task_logger.warning(f"   这可能导致LLM无法生成正确的SQL！")
            
            schema_line = f"Table full name: {table_name}"
            if use_description and table_name in table_descriptions:
                desc_raw = table_descriptions[table_name]
                desc = desc_raw.get("description", str(desc_raw)) if isinstance(desc_raw, dict) else str(desc_raw)
                if desc:
                    schema_line += f"\nTable Description: {desc}"
            
            # 添加警告信息到schema中，提醒LLM列信息缺失
            schema_line += f"\n[WARNING: Column schema information not available for this table. Query this table to discover its columns.]"
            schema_lines.append(schema_line)
    
    return "\n\n".join(schema_lines)




def _extract_sql_from_response(response):
    """从LLM响应中提取SQL查询"""
    import re
    
    # 查找 ```sql ... ``` 代码块
    sql_pattern = r"```sql\s+(.*?)\s+```"
    matches = re.findall(sql_pattern, str(response), re.DOTALL | re.IGNORECASE)
    
    if matches:
        sql_query = matches[0].strip()
        # 移除注释行
        lines = [line for line in sql_query.split('\n') if not line.strip().startswith('--')]
        return '\n'.join(lines).strip()
    
    # 如果没有代码块，尝试直接查找SELECT语句
    if "SELECT" in str(response).upper():
        lines = str(response).split('\n')
        sql_lines = []
        in_sql = False
        for line in lines:
            if "SELECT" in line.upper():
                in_sql = True
            if in_sql:
                sql_lines.append(line)
                if line.strip().endswith(';'):
                    break
        if sql_lines:
            return '\n'.join(sql_lines).strip().rstrip(';')
    
    return None





def _plan_step3_action(task, schema_summary, model, db_engine, iteration, max_iterations,
                       last_sql=None, last_sql_error=None, last_sql_feedback=None, task_logger=None):
    """统一决策：让LLM在当前上下文选择动作
    这是一个迭代式的schema验证与优化过程：
      - GENERATE_SQL: 当不确定schema是否足够时，生成SQL验证，结果会传递到下一轮
      - SUBTASK: 当schema不充分时，生成子任务搜索新表，新表会传递到下一轮
      - FINISH_SUFFICIENT: schema充分，可以完成任务
      - FINISH_INSUFFICIENT: 确认无法完成任务
    """
    # 根据数据库类型添加SQL方言指导
    dialect_guide = ""
    if db_engine == "snowflake":
        dialect_guide = """
=== SNOWFLAKE SQL SYNTAX RULES (CRITICAL!) ===
**Column Name Quoting (MUST FOLLOW):**
- Snowflake converts unquoted identifiers to UPPERCASE automatically
- If a column name contains lowercase letters (e.g., HGNC_gene_symbol), you MUST use double quotes: "HGNC_gene_symbol"
- Without quotes, HGNC_gene_symbol becomes HGNC_GENE_SYMBOL and will cause "invalid identifier" error
- **ALWAYS enclose ALL column names in double quotes to preserve case: "column_name"**

**Correct Examples:**
- ✅ SELECT "HGNC_gene_symbol", "normalized_count" FROM table WHERE "HGNC_gene_symbol" = 'TP53'
- ❌ SELECT HGNC_gene_symbol FROM table  (becomes HGNC_GENE_SYMBOL - wrong!)
- ❌ SELECT 'HGNC_gene_symbol' FROM table  (single quotes are for strings, not identifiers!)

**Table Names:**
- Full format: DATABASE.SCHEMA.TABLE or "DATABASE"."SCHEMA"."TABLE"
- Always use the exact format shown in the schema above
"""
    elif db_engine == "bigquery":
        dialect_guide = """
=== BIGQUERY SQL SYNTAX RULES ===
- Enclose column names and table identifiers with backticks: `column_name`
- Table format: `database.schema.table`
- Example: SELECT `column_name` FROM `database.schema.table` WHERE `column_name` = 'value'
"""
    elif db_engine == "sqlite":
        dialect_guide = """
=== SQLITE SQL SYNTAX RULES ===
- Enclose table and column names with double quotes if they contain special characters: "column_name"
- Example: SELECT "column_name" FROM "table_name" WHERE "column_name" = 'value'
"""
    
    prompt = f"""You are an intelligent schema validation assistant. Your task is to determine if the current schema is sufficient to answer the user's question through an iterative verification process.

=== USER QUESTION ===
{task}

=== CURRENT CONTEXT ===
Database Engine: {db_engine}
Current Iteration: {iteration}/{max_iterations}
{dialect_guide}
=== CURRENT LINKED SCHEMA ===
{schema_summary}

=== VERIFICATION HISTORY ===
Last SQL Query: {last_sql if last_sql else "None (first iteration)"}
Last SQL Error: {last_sql_error if last_sql_error else "None"}
Last SQL Feedback: {last_sql_feedback if last_sql_feedback else "None"}

=== YOUR GOAL ===
Analyze the current schema and verification history to decide the next action. This is an iterative process where:
1. SQL verification results (success or error) are passed to the next iteration
2. New tables from subtask searches are added to the schema for the next iteration
3. You can refine your understanding through multiple rounds

=== DECISION RULES ===

**CRITICAL: Learn from previous errors! If the last SQL query failed, DO NOT repeat the same mistake.**
- Analyze the error message carefully
- Check the actual column names in the schema
- Use exact column names as shown in the schema (case-sensitive)
- **For Snowflake: ALWAYS wrap column names in double quotes to preserve case**
- **"invalid identifier" error usually means: wrong column name OR missing quotes in Snowflake**
- DO NOT generate the same SQL that just failed

1. **UNCERTAIN about schema sufficiency** → Choose GENERATE_SQL
   - Generate a SQL query to test if current schema can answer the question
   - **MUST use the EXACT column names shown in the schema above**
   - **SNOWFLAKE: Wrap ALL column names in double quotes: "column_name"**
   - **BIGQUERY: Wrap identifiers in backticks: `column_name`**
   - **SQLITE: Use double quotes for special cases: "column_name"**
   - **If a table has WARNING about missing column schema, use SELECT * FROM table LIMIT 1 to discover its columns first**
   - Keep it simple and add LIMIT 20
   - If previous SQL failed with "invalid identifier" error:
     * Check if column name is correct in schema
     * For Snowflake: Add double quotes around ALL column names
     * Example: Change `HGNC_gene_symbol` to `"HGNC_gene_symbol"`
   - **DO NOT guess column names! If column names are not shown in schema, query the table to discover them**
   - The SQL execution result (data preview or error message) will be passed to the next iteration

2. **INSUFFICIENT schema (missing critical tables/columns)** → Choose SUBTASK
   - Only when truly missing required tables (e.g., no expression table, no mutation table, no join keys)
   - **NOT when SQL fails due to wrong column names or missing quotes** - that means you need to fix the SQL syntax, not request new tables
   - **NOT when you get "invalid identifier" error** - first try adding proper quotes (for Snowflake: double quotes)
   - Describe what specific data/table is needed
   - Graph search will find new tables, which will be added to schema for the next iteration
   - Prefer using WHERE filters on existing tables rather than requesting new tables

3. **SUFFICIENT schema** → Choose FINISH_SUFFICIENT
   - Current schema contains all necessary tables and columns
   - SQL verification succeeded OR you're confident it will work with correct syntax

4. **IMPOSSIBLE to complete** → Choose FINISH_INSUFFICIENT
   - After multiple attempts with different approaches, the question cannot be answered
   - Critical information is fundamentally missing from the database
   - **NOT when SQL just has syntax/quoting errors** - fix the SQL instead
   - Only use this when you've tried SQL validation and discovered the required data truly doesn't exist

=== OUTPUT FORMAT ===
You must output exactly two lines:

ACTION: <one of: GENERATE_SQL | SUBTASK | FINISH_SUFFICIENT | FINISH_INSUFFICIENT>
CONTENT: <the SQL query | what table/data is missing | reason for sufficiency | reason for impossibility>

=== EXAMPLES ===

Example 1 - Snowflake with mixed-case column names (MUST use double quotes):
Schema shows: Column name: HGNC_gene_symbol Type: TEXT
ACTION: GENERATE_SQL
CONTENT: SELECT "HGNC_gene_symbol", "normalized_count" FROM TCGA_HG19_DATA_V0.RNASEQ_GENE_EXPRESSION_UNC_RSEM WHERE "HGNC_gene_symbol" = 'TP53' LIMIT 20

Example 2 - Snowflake error "invalid identifier 'HGNC_GENE_SYMBOL'" - fix by adding quotes:
(Previous SQL: SELECT HGNC_gene_symbol ... → Snowflake converts to HGNC_GENE_SYMBOL)
(Schema shows: Column name: HGNC_gene_symbol)
ACTION: GENERATE_SQL
CONTENT: SELECT "HGNC_gene_symbol" FROM table WHERE "HGNC_gene_symbol" = 'TP53' LIMIT 20

Example 3 - BigQuery with backticks:
ACTION: GENERATE_SQL
CONTENT: SELECT `gene_name`, `expression_value` FROM `database.schema.table` WHERE `gene_name` = 'TP53' LIMIT 20

Example 4 - Table has WARNING about missing column schema, discover columns first:
ACTION: GENERATE_SQL
CONTENT: SELECT * FROM RNASEQ_GENE_EXPRESSION_UNC_RSEM LIMIT 1

Example 5 - SQL failed because truly missing mutation table:
ACTION: SUBTASK
CONTENT: Need mutation data table to correlate with expression levels for TP53 gene

Example 6 - SQL succeeded after correction:
ACTION: FINISH_SUFFICIENT
CONTENT: Current schema contains all required tables and columns, SQL verification succeeded

Example 7 - After 3 iterations, fundamental data missing:
ACTION: FINISH_INSUFFICIENT
CONTENT: Database does not contain patient survival data needed for the analysis

Now analyze the context above and provide your decision:
"""

    try:
        if task_logger:
            task_logger.info("⏳ 调用LLM进行决策...")
        
        model.init_messages()
        response = model.get_model_response_txt(prompt)
        
        if task_logger:
            task_logger.info("")
            task_logger.info("🤖 LLM原始响应:")
            task_logger.info("-" * 80)
            task_logger.info(f"{response}")
            task_logger.info("-" * 80)

        lines = str(response).split('\n')
        action = None
        content = None
        for line in lines:
            if line.strip().upper().startswith("ACTION:"):
                action = line.split(":", 1)[1].strip().upper()
            elif line.strip().upper().startswith("CONTENT:"):
                content = line.split(":", 1)[1].strip()
        if action is None:
            # fallback: try to detect keywords
            resp_upper = str(response).upper()
            if "SUBTASK" in resp_upper:
                action = "SUBTASK"
            elif "GENERATE_SQL" in resp_upper or "SELECT" in resp_upper:
                action = "GENERATE_SQL"
            elif "INSUFFICIENT" in resp_upper:
                action = "FINISH_INSUFFICIENT"
            else:
                action = "FINISH_SUFFICIENT"
            content = content or response
        return action, content
    except Exception as e:
        logger.warning(f"决策器异常，默认尝试生成SQL: {e}")
        return "GENERATE_SQL", None


def _search_tables_for_subtask(subtask, database_graphs, table_descriptions, rerank_components,
                               all_embeddings, embedding_model, use_description,
                               example_root, current_example_id, task_logger=None,
                               enable_batch_rerank=False, batch_size=10,
                               conditional_manager=None, query_context=None, workload_weight=0.1,
                                          edge_workload_weight=None, node_workload_weight=None,
                               enable_graph_topology=True):
    """为新子任务搜索相关表（复用PageRank搜索逻辑）"""
    # 🔬 处理消融实验参数
    if edge_workload_weight is None:
        edge_workload_weight = workload_weight
    if node_workload_weight is None:
        node_workload_weight = workload_weight
    if task_logger:
        task_logger.info("")
        task_logger.info("=" * 80)
        task_logger.info("🔍 子任务表搜索流程:")
        task_logger.info("=" * 80)
        task_logger.info(f"📥 输入信息:")
        task_logger.info(f"  🎯 子任务: {subtask}")
        task_logger.info(f"  📊 可用数据库图: {len(database_graphs)} 个")
        for db_name, graph in database_graphs.items():
            if graph and hasattr(graph, 'nodes'):
                task_logger.info(f"    - {db_name}: {len(graph.nodes())} 节点")
        task_logger.info("-" * 80)
    
    logger.info(f"🔍 为子任务搜索相关表...")
    
    # 复用PageRank搜索逻辑
    try:
        # 识别相关数据库（创建副本避免修改原始图）
        working_graphs = {}
        for db_name, graph in database_graphs.items():
            working_graphs[db_name] = graph.copy()
        
        # 🔧 加载预计算的表embedding缓存（如果尚未加载）
        for db_name, graph in working_graphs.items():
            if not hasattr(graph, '_table_embeddings_cache'):
                if task_logger:
                    task_logger.info(f"📊 为数据库 {db_name} 加载预计算的表embedding...")
        
        _load_precomputed_table_embeddings(working_graphs, example_root, current_example_id)
        
        # 🎯 计算所有表与子任务的相似度（用作节点-总任务相似度的替代）
        node_task_similarities = {}
        if embedding_model:
            try:
                subtask_embedding = embedding_model.encode(subtask, normalize_embeddings=True)
                for db_name, graph in working_graphs.items():
                    if hasattr(graph, '_table_embeddings_cache'):
                        for table_name, table_emb in graph._table_embeddings_cache.items():
                            if table_emb is not None:
                                similarity = float(np.dot(subtask_embedding, table_emb))
                                node_task_similarities[table_name] = similarity
                if task_logger and node_task_similarities:
                    task_logger.info(f"📊 计算了 {len(node_task_similarities)} 个表与子任务的相似度")
            except Exception as e:
                logger.warning(f"计算节点-子任务相似度失败: {e}")
        
        # 更新图权重
        for db_name in working_graphs:
            if working_graphs[db_name].nodes():
                working_graphs[db_name] = update_graph_weights_for_subquery_optimized(
                    working_graphs[db_name], subtask, table_descriptions, embedding_model,
                    node_task_similarities=node_task_similarities,  # 🔧 传递相似度字典
                    conditional_manager=conditional_manager,  # 🆕 Workload boost
                    query_context=query_context,  # 🆕 Query context
                    workload_weight=workload_weight,  # 🆕 Workload weight

                    edge_workload_weight=edge_workload_weight,  # 🔬 λ: 边增强权重

                    node_workload_weight=node_workload_weight  # 🔬 γ: 节点先验权重
                )
        
        # 选择起始节点
        start_nodes = []
        for db_name, graph in working_graphs.items():
            if graph.nodes():
                nodes = select_best_start_node_for_subquery(
                    subtask, graph, table_descriptions, embedding_model, top_k=3
                )
                start_nodes.extend(nodes)
        
        if not start_nodes:
            logger.warning("未找到起始节点")
            return []
        
        # 运行PageRank搜索
        results = expand_via_pagerank_subgraph(
            subtask, working_graphs, set(start_nodes),
            table_descriptions, rerank_components, embedding_model,
            subtask, use_description, task_logger=task_logger,
            example_root=example_root, current_example_id=current_example_id,
            enable_batch_rerank=enable_batch_rerank, batch_size=batch_size,
            edge_workload_weight=0.0, node_workload_weight=0.0,  # 🔬 这里未启用 workload
            enable_graph_topology=enable_graph_topology
        )
        
        # 只返回相关的表
        relevant_tables = [r for r in results if r.get('answer') == 'Y']
        
        if task_logger:
            task_logger.info("")
            task_logger.info("📤 子任务搜索输出结果:")
            task_logger.info("-" * 80)
            task_logger.info(f"📊 搜索结果统计:")
            task_logger.info(f"  - 总评估表数: {len(results)}")
            task_logger.info(f"  - 相关表数: {len(relevant_tables)}")
            task_logger.info(f"  - 不相关表数: {len(results) - len(relevant_tables)}")
            if relevant_tables:
                task_logger.info("")
                task_logger.info("✅ 相关表列表:")
                for idx, table in enumerate(relevant_tables, 1):
                    table_name = table.get('table name', 'unknown')
                    db_name = table.get('db_name', 'unknown')
                    pagerank_score = table.get('pagerank_score', 0.0)
                    task_logger.info(f"  {idx}. {table_name}")
                    task_logger.info(f"      数据库: {db_name}")
                    task_logger.info(f"      PageRank: {pagerank_score:.6f}")
            task_logger.info("=" * 80)
        
        return relevant_tables
        
    except Exception as e:
        logger.error(f"子任务表搜索失败: {e}")
        if task_logger:
            task_logger.error("")
            task_logger.error("=" * 80)
            task_logger.error(f"❌ 子任务表搜索失败: {e}")
            task_logger.error("=" * 80)
        return []


def schema_linking_with_graph_search(task, rerank_components, database_graphs, 
                                   table_descriptions, use_description=True, 
                                   max_consecutive_failures=3, all_embeddings=None,
                                   current_table_set=None, expansion_search_count=None,
                                   example_root=None, current_example_id=None,
                                   use_subquery_decomposition=False, embedding_model=None,
                                   task_logger=None, enable_topk_rerank=False, 
                                   top_k_preselection=5, use_coverage_bonus=False, coverage_beta=0.3,
                                   enable_sql_validation=False, max_validation_iterations=3,
                                   sqlite_path=None, db_engine=None,
                                   enable_batch_rerank=False, batch_size=10,
                                   use_workload_evolution=False,
                                   workload_stats_path='data/workload_stats.json',
                                   workload_weight=0.1,
                                   edge_workload_weight=None, node_workload_weight=None,
                                   enable_graph_topology=True):
    """
    图搜索的统一入口函数，支持传统方法和子查询分解方法
    
    Args:
        task: 原始任务
        rerank_components: LLM组件
        database_graphs: 数据库图字典
        table_descriptions: 表描述字典
        use_description: 是否使用描述
        max_consecutive_failures: 最大连续失败次数
        all_embeddings: 预计算的嵌入向量
        current_table_set: 当前表集合
        expansion_search_count: 扩展搜索计数
        example_root: 样本根目录
        current_example_id: 当前样本ID
        use_subquery_decomposition: 是否使用子查询分解方法
        embedding_model: 嵌入模型
        use_coverage_bonus: 是否使用Coverage Bonus增强多样性 (新增)
        coverage_beta: Coverage Bonus权重系数 (新增)
        enable_sql_validation: 是否启用SQL验证迭代优化 (新增)
        max_validation_iterations: SQL验证最大迭代次数 (新增)
        sqlite_path: SQLite数据库路径 (用于SQL验证)
        db_engine: 数据库引擎类型 (sqlite/snowflake/bigquery)
    
    Returns:
        List[Dict]: 选中的表列表
    """
    # 🔬 处理消融实验参数
    if edge_workload_weight is None:
        edge_workload_weight = workload_weight
    if node_workload_weight is None:
        node_workload_weight = workload_weight
    
    processed_tables = set()
    
    # 🆕 Workload Evolution: 加载 conditional functions 和提取 query context
    conditional_manager = None
    query_context = None
    
    if use_workload_evolution:
        try:
            # 检查 workload stats 文件是否存在
            if os.path.exists(workload_stats_path):
                logger.info(f"✨ 启用 Workload Evolution")
                logger.info(f"📂 加载 workload statistics: {workload_stats_path}")
                
                # 导入模块
                from conditional_functions import ConditionalFunctionManager
                from query_context_extractor import QueryContextExtractor
                
                # 加载 workload statistics
                import json
                with open(workload_stats_path, 'r') as f:
                    workload_stats = json.load(f)
                
                # 创建 conditional function manager
                conditional_manager = ConditionalFunctionManager(workload_stats)
                
                # 提取 query context（规则模式，快速）
                extractor = QueryContextExtractor(use_llm=False)
                query_context = extractor.extract(task)
                
                logger.info(f"✅ Conditional Functions 已加载")
                logger.info(f"🔍 Query Context: {query_context.to_dict()}")
                
                if task_logger:
                    task_logger.info(f"✨ Workload Evolution 已启用")
                    task_logger.info(f"  Workload weight: {workload_weight}")
                    task_logger.info(f"  Query context: {query_context.to_dict()}")
            else:
                logger.warning(f"⚠️ Workload stats file not found: {workload_stats_path}")
                logger.warning(f"⚠️ Workload Evolution 被禁用")
                use_workload_evolution = False
        except Exception as e:
            logger.error(f"❌ 加载 Workload Evolution 失败: {e}")
            logger.warning("⚠️ 回退到不使用 workload evolution")
            use_workload_evolution = False
            conditional_manager = None
            query_context = None
    
    if use_subquery_decomposition:
        logger.info("🎯 使用子查询分解方法进行表搜索")
        if task_logger:
            task_logger.info("🚀 启用子查询分解搜索方法")
            task_logger.info("=" * 80)
            task_logger.info("第一步: 任务分解")
            task_logger.info("=" * 80)
        
        selected_tables = expand_via_subquery_decomposition(
            task, rerank_components, database_graphs, table_descriptions,
            processed_tables, use_description, max_consecutive_failures,
            all_embeddings, current_table_set, expansion_search_count,
            example_root, current_example_id, embedding_model, task_logger,
            top_k_preselection=top_k_preselection,
            enable_topk_rerank=enable_topk_rerank,
            use_coverage_bonus=use_coverage_bonus,
            coverage_beta=coverage_beta,
            enable_batch_rerank=enable_batch_rerank,
            batch_size=batch_size,
            conditional_manager=conditional_manager,
            query_context=query_context,
            workload_weight=workload_weight,
            edge_workload_weight=edge_workload_weight,
            node_workload_weight=node_workload_weight,
            enable_graph_topology=enable_graph_topology
        )
        
        # 🚀 Step 3: SQL验证与迭代优化 (可选)
        # 支持所有数据库类型：SQLite使用sqlite_path，Snowflake/BigQuery使用current_example_id
        if enable_sql_validation:
            logger.info("=" * 80)
            logger.info("🔍 Step 3: Schema验证与SQL迭代优化")
            logger.info("=" * 80)
            if task_logger:
                task_logger.info("=" * 80)
                task_logger.info("第三步: Schema验证与SQL迭代优化")
                task_logger.info("=" * 80)

            # 仅保留标记为相关(Y)的表，且按表名去重
            filtered_tables = []
            seen_table_names = set()
            for tbl in selected_tables:
                if tbl.get("answer") != "Y":
                    continue
                table_name = tbl.get("table name")
                if not table_name or table_name in seen_table_names:
                    continue
                filtered_tables.append(tbl)
                seen_table_names.add(table_name)

            logger.info(f"📊 Step 3: using {len(filtered_tables)}/{len(selected_tables)} relevant tables after filtering")
            if task_logger:
                task_logger.info(f"📊 Step 3: 仅保留相关表 {len(filtered_tables)}/{len(selected_tables)} 个用于验证")

            if not filtered_tables:
                logger.warning("⚠️ 没有标记为相关的表，跳过Schema验证")
                if task_logger:
                    task_logger.warning("⚠️ 没有标记为相关的表，跳过Schema验证")
                return selected_tables
            
            selected_tables = validate_and_refine_schema_with_sql(
                task=task,
                selected_tables=filtered_tables,
                rerank_components=rerank_components,
                database_graphs=database_graphs,
                table_descriptions=table_descriptions,
                use_description=use_description,
                all_embeddings=all_embeddings,
                current_table_set=current_table_set,
                expansion_search_count=expansion_search_count,
                example_root=example_root,
                current_example_id=current_example_id,
                embedding_model=embedding_model,
                task_logger=task_logger,
                max_iterations=max_validation_iterations,
                sqlite_path=sqlite_path,
                db_engine=db_engine,
                enable_batch_rerank=enable_batch_rerank,
                batch_size=batch_size
            )
        
        return selected_tables
    else:
        logger.info("🔍 使用传统MinHash扩展方法进行表搜索")
        
        # 使用传统方法：首先进行初始种子选择，然后扩展
        # 这里需要实现初始种子选择逻辑
        initial_seeds = select_initial_seed_tables(
            task, database_graphs, table_descriptions, all_embeddings, 
            embedding_model, top_k=3
        )
        
        all_expanded_tables = []
        successful_paths = {}
        global_candidate_cache = set()
        duplicate_candidates_avoided = [0]
        
        for seed_info in initial_seeds:
            seed_table = seed_info['table_name']
            logger.info(f"🌱 从种子表开始扩展: {seed_table}")
            
            # 构建种子表信息
            seed_enhanced_info = {
                'table_name': seed_table,
                'enhanced_schema': seed_info.get('schema', f"Table full name: {seed_table}"),
                'level': 0
            }
            
            # 使用原有的expand_via_minhash函数进行扩展
            expanded_from_seed = expand_via_minhash(
                seed_table, seed_enhanced_info, task, rerank_components,
                database_graphs, table_descriptions, processed_tables,
                use_description, max_consecutive_failures, all_embeddings,
                current_table_set, expansion_search_count, 1, 3, None,
                global_candidate_cache, duplicate_candidates_avoided,
                successful_paths, example_root, current_example_id
            )
            
            all_expanded_tables.extend(expanded_from_seed)
            
            # 将种子表本身也加入结果（如果相关）
            if seed_table not in processed_tables:
                seed_schema = get_complete_neighbor_schema(
                    seed_table, table_descriptions, use_description,
                    example_root, current_example_id
                )
                if not seed_schema:
                    seed_schema = f"Table full name: {seed_table}"
                    if use_description and seed_table in table_descriptions:
                        desc = table_descriptions[seed_table]
                        if desc:
                            seed_schema += f"\nTable Description: {desc}"
                
                is_seed_relevant = rerank_single_table(
                    task, seed_schema, rerank_components
                )
                
                if is_seed_relevant:
                    seed_result = {
                        "think": "",
                        "answer": "Y",
                        "columns": [],
                        "table name": seed_table,
                        "score": 0.95,  # 种子表给高分
                        "similarity": seed_info.get('similarity', 0.9),
                        "expansion_level": 0,
                        "parent_table": "seed",
                        "selection_method": "initial_seed"
                    }
                    all_expanded_tables.insert(0, seed_result)  # 种子表放在前面
            
            processed_tables.add(seed_table)
        
        logger.info(f"🎉 传统方法搜索完成，总共选中 {len(all_expanded_tables)} 个表")
        return all_expanded_tables


def select_initial_seed_tables(task, database_graphs, table_descriptions, 
                              all_embeddings=None, embedding_model=None, top_k=3):
    """
    选择初始种子表
    
    Args:
        task: 原始任务
        database_graphs: 数据库图字典
        table_descriptions: 表描述字典
        all_embeddings: 预计算的嵌入向量
        embedding_model: 嵌入模型
        top_k: 返回前k个种子表
    
    Returns:
        List[Dict]: 种子表信息列表
    """
    logger.info(f"🌱 选择初始种子表 (top-{top_k})")
    
    # 收集所有表
    all_tables = set()
    for db_name, graph in database_graphs.items():
        all_tables.update(graph.nodes())
    
    table_scores = []
    
    if embedding_model is not None:
        # 使用嵌入模型计算相似度
        task_embedding = embedding_model.encode(task, normalize_embeddings=True)
        
        for table in all_tables:
            try:
                # 优先使用预计算的嵌入
                if all_embeddings and table in all_embeddings:
                    table_embedding = all_embeddings[table]
                else:
                    # 现场计算
                    table_text = _build_table_text_representation(table, table_descriptions)
                table_embedding = embedding_model.encode(table_text, normalize_embeddings=True)
                
                similarity = float(np.dot(task_embedding, table_embedding))
                table_scores.append({
                    'table_name': table,
                'similarity': similarity,
                    'method': 'embedding'
            })
            
            except Exception as e:
                logger.debug(f"计算表 {table} 嵌入相似度失败: {e}")
                # 回退到文本匹配
                table_text = _build_table_text_representation(table, table_descriptions).lower()
                task_lower = task.lower()
                keywords = re.findall(r'\b\w+\b', task_lower)
                keywords = [kw for kw in keywords if len(kw) > 2]
                
                score = sum(1 for kw in keywords if kw in table_text)
                if keywords:
                    score = score / len(keywords)
                
                table_scores.append({
                    'table_name': table,
                    'similarity': score,
                    'method': 'text_match'
                })
    else:
        # 使用文本匹配
        task_lower = task.lower()
        keywords = re.findall(r'\b\w+\b', task_lower)
        keywords = [kw for kw in keywords if len(kw) > 2]
        
        for table in all_tables:
            table_text = _build_table_text_representation(table, table_descriptions).lower()
            score = sum(1 for kw in keywords if kw in table_text)
            if keywords:
                score = score / len(keywords)
            
            table_scores.append({
                'table_name': table,
                'similarity': score,
                'method': 'text_match'
            })
    
    # 排序并选择top-k
    table_scores.sort(key=lambda x: x['similarity'], reverse=True)
    selected_seeds = table_scores[:top_k]
    
    logger.info(f"✅ 选择的种子表:")
    for i, seed in enumerate(selected_seeds, 1):
        logger.info(f"  {i}. {seed['table_name']} (相似度: {seed['similarity']:.3f}, 方法: {seed['method']})")
    
    return selected_seeds


# ================= PageRank子图搜索模块 =================

def extract_table_columns_from_graph(graph, table_name):
    """
    🆕 从图节点的 metadata 中提取表的列信息
    
    Args:
        graph: NetworkX图对象
        table_name: 表名
    
    Returns:
        List[str]: 列名列表
    """
    columns = []
    
    # 尝试从图节点获取列信息
    if table_name in graph.nodes():
        node_data = graph.nodes[table_name]
        
        # 方法 1: 直接从 'columns' 属性获取（优先）
        if 'columns' in node_data:
            columns = node_data['columns']
            logger.debug(f"✅ 从 'columns' 属性提取 {table_name}: {len(columns)} 列")
        
        # 方法 2: 🆕 从 'schema_str' 解析列名（最可靠）
        elif 'schema_str' in node_data:
            schema_str = node_data['schema_str']
            # 解析格式：Column name: xxx Type: yyy
            import re
            col_matches = re.findall(r'Column name:\s*([^\s]+)', schema_str)
            if col_matches:
                columns = [col.strip() for col in col_matches]
                logger.debug(f"✅ 从 'schema_str' 解析 {table_name}: {len(columns)} 列")
        
        # 方法 3: 从 'description' 中解析（fallback）
        elif 'description' in node_data:
            desc = node_data['description']
            # 尝试解析 "Column name: xxx Type: yyy" 格式（如果 description 包含 schema）
            import re
            col_match = re.findall(r'Column name:\s*([^\s\n]+)', desc)
            if col_match:
                columns = [c.strip() for c in col_match]
                logger.debug(f"✅ 从 'description' (schema格式) 解析 {table_name}: {len(columns)} 列")
            else:
                # 尝试简单的 "Column: xxx" 或 "Columns: xxx, yyy" 格式
                col_match = re.findall(r'Column[s]?:\s*([^\n]+)', desc, re.IGNORECASE)
                if col_match:
                    # 分割列名
                    columns = [c.strip() for col_str in col_match 
                              for c in col_str.split(',')]
                    logger.debug(f"✅ 从 'description' (简单格式) 解析 {table_name}: {len(columns)} 列")
        
        # 方法 4: 查找与该表连接的列节点（如果是列级图）
        # （表级图通常不需要这个）
        
        if not columns:
            logger.debug(f"⚠️ 无法提取 {table_name} 的列信息，节点属性: {list(node_data.keys())[:5]}...")
    
    return columns


def build_table_to_columns_map(graph, tables):
    """
    🆕 批量构建表到列的映射
    
    Args:
        graph: NetworkX图对象
        tables: 表名列表
    
    Returns:
        Dict[str, List[str]]: {table_name: [column_names]}
    """
    table_to_columns = {}
    
    for table in tables:
        columns = extract_table_columns_from_graph(graph, table)
        table_to_columns[table] = columns
    
    return table_to_columns
  
def run_personalized_pagerank_for_subquery(graph, subquery, embedding_model, 
                                         table_descriptions, current_layer_nodes,
                                         start_node_bias_weight=2.0, task_logger=None,
                                         conditional_manager=None, query_context=None, 
                                         workload_weight=0.1,
                                          edge_workload_weight=None, node_workload_weight=None):
    """
    基于边权重和起始节点偏置计算个性化PageRank分布
    
    🆕 增强版：加入 workload-based node prior
    个性化分布 = softmax(边权重总和 + 起始节点偏置 + workload_prior)
    
    Args:
        graph: NetworkX图对象
        subquery: 当前子查询
        embedding_model: 嵌入模型
        table_descriptions: 表描述字典
        current_layer_nodes: 当前层起始节点集合（用于偏置）
        start_node_bias_weight: 起始节点偏置权重，默认2.0
        task_logger: 任务日志记录器
        conditional_manager: 🆕 条件函数管理器
        query_context: 🆕 查询上下文
        workload_weight: 🆕 workload 权重系数
        
    Returns:
        Tuple[Dict, Dict]: (PageRank分数字典, 节点权重详情字典)
    """
    # 🔬 处理消融实验参数
    if edge_workload_weight is None:
        edge_workload_weight = workload_weight
    if node_workload_weight is None:
        node_workload_weight = workload_weight
    if task_logger:
        task_logger.info(f"🎯 计算混合个性化PageRank分布（边权重 + 起始节点偏置 + workload_prior）")
        task_logger.info(f"📊 图信息: {len(graph.nodes())} 个节点, {len(graph.edges())} 条边")
        if current_layer_nodes:
            task_logger.info(f"🚀 起始节点偏置: {len(current_layer_nodes)} 个节点，权重 {start_node_bias_weight}")
        if conditional_manager and query_context:
            task_logger.info(f"✨ 启用 workload-based node prior（权重系数: {workload_weight}）")
    
    # 1. 计算每个节点的邻接边权重总和 + 起始节点偏置 + workload_prior
    node_edge_weight_sums = {}
    node_weight_details = {}  # 详细权重分解信息
    
    # 确保current_layer_nodes是集合类型
    start_nodes = set(current_layer_nodes) if current_layer_nodes else set()
    
    # 🆕 准备列级 boost 所需的数据
    table_to_columns = {}
    selected_tables_list = list(start_nodes) if start_nodes else []
    
    if conditional_manager and query_context:
        # 批量提取所有节点的列信息
        all_nodes = list(graph.nodes())
        table_to_columns = build_table_to_columns_map(graph, all_nodes)
        
        if task_logger:
            tables_with_columns = sum(1 for cols in table_to_columns.values() if cols)
            task_logger.info(f"📋 提取列信息: {tables_with_columns}/{len(all_nodes)} 个表有列数据")
    
    for node in graph.nodes():
        # 计算该节点所有邻接边的权重总和
        edge_weight_sum = 0.0
        neighbor_count = 0
        
        for neighbor in graph.neighbors(node):
            edge_data = graph.get_edge_data(node, neighbor)
            if edge_data and 'weight' in edge_data:
                edge_weight_sum += edge_data['weight']
                neighbor_count += 1
        
        # 基础权重：边权重总和（结构 + 语义，对孤立节点给最小权重）
        base_weight = edge_weight_sum if neighbor_count > 0 else 0.1
        
        # 起始节点偏置
        start_bias = start_node_bias_weight if node in start_nodes else 0.0
        
        # 🆕 Workload 节点先验（Predicate + Aggregation）
        workload_prior = 0.0
        if conditional_manager and query_context:
            node_columns = table_to_columns.get(node, [])
            if node_columns:
                # 计算该表的节点先验分数（基于 pred/agg，已 p95 归一化到 [0, ~1]）
                workload_prior = conditional_manager.compute_table_prior(
                    table=node,
                    table_columns=node_columns,
                    context=query_context,
                    weights={'predicate': 0.4, 'aggregation': 0.6}
                )
        
        # 🔧 最终权重 = 基础权重 + 起始偏置 + workload 偏置
        # workload_weight ∈ [0.2, 1.0] 控制 workload 影响力（默认 1.0）
        # workload_prior 已经是 [0, ~1]，不需要额外缩放
        final_weight = base_weight + start_bias + node_workload_weight * workload_prior  # 🔬 γ: 节点先验权重
        
        node_edge_weight_sums[node] = final_weight
        node_weight_details[node] = {
            'base_weight': base_weight,
            'start_bias': start_bias,
            'workload_prior': workload_prior,  # 🆕 节点先验
            'final_weight': final_weight,
            'is_start_node': node in start_nodes,
            'neighbor_count': neighbor_count
        }
    
    if task_logger:
        # 显示权重分布统计
        weight_values = list(node_edge_weight_sums.values())
        task_logger.info(f"📊 最终权重分布: min={min(weight_values):.3f}, "
                        f"max={max(weight_values):.3f}, mean={np.mean(weight_values):.3f}")
        
        # 🆕 显示 workload 节点先验效果
        if conditional_manager and query_context:
            workload_priors = [details.get('workload_prior', 0.0) for details in node_weight_details.values()]
            non_zero_priors = [p for p in workload_priors if p > 0]
            if non_zero_priors:
                task_logger.info(f"✨ Workload节点先验: {len(non_zero_priors)}/{len(workload_priors)} 个节点有先验分数")
                task_logger.info(f"   先验分数: 平均={np.mean(non_zero_priors):.3f}, 最大={max(non_zero_priors):.3f}")
        
        # 显示起始节点偏置效果
        if start_nodes:
            start_weights = [node_edge_weight_sums[node] for node in start_nodes if node in node_edge_weight_sums]
            non_start_weights = [node_edge_weight_sums[node] for node in node_edge_weight_sums if node not in start_nodes]
            if start_weights and non_start_weights:
                task_logger.info(f"🚀 偏置效果: 起始节点平均权重 {np.mean(start_weights):.3f} vs 非起始节点 {np.mean(non_start_weights):.3f}")
        
        # 显示top-5权重节点（包含偏置详情 + workload先验）
        top_nodes = sorted(node_edge_weight_sums.items(), key=lambda x: x[1], reverse=True)[:5]
        task_logger.info(f"🏆 Top-5最终权重节点:")
        for i, (node, weight) in enumerate(top_nodes, 1):
            details = node_weight_details[node]
            bias_mark = "🚀" if details['is_start_node'] else "  "
            prior = details.get('workload_prior', 0.0)
            prior_mark = "✨" if prior > 0 else "  "
            task_logger.info(f"  {i}. {bias_mark}{prior_mark} {node}: {weight:.3f} "
                           f"(基础:{details['base_weight']:.3f} + 偏置:{details['start_bias']:.3f} + 先验:{prior:.3f})")
    
    # 2. 应用softmax归一化得到个性化分布
    node_names = list(node_edge_weight_sums.keys())
    weight_values = np.array([node_edge_weight_sums[node] for node in node_names])
    
    # Softmax计算（数值稳定版本）
    weight_values_stable = weight_values - np.max(weight_values)
    exp_weights = np.exp(weight_values_stable)
    softmax_probs = exp_weights / np.sum(exp_weights)
    
    # 构建个性化字典
    personalization = {}
    for i, node in enumerate(node_names):
        personalization[node] = float(softmax_probs[i])
    
    if task_logger:
        # 验证归一化
        total_prob = sum(personalization.values())
        task_logger.info(f"✅ Softmax归一化检查: 总概率 = {total_prob:.6f}")
        
        # 显示top-5个性化概率
        top_personalization = sorted(personalization.items(), key=lambda x: x[1], reverse=True)[:5]
        task_logger.info(f"🎯 Top-5个性化概率: {dict(top_personalization)}")
    
    # 3. 运行PageRank
    if task_logger:
        task_logger.info("🚀 开始运行个性化PageRank算法...")
    
    try:
        import networkx as nx
        pagerank_scores = nx.pagerank(
            graph, 
            personalization=personalization,
            alpha=0.85,  # 阻尼系数
            max_iter=100,
            tol=1e-06,
            weight='weight'  # 使用更新后的边权重
        )
        
        if task_logger:
            task_logger.info("✅ PageRank计算完成")
            # 显示PageRank结果统计
            pr_values = list(pagerank_scores.values())
            task_logger.info(f"📊 PageRank分数分布: min={min(pr_values):.6f}, "
                            f"max={max(pr_values):.6f}, mean={np.mean(pr_values):.6f}")
            
            # 显示top-5 PageRank分数
            top_pagerank = sorted(pagerank_scores.items(), key=lambda x: x[1], reverse=True)[:5]
            task_logger.info(f"🏆 Top-5 PageRank分数: {dict(top_pagerank)}")
    
    except Exception as e:
        logger.error(f"PageRank计算失败: {e}")
        if task_logger:
            task_logger.error(f"PageRank计算失败: {e}")
        # 回退到基于边权重的简单排序
        pagerank_scores = {node: weight for node, weight in node_edge_weight_sums.items()}
    
    return pagerank_scores, node_weight_details


def _select_with_coverage_bonus(graph, pagerank_scores, beta, task_logger=None):
    """
    使用Coverage Bonus策略迭代选择节点
    
    公式: Final_Score = PageRank / max(Avg_Similarity, 0.1)
    
    相似度越高，分数被惩罚越大（除数大）
    相似度越低，分数被放大越多（除数小）
    
    早停策略: 动态自适应阈值 = mean(已选表分数) × 0.5
    
    Args:
        graph: NetworkX图对象
        pagerank_scores: PageRank分数字典
        beta: 未使用（保留兼容性）
        task_logger: 日志记录器
        
    Returns:
        List[str]: 选中的节点列表
    """
    selected_nodes = []
    candidates = list(pagerank_scores.keys())
    min_tables = 5
    max_tables = 20
    
    # 尝试从图中获取embedding缓存
    table_embeddings = getattr(graph, '_table_embeddings_cache', {})
    
    # 追踪每轮的分数，用于动态阈值计算
    score_history = []
    
    if task_logger:
        task_logger.info(f"🎯 Coverage Bonus选择 (除法模式, embedding缓存: {len(table_embeddings)} 个表)")
    
    for round_num in range(1, max_tables + 1):
        if not candidates:
            break
        
        best_table = None
        best_score = -float('inf')
        
        for table in candidates:
            pagerank = pagerank_scores.get(table, 0.0)
            
            if not selected_nodes:
                # 第一轮：直接用PageRank
                final_score = pagerank
            else:
                # 计算与已选表的平均相似度
                table_emb = table_embeddings.get(table)
                
                if table_emb is None:
                    # 如果没有embedding，使用默认相似度0.5
                    avg_sim = 0.5
                else:
                    similarities = []
                    for sel_table in selected_nodes:
                        sel_emb = table_embeddings.get(sel_table)
                        if sel_emb is not None:
                            # 计算余弦相似度（embedding已归一化）
                            sim = float(np.dot(table_emb, sel_emb))
                            similarities.append(sim)
                    
                    avg_sim = np.mean(similarities) if similarities else 0.5
                
                # 除法模式：PageRank / Avg_Similarity
                # 添加下界保护，避免除以过小的数导致分数爆炸
                safe_avg_sim = max(avg_sim, 0.1)  # 最小0.1，避免过度放大
                final_score = pagerank / safe_avg_sim
            
            if final_score > best_score:
                best_score = final_score
                best_table = table
        
        if best_table is None:
            break
        
        selected_nodes.append(best_table)
        candidates.remove(best_table)
        score_history.append(best_score)
        
        # 详细日志（只在前3轮和关键轮次）
        if task_logger and (round_num <= 3 or round_num == max_tables or len(selected_nodes) == min_tables):
            short_name = best_table.split('.')[-1]
            task_logger.info(f"  Round {round_num}: 选中 {short_name} (Final_Score={best_score:.4f})")
        
        # 动态自适应早停策略
        if len(selected_nodes) >= min_tables:
            # 计算已选表分数的平均值
            avg_score = np.mean(score_history)
            # 动态阈值：均值的50%
            dynamic_threshold = avg_score * 0.5
            
            if best_score < dynamic_threshold:
                if task_logger:
                    task_logger.info(f"🛑 早停触发: 当前分数 {best_score:.4f} < 动态阈值 {dynamic_threshold:.4f} (均值 {avg_score:.4f} × 50%)")
                break
    
    if task_logger:
        task_logger.info(f"✅ Coverage Bonus选择完成: {len(selected_nodes)} 个表")
        if score_history:
            task_logger.info(f"📊 分数统计: 最高={max(score_history):.4f}, 最低={min(score_history):.4f}, 均值={np.mean(score_history):.4f}")
    
    return selected_nodes


def select_subgraph_by_pagerank(graph, pagerank_scores, selection_strategy="adaptive", 
                                use_coverage_bonus=False, coverage_beta=0.3, task_logger=None):
    """
    基于PageRank分数选择子图
    
    Args:
        graph: NetworkX图对象
        pagerank_scores: PageRank分数字典
        selection_strategy: 选择策略 ("adaptive", "top_k", "threshold")
        use_coverage_bonus: 是否使用Coverage Bonus增强多样性 (新增)
        coverage_beta: Coverage Bonus权重系数 (默认0.3) (新增)
        task_logger: 任务日志记录器
        
    Returns:
        Tuple[nx.Graph, List[str]]: (选中的子图, 选中的节点列表)
    """
    
    # 如果启用Coverage Bonus，使用迭代选择策略
    if use_coverage_bonus:
        selected_nodes = _select_with_coverage_bonus(
            graph, pagerank_scores, coverage_beta, task_logger
        )
        if task_logger:
            task_logger.info(f"📈 PageRank子图选择策略: Coverage Bonus (β={coverage_beta})")
            task_logger.info(f"🎯 选中节点数量: {len(selected_nodes)}/{len(graph.nodes())}")
        # Coverage Bonus模式下直接使用选中的节点，不需要后续的selection_strategy逻辑
    else:
        # 原有逻辑保持不变
        sorted_nodes = sorted(pagerank_scores.items(), key=lambda x: x[1], reverse=True)
        
        if selection_strategy == "adaptive":
            # 自适应策略：基于分数分布动态确定阈值
            scores = [score for _, score in sorted_nodes]
            mean_score = np.mean(scores)
            std_score = np.std(scores)
            
            # 选择超过 (mean + 0.5*std) 的节点，最少5个，最多20个
            threshold = mean_score + 0.5 * std_score
            selected_nodes = []
            
            for node, score in sorted_nodes:
                if score >= threshold or len(selected_nodes) < 5:
                    selected_nodes.append(node)
                if len(selected_nodes) >= 20:
                    break
                    
        elif selection_strategy == "top_k":
            k = min(15, max(5, len(graph.nodes()) // 4))  # 动态K值
            selected_nodes = [node for node, _ in sorted_nodes[:k]]
            
        else:  # threshold strategy
            threshold = np.percentile([score for _, score in sorted_nodes], 80)
            selected_nodes = [node for node, score in sorted_nodes if score >= threshold]
        
        if task_logger:
            task_logger.info(f"📈 PageRank子图选择策略: {selection_strategy}")
            task_logger.info(f"🎯 选中节点数量: {len(selected_nodes)}/{len(graph.nodes())}")
            if selection_strategy == "adaptive":
                task_logger.info(f"📊 自适应阈值: {threshold:.6f} (mean={mean_score:.6f}, std={std_score:.6f})")
    
    # 构建子图（包含选中节点及其之间的边）
    subgraph = graph.subgraph(selected_nodes).copy()
    
    return subgraph, selected_nodes


def evaluate_subgraph_tables(subgraph, selected_nodes, subquery, task, 
                           rerank_components, table_descriptions, 
                           pagerank_scores, use_description=True, task_logger=None,
                           example_root=None, current_example_id=None,
                           enable_batch_rerank=False, batch_size=10,
                           working_graphs=None, enable_graph_topology=True):
    """
    对子图中的表进行LLM相关性判断
    
    Args:
        subgraph: 选中的子图
        selected_nodes: 选中的节点列表
        subquery: 当前子查询
        task: 原始任务
        rerank_components: LLM组件
        table_descriptions: 表描述字典
        pagerank_scores: PageRank分数字典
        use_description: 是否使用表描述
        task_logger: 任务日志记录器
        enable_batch_rerank: 是否启用批量rerank判断（默认False）
        batch_size: 批量判断的批次大小（默认10）
        
    Returns:
        List[Dict]: 评估结果列表
    """
    
    evaluated_tables = []
    
    # 按PageRank分数排序进行评估
    nodes_by_score = sorted(selected_nodes, 
                           key=lambda x: pagerank_scores[x], 
                           reverse=True)
    
    if task_logger:
        task_logger.info(f"🤖 开始对 {len(selected_nodes)} 个子图节点进行LLM相关性判断")
        if enable_batch_rerank:
            task_logger.info(f"⚡ 批量判断模式已启用: 每批 {batch_size} 个表")
    
    # 🚀 批量判断模式
    if enable_batch_rerank:
        # 收集所有可用表（用于分片表扩展）
        all_available_tables = set()
        if working_graphs:
            for graph_name, graph_obj in working_graphs.items():
                all_available_tables.update(graph_obj.nodes())
        
        # 构建所有表的schema字典
        full_task = f"""Original Task: {task}

Current Subquery: {subquery}"""
        
        # 分批处理
        for batch_start in range(0, len(nodes_by_score), batch_size):
            batch_nodes = nodes_by_score[batch_start:batch_start + batch_size]
            
            if task_logger:
                task_logger.info(f"📦 批次 {batch_start//batch_size + 1}: 评估 {len(batch_nodes)} 个表")
            
            # 构建批次的schema字典
            batch_schemas = {}
            for node in batch_nodes:
                table_text = _build_table_text_representation(
                    node, table_descriptions, 
                    example_root=example_root, 
                    current_example_id=current_example_id, 
                    use_description=use_description
                )
                batch_schemas[node] = table_text
            
            # 批量判断（包含分片表扩展），返回 (relevance_dict, columns_map)
            # 传入 subgraph 使 LLM 感知 batch 内部的图拓扑关系（FK/IND/相似度边）
            batch_results, batch_columns_map = rerank_batch_tables(
                full_task, batch_schemas, rerank_components,
                parent_info=None, instruction_suffix="",
                task_logger=task_logger,
                all_available_tables=all_available_tables,
                graph=subgraph,
                enable_graph_topology=enable_graph_topology
            )
            
            # 记录批次结果
            for node in batch_nodes:
                is_relevant = batch_results.get(node, False)
                table_result = {
                    "think": f"Selected by PageRank subgraph search (score: {pagerank_scores[node]:.6f}, batch mode)",
                    "answer": "Y" if is_relevant else "N",
                    "columns": batch_columns_map.get(node, []) if is_relevant else [],
                    "table name": node,
                    "score": 0.9 if is_relevant else 0.1,
                    "pagerank_score": pagerank_scores[node],
                    "subgraph_rank": nodes_by_score.index(node) + 1,
                    "selection_method": "pagerank_subgraph_batch",
                    "subquery": subquery
                }
                evaluated_tables.append(table_result)
                
                if task_logger:
                    status = "✅ 相关" if is_relevant else "❌ 不相关"
                    col_info = f", 列: {batch_columns_map.get(node, [])}" if is_relevant and batch_columns_map.get(node) else ""
                    task_logger.info(f"  {node}: {status} (PageRank: {pagerank_scores[node]:.6f}{col_info})")
            
            # 🔥 处理扩展的分片表（不在batch_nodes中但在batch_results中为True的表）
            # 注意：不影响搜索路径，但加入最终schema
            expanded_tables = [table for table, is_relevant in batch_results.items() 
                             if is_relevant and table not in batch_nodes]
            if expanded_tables:
                if task_logger:
                    task_logger.info(f"🎯 分片表扩展: 添加 {len(expanded_tables)} 个扩展表到最终schema（不影响搜索路径）")
                
                for table in expanded_tables:
                    table_result = {
                        "think": f"Auto-expanded partition table (co-occurrence, not in search path)",
                        "answer": "Y",
                        "columns": batch_columns_map.get(table, []),
                        "table name": table,
                        "score": 0.9,
                        "pagerank_score": 0.0,  # 扩展的表没有PageRank分数
                        "subgraph_rank": -1,
                        "selection_method": "pagerank_subgraph_batch_partition_expanded",
                        "subquery": subquery
                    }
                    evaluated_tables.append(table_result)
                    
                    if task_logger:
                        task_logger.info(f"   ✅ 扩展表 {table} 添加到最终schema")
        
        if task_logger:
            relevant_count = sum(1 for r in evaluated_tables if r["answer"] == "Y")
            task_logger.info(f"✅ 批量评估完成: {relevant_count}/{len(evaluated_tables)} 个表相关")
        
        return evaluated_tables
    
    # 🔄 原有的逐个判断模式
    # 🚀 早停机制参数
    consecutive_failures = 0
    max_consecutive_failures = 3  # 连续3个高分表不相关则停止
    min_evaluations = 3  # 至少评估3个表
    
    for i, node in enumerate(nodes_by_score):
        try:
            # 构建表的完整信息（包含列schema）
            table_text = _build_table_text_representation(
                node, table_descriptions, 
                example_root=example_root, 
                current_example_id=current_example_id, 
                use_description=use_description
            )
            
            # 🔧 不添加PageRank元信息，让LLM纯粹基于任务和表schema判断
            # neighbor_count = len(list(subgraph.neighbors(node)))
            # context_info = f"""
# PageRank Score: {pagerank_scores[node]:.6f} (Rank #{i+1}/{len(selected_nodes)})
# Subgraph Context: Connected to {neighbor_count} other candidate tables"""
            
            # LLM判断 - 只提供任务和子查询，不提供PageRank提示
            full_task = f"""Original Task: {task}

Current Subquery: {subquery}"""
            
            is_relevant = rerank_single_table(
                full_task, table_text, rerank_components,
                parent_info=None  # 不使用parent_info，避免任何PageRank提示
            )
            
            # 记录结果
            table_result = {
                "think": f"Selected by PageRank subgraph search (score: {pagerank_scores[node]:.6f})",
                "answer": "Y" if is_relevant else "N",
                "columns": [],
                "table name": node,
                "score": 0.9 if is_relevant else 0.1,
                "pagerank_score": pagerank_scores[node],
                "subgraph_rank": i + 1,
                "selection_method": "pagerank_subgraph",
                "subquery": subquery
            }
            
            evaluated_tables.append(table_result)
            
            # 🚀 早停逻辑
            if is_relevant:
                consecutive_failures = 0  # 重置连续失败计数
                if task_logger:
                    task_logger.info(f"  {i+1}/{len(selected_nodes)} {node}: ✅ 相关 (PageRank: {pagerank_scores[node]:.6f})")
            else:
                consecutive_failures += 1
                if task_logger:
                    task_logger.info(f"  {i+1}/{len(selected_nodes)} {node}: ❌ 不相关 (PageRank: {pagerank_scores[node]:.6f}) [连续失败: {consecutive_failures}]")
                
                # 检查早停条件
                if (consecutive_failures >= max_consecutive_failures and 
                    i + 1 >= min_evaluations):  # 至少评估min_evaluations个表
                    
                    remaining_count = len(selected_nodes) - (i + 1)
                    if task_logger:
                        task_logger.info(f"🛑 早停触发: 连续{consecutive_failures}个高分表不相关，跳过剩余{remaining_count}个表")
                        task_logger.info(f"📝 跳过的表不会添加到最终JSON结果中")
                    
                    # 🚀 优化：不再为跳过的表添加记录，只记录实际经过LLM判断的表
                    break
                
        except Exception as e:
            logger.error(f"评估表 {node} 时出错: {e}")
            if task_logger:
                task_logger.error(f"评估表 {node} 时出错: {e}")
            
            # 添加错误结果
            error_result = {
                "think": f"PageRank subgraph evaluation error: {str(e)}",
                "answer": "N",
                "columns": [],
                "table name": node,
                "score": 0.0,
                "pagerank_score": pagerank_scores.get(node, 0.0),
                "subgraph_rank": i + 1,
                "selection_method": "pagerank_subgraph_error",
                "subquery": subquery
            }
            evaluated_tables.append(error_result)
            consecutive_failures += 1
    
    if task_logger:
        relevant_count = sum(1 for r in evaluated_tables if r["answer"] == "Y")
        total_evaluated = len(evaluated_tables)
        total_candidates = len(selected_nodes)
        skipped_count = total_candidates - total_evaluated
        task_logger.info(f"✅ PageRank子图评估完成: {relevant_count}/{total_evaluated} 个表相关 (跳过{skipped_count}个，未加入结果)")
    
    return evaluated_tables


def expand_via_pagerank_subgraph(subquery, working_graphs, current_layer_nodes,
                                table_descriptions, rerank_components, 
                                embedding_model, task, use_description=True,
                                use_coverage_bonus=False, coverage_beta=0.3,
                                task_logger=None, processed_tables=None,
                                example_root=None, current_example_id=None,
                                enable_batch_rerank=False, batch_size=10,
                                conditional_manager=None, query_context=None, workload_weight=0.1,
                                          edge_workload_weight=None, node_workload_weight=None,
                                enable_graph_topology=True):
    """
    基于PageRank的子图扩展主函数
    
    Args:
        subquery: 当前子查询
        working_graphs: 工作图字典
        current_layer_nodes: 当前层节点集合
        table_descriptions: 表描述字典
        rerank_components: LLM组件
        embedding_model: 嵌入模型
        task: 原始任务
        use_description: 是否使用表描述
        use_coverage_bonus: 是否使用Coverage Bonus增强多样性 (新增)
        coverage_beta: Coverage Bonus权重系数 (新增)
        task_logger: 任务日志记录器
        enable_batch_rerank: 是否启用批量rerank判断（新增）
        batch_size: 批量判断的批次大小（新增）
        
    Returns:
        List[Dict]: 所有数据库的评估结果列表
    """
    # 🔬 处理消融实验参数
    if edge_workload_weight is None:
        edge_workload_weight = workload_weight
    if node_workload_weight is None:
        node_workload_weight = workload_weight
    if task_logger:
        task_logger.info("")
        task_logger.info("=" * 80)
        task_logger.info("🔍 PageRank子图搜索详情:")
        task_logger.info("=" * 80)
        task_logger.info(f"📥 输入信息:")
        task_logger.info(f"  🎯 子查询: {subquery}")
        task_logger.info(f"  📊 工作图数量: {len(working_graphs)}")
        for db_name, graph in working_graphs.items():
            task_logger.info(f"    - {db_name}: {len(graph.nodes())} 节点, {len(graph.edges())} 边")
        task_logger.info(f"  📍 当前层节点数: {len(current_layer_nodes)}")
        if current_layer_nodes and len(current_layer_nodes) <= 10:
            task_logger.info(f"  📝 当前层节点: {list(current_layer_nodes)}")
        task_logger.info(f"  🔧 使用描述: {use_description}")
        task_logger.info(f"  🎲 Coverage Bonus: {use_coverage_bonus}")
        if processed_tables:
            task_logger.info(f"  ⏭️ 已处理表数: {len(processed_tables)}")
        task_logger.info("-" * 80)
    
    all_subquery_results = []
    
    # 对每个相关数据库运行PageRank
    for db_name, graph in working_graphs.items():
        if not graph.nodes():
            if task_logger:
                task_logger.warning(f"⚠️ 数据库 {db_name} 图为空，跳过")
            continue
            
        if task_logger:
            task_logger.info(f"📊 在数据库 {db_name} 上运行PageRank ({len(graph.nodes())} 个节点, {len(graph.edges())} 条边)")
        
        try:
            # 1. 运行个性化PageRank（带起始节点偏置 + 🆕 workload_prior）
            pagerank_scores, weight_details = run_personalized_pagerank_for_subquery(
                graph, subquery, embedding_model, table_descriptions, 
                current_layer_nodes, start_node_bias_weight=2.0, task_logger=task_logger,
                conditional_manager=conditional_manager, query_context=query_context,
                workload_weight=workload_weight
            )
            
            # 2. 选择子图
            subgraph, selected_nodes = select_subgraph_by_pagerank(
                graph, pagerank_scores, selection_strategy="adaptive", 
                use_coverage_bonus=use_coverage_bonus, coverage_beta=coverage_beta,
                task_logger=task_logger
            )
            
            # 🚀 重要：过滤掉已经评估过的表，避免重复判断
            if processed_tables:
                original_count = len(selected_nodes)
                selected_nodes = [node for node in selected_nodes if node not in processed_tables]
                filtered_count = original_count - len(selected_nodes)
                
                if task_logger and filtered_count > 0:
                    task_logger.info(f"🔄 数据库 {db_name}: 过滤掉{filtered_count}个已评估的表，剩余{len(selected_nodes)}个候选表")
                
                # 如果过滤后没有候选表，跳过这个数据库
                if not selected_nodes:
                    if task_logger:
                        task_logger.info(f"⚠️ 数据库 {db_name} 过滤后无候选表，跳过")
                    continue
                
                # 重新构建子图（只包含未评估的表）
                subgraph = graph.subgraph(selected_nodes).copy()
            
            if task_logger:
                task_logger.info(f"📈 数据库 {db_name}: PageRank选择了 {len(selected_nodes)} 个候选表")
            
            # 3. 评估子图中的表（包含完整的列schema信息）
            subgraph_results = evaluate_subgraph_tables(
                subgraph, selected_nodes, subquery, task,
                rerank_components, table_descriptions, pagerank_scores, use_description, task_logger,
                example_root=example_root, current_example_id=current_example_id,
                enable_batch_rerank=enable_batch_rerank, batch_size=batch_size,
                working_graphs=working_graphs,
                enable_graph_topology=enable_graph_topology
            )
            
            # 为结果添加数据库信息
            for result in subgraph_results:
                result['db_name'] = db_name
            
            all_subquery_results.extend(subgraph_results)
            
            if task_logger:
                relevant_count = sum(1 for r in subgraph_results if r["answer"] == "Y")
                task_logger.info(f"✅ 数据库 {db_name}: {relevant_count}/{len(subgraph_results)} 个表被判断为相关")
                
        except Exception as e:
            logger.error(f"数据库 {db_name} PageRank处理失败: {e}")
            if task_logger:
                task_logger.error(f"数据库 {db_name} PageRank处理失败: {e}")
    
    if task_logger:
        total_relevant = sum(1 for r in all_subquery_results if r["answer"] == "Y")
        task_logger.info("")
        task_logger.info("📤 PageRank子图搜索输出结果:")
        task_logger.info("-" * 80)
        task_logger.info(f"📊 总计: {total_relevant}/{len(all_subquery_results)} 个表被判断为相关")
        task_logger.info("")
        if all_subquery_results:
            task_logger.info("相关表详情:")
            relevant_tables = [r for r in all_subquery_results if r["answer"] == "Y"]
            for idx, result in enumerate(relevant_tables, 1):
                table_name = result.get('table name', 'unknown')
                db_name = result.get('db_name', 'unknown')
                pagerank_score = result.get('pagerank_score', 0.0)
                subgraph_rank = result.get('subgraph_rank', 0)
                task_logger.info(f"  {idx}. {table_name}")
                task_logger.info(f"      数据库: {db_name}")
                task_logger.info(f"      PageRank分数: {pagerank_score:.6f} (排名: #{subgraph_rank})")
        task_logger.info("=" * 80)
    
    return all_subquery_results


# 第一个重复的if __name__ == '__main__':块已删除，统一使用下面的完整版本

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, default="lite")
    parser.add_argument('--db_path', type=str, default="examples_lite")
    parser.add_argument('--linked_json_pth', type=str, default=None)
    parser.add_argument('--reduce_col', action="store_true")
    parser.add_argument('--gold_tb_pth', type=str, default=None)
    parser.add_argument('--score_threshold', type=float, default=0.5, help='Rerank score threshold for table selection')
    parser.add_argument('--profiling_dir', type=str, default=None, help='目录，存放profiling结果json')
    parser.add_argument('--gen_desc', action="store_true", help='遍历schema json生成表级中文描述')
    parser.add_argument('--model', type=str, default="Qwen3-235B-A22B-Instruct-2507-FP8", help='描述生成所用LLM')
    parser.add_argument('--desc_output', type=str, default="data/table_descriptions.json", help='保存表描述的json路径')
    parser.add_argument('--gen_profiling', action="store_true", help='生成/更新profiling json')
    parser.add_argument('--build_graph', action="store_true", help='根据profiling+描述构建schema graph')
    parser.add_argument('--graph_output', type=str, default="schema_graph.gpickle", help='schema graph 保存路径')
    parser.add_argument('--desc_sim_th', type=float, default=0.2, help='表描述语义相似度阈值')
    parser.add_argument('--minhash_th', type=float, default=0.8, help='字段值MinHash相似度阈值')
    parser.add_argument('--prompts_file', type=str, default=None, help='单个 prompts.txt 路径，直接批量生成表描述')
    parser.add_argument('--use_desc_in_rerank', action='store_true', default=True, help='Whether to use table descriptions in rerank (default: True)')
    parser.add_argument('--no_desc_in_rerank', action='store_true', help='Disable table descriptions in rerank')
    parser.add_argument('--use_semantic_graph_search', action='store_true', help='Use semantic graph search algorithm instead of traditional method')
    parser.add_argument('--database_graphs_dir', type=str, default='database_graphs', help='Directory containing database graph files')
    parser.add_argument('--build_db_graphs', action='store_true', help='按数据库分组构建Schema Graph')
    parser.add_argument('--db_graphs_output', type=str, default='database_graphs', help='数据库图输出目录')
    parser.add_argument('--analyze_db_graphs', action='store_true', help='分析已构建的数据库图')
    parser.add_argument('--use_desc_in_graphs', action='store_true', default=True, help='在图构建中是否使用表描述')
    parser.add_argument('--no_desc_in_graphs', action='store_true', help='禁用图构建中的表描述（只使用MinHash+外键）')
    parser.add_argument('--gen_embeddings', action='store_true', help='批量计算并保存表语义向量')
    parser.add_argument('--force_recompute_embeddings', action='store_true', help='强制重新计算语义向量（即使已存在）')
    parser.add_argument('--gen_prompts', action='store_true', help='生成/重建所有样本的prompts.txt文件')
    parser.add_argument('--force_rebuild_prompts', action='store_true', help='强制重建prompts.txt（即使已存在）')
    parser.add_argument('--test_search_isolation', action='store_true', help='测试搜索空间独立性')
    parser.add_argument('--analyze_expansion_paths', action='store_true', help='分析扩展路径文件，生成统计摘要')
    parser.add_argument('--expansion_paths_dir', type=str, default=None, help='扩展路径文件所在目录')
    parser.add_argument('--use_subquery_decomposition', action='store_true', help='使用子查询分解方法进行表搜索')
    parser.add_argument('--enable_topk_rerank', action='store_true', help='对Top-K预选表进行LLM判断（默认直接标记为相关）')
    parser.add_argument('--top_k_preselection', type=int, default=5, help='预选的Top-K表数量（默认5）')
    parser.add_argument('--use_coverage_bonus', action='store_true', help='使用Coverage Bonus增强表选择多样性（默认False）')
    parser.add_argument('--coverage_beta', type=float, default=0.3, help='Coverage Bonus权重系数（默认0.3）')
    parser.add_argument('--enable_sql_validation', action='store_true', help='启用SQL验证与迭代优化（Step 3）')
    parser.add_argument('--max_validation_iterations', type=int, default=3, help='SQL验证最大迭代次数（默认3）')
    parser.add_argument('--enable_batch_rerank', action='store_true', help='启用批量rerank判断（PageRank阶段多个表一起判断）')
    parser.add_argument('--batch_size', type=int, default=10, help='批量判断的批次大小（默认10）')
    parser.add_argument('--max_samples_debug', type=int, help='调试模式下最大处理样本数')
    # 🆕 Workload Evolution 参数
    parser.add_argument('--use_workload_evolution', action='store_true', help='启用 Workload Evolution（query解析+conditional functions）')
    parser.add_argument('--workload_stats', type=str, default='graph_evolution_data/workload_stats.json', help='Workload 统计文件路径')
    parser.add_argument('--workload_weight', type=float, default=1.0, help='统一的 Workload boost 权重系数（默认1.0，当未指定分离参数时使用）')
    
    # 🔬 消融实验：分别控制边增强和节点先验
    parser.add_argument('--edge_workload_weight', type=float, default=None, 
                        help='λ (lambda): 边增强权重（Join + Table Cooccur），用于离线图增强。设为0禁用边增强。')
    parser.add_argument('--node_workload_weight', type=float, default=None, 
                        help='γ (gamma): 节点先验权重（Predicate + Aggregation），用于在线personalization。设为0禁用节点先验。')
    
    # 🆕 Batch 拓扑增强参数
    parser.add_argument('--disable_graph_topology', action='store_true',
                        help='禁用batch pruning中的图拓扑关系注入（默认启用）')

    # 🆕 IND/AIND 隐式外键检测参数
    parser.add_argument('--enable_ind_aind', action='store_true', default=True,
                        help='启用IND/AIND隐式外键检测（默认启用）')
    parser.add_argument('--disable_ind_aind', action='store_true',
                        help='禁用IND/AIND检测（覆盖 --enable_ind_aind）')
    parser.add_argument('--aind_threshold', type=float, default=0.95,
                        help='AIND置信度阈值（默认0.95，取值0~1）')
    parser.add_argument('--ind_sample_limit', type=int, default=50000,
                        help='IND/AIND检测时每列最大采样行数（默认50000）')
    
    args = parser.parse_args()
    
    # 🔬 处理消融实验参数：如果没有指定分离参数，使用统一的 workload_weight
    if args.edge_workload_weight is None:
        args.edge_workload_weight = args.workload_weight
    if args.node_workload_weight is None:
        args.node_workload_weight = args.workload_weight
    
    # 📊 显示消融实验配置
    if args.use_workload_evolution:
        logger.info("🔬 Workload Evolution 消融实验配置:")
        logger.info(f"   λ (edge_workload_weight):  {args.edge_workload_weight}  ← 边增强（Join + Table Cooccur）")
        logger.info(f"   γ (node_workload_weight):  {args.node_workload_weight}  ← 节点先验（Predicate + Aggregation）")
    
    # Handle description usage flag
    if args.no_desc_in_rerank:
        args.use_desc_in_rerank = False
    
    # Handle graph description usage flag
    if args.no_desc_in_graphs:
        args.use_desc_in_graphs = False
    dictionaries, task_dict = get_dictionary(args.db_path, args.task)
    if args.gold_tb_pth:
        with open(args.gold_tb_pth) as f:
            gold = [json.loads(i) for i in f]
    
    # 生成 prompts.txt 文件
    if args.gen_prompts:
        print("[Main] Generating prompts.txt files …")
        generate_all_prompts_txt(args.db_path, force_rebuild=args.force_rebuild_prompts)
    
    if args.gen_profiling and args.profiling_dir is not None:
        print(f"[Main] Profiling dir: {args.profiling_dir}")
        if not os.path.exists(args.profiling_dir) or len(os.listdir(args.profiling_dir)) == 0:
            print(f"[Main] No profiling jsons found, generating from {args.db_path} …")
            save_all_table_profiling_from_json(args.db_path, args.profiling_dir)
        else:
            print("[Main] Profiling json 已存在，跳过生成。")

    if args.gen_desc:
        print("[Main] Generating table descriptions …")
        generate_table_descriptions(args.db_path, args.model, args.desc_output)

    # 批量计算语义向量
    if args.gen_embeddings:
        print("[Main] Computing and saving table embeddings …")
        batch_compute_and_save_embeddings(
            example_root=args.db_path,
            force_recompute=args.force_recompute_embeddings
        )

    # 构建图
    if args.build_graph and args.profiling_dir:
        print("[Main] Building schema graph …")
        
        # 收集外键关系（如果提供了example_root）
        fk_relations = None
        if args.db_path:
            print("[Main] Collecting foreign key relationships...")
            fk_relations = collect_foreign_key_relationships(args.db_path)
            # 将所有数据库的外键关系合并为一个列表
            all_fk_relations = []
            for fk_list in fk_relations.values():
                all_fk_relations.extend(fk_list)
            fk_relations = all_fk_relations
            print(f"[Main] Collected {len(fk_relations)} total foreign key relationships")
        
        build_schema_graph_from_profiling(
            profiling_dir=args.profiling_dir,
            output_path=args.graph_output,
            desc_sim_th=args.desc_sim_th,
            minhash_th=args.minhash_th,
            example_root=args.db_path,
            fk_relations=fk_relations  # 传递外键关系
        )

    # 按数据库分组构建图
    if args.build_db_graphs:
        print("[Main] Building database-specific schema graphs with task isolation...")
        
        # 收集外键关系
        print("[Main] Collecting foreign key relationships...")
        fk_relations = collect_foreign_key_relationships(args.db_path)
        
        print("[Main] Collecting database groups with task isolation...")
        database_groups = collect_tables_by_database(args.db_path)
        
        if database_groups:
            print(f"[Main] Found {len(database_groups)} task-isolated databases")
            
            # 统计各类型数据库
            local_dbs = [k for k in database_groups.keys() if k.startswith('local')]
            bq_dbs = [k for k in database_groups.keys() if k.startswith('bq')]
            sf_dbs = [k for k in database_groups.keys() if k.startswith('sf')]
            ga_dbs = [k for k in database_groups.keys() if k.startswith('ga')]
            
            print(f"[Main] Database breakdown:")
            print(f"  - Local (SQLite): {len(local_dbs)} databases")
            print(f"  - BigQuery: {len(bq_dbs)} databases") 
            print(f"  - Snowflake: {len(sf_dbs)} databases")
            print(f"  - Google Analytics: {len(ga_dbs)} databases")
            
            # 构建图
            # 处理IND/AIND参数（--disable_ind_aind 可覆盖默认值）
            enable_ind_aind = args.enable_ind_aind and not args.disable_ind_aind
            if enable_ind_aind:
                print(f"[Main] 🔗 IND/AIND detection enabled (threshold: {args.aind_threshold}, sample_limit: {args.ind_sample_limit})")
            else:
                print("[Main] ⚠️  IND/AIND detection disabled")

            print("[Main] Building graphs for all task-isolated databases...")
            database_graphs = build_database_specific_graphs(
                database_groups=database_groups,
                output_dir=args.db_graphs_output,
                desc_sim_th=args.desc_sim_th,
                minhash_th=args.minhash_th,
                use_descriptions=args.use_desc_in_graphs,
                profiling_dir=args.profiling_dir,
                example_root=args.db_path,
                fk_relations=fk_relations,  # 传递显式外键关系
                enable_ind_aind=enable_ind_aind,
                aind_threshold=args.aind_threshold,
                ind_sample_limit=args.ind_sample_limit
            )
            print(f"[Main] Successfully built {len(database_graphs)} database-specific graphs")
            
            # 验证local数据库图的构建
            local_graphs = {k: v for k, v in database_graphs.items() if k.startswith('local')}
            if local_graphs:
                print(f"[Main] ✅ Successfully built {len(local_graphs)} SQLite database graphs:")
                for db_name, graph in list(local_graphs.items())[:5]:  # 显示前5个
                    print(f"  - {db_name}: {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges")
                if len(local_graphs) > 5:
                    print(f"  ... and {len(local_graphs) - 5} more SQLite graphs")
            else:
                print("[Main] ⚠️ No SQLite database graphs were built")
                
        else:
            print("[Main] ❌ No database groups found - check data collection process")

    # 分析数据库图
    if args.analyze_db_graphs:
        print("[Main] Analyzing database graphs …")
        analyze_database_graphs(args.db_graphs_output)
    
    # 测试搜索空间独立性
    if args.test_search_isolation:
        print("[Main] Testing search space isolation …")
        test_search_space_isolation(args.db_path, args.db_graphs_output)
    
    # 分析扩展路径
    if args.analyze_expansion_paths:
        print("[Main] Analyzing expansion paths …")
        paths_dir = args.expansion_paths_dir or os.environ.get("GRAPHLINK_EXPANSION_PATHS_DIR", "data/expansion_paths")
        analyze_expansion_paths_summary(paths_dir)

    # Schema Linking相关功能（仅在需要时执行）
    if args.linked_json_pth is not None:
        if not os.path.exists(args.linked_json_pth):
            print("[Main] Performing schema linking ...")
            
            ask_model_sl(args.db_path, args.linked_json_pth, args.score_threshold, args.use_desc_in_rerank,
                    args.use_semantic_graph_search, args.database_graphs_dir,
                    args.use_subquery_decomposition, args.max_samples_debug, 
                    args.enable_topk_rerank, args.top_k_preselection,
                    args.use_coverage_bonus, args.coverage_beta,
                    args.enable_sql_validation, args.max_validation_iterations,
                    args.enable_batch_rerank, args.batch_size,
                    args.use_workload_evolution, args.workload_stats, args.workload_weight,
                    args.edge_workload_weight, args.node_workload_weight,  # 🔬 消融实验参数
                    enable_graph_topology=not args.disable_graph_topology,
                    task=args.task)
        
        if args.gold_tb_pth:  # 只有提供了gold数据才计算metrics
            print("[Main] Computing schema linking metrics ...")
            compute_metrics_sl(args.linked_json_pth, args.db_path)
        
        if args.reduce_col:  # 只有设置了reduce_col才执行DDL处理
            print("[Main] Reducing DDL ...")
            reduce_ddl(args.db_path, dictionaries, args.linked_json_pth, args.reduce_col)
