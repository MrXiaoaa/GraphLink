#!/usr/bin/env python3
"""
从examples_lite目录随机选K个表，生成N个问题和SQL

Usage:
    python generate_qa_from_tables.py \
        --input_path /path/to/examples_lite \
        --k_tables 3 \
        --n_questions 5 \
        --output generated_qa.json
"""

import os
import sys
import json
import csv
import random
import re
import logging
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, asdict, replace
import time

from graphlink.spider2_compat.sql import SqlEnv
from graphlink.spider2_compat.chat import GPTChat
from graphlink.spider2_compat.prompt import Prompts
from graphlink.spider2_compat.utils import get_api_name

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class QAPair:
    """问题-SQL对的数据结构
    
    输出字段要求：
    - query: 问题
    - tables: 使用的表列表
    - relevant_columns: 相关列
    - sql: SQL语句
    - data: 执行结果
    - size: 表schema文本长度
    - success: 能否正确执行（bool）
    """
    query: str                          # 问题
    tables: List[str]                   # 使用的表列表
    relevant_columns: List[str]         # 相关列
    sql: str                            # SQL语句
    data: Optional[str]                 # 执行结果
    size: int                           # 表schema文本长度
    success: bool                       # 能否正确执行
    error_message: Optional[str] = None # 错误信息（如果失败）
    execution_time: Optional[float] = None  # 执行时间
    source_example: Optional[str] = None    # 来源example


class TableBasedQuestionGenerator:
    """基于examples_lite目录的问题生成器"""
    
    def __init__(self, input_path: str, llm_client: GPTChat, sql_env: SqlEnv, dir_prefix: str = None):
        """
        Args:
            input_path: examples_lite 父目录路径
            llm_client: GPTChat客户端（已初始化）
            sql_env: SQL执行环境
            dir_prefix: 子目录名前缀过滤（例如 "local" 只处理 local 开头的目录）
        """
        self.input_path = input_path
        self.llm_client = llm_client
        self.sql_env = sql_env
        self.prompts = Prompts()  # 初始化 Prompts 实例，用于获取数据库特定的提示
        self.dir_prefix = dir_prefix  # 用于过滤子目录的前缀
        
        # 为每个 example 保存表描述（来自 table_descriptions.json）
        self.table_descriptions_map: Dict[str, Dict[str, str]] = {}
        
        # 扫描所有子目录，加载所有候选表（从 table_descriptions.json 中读取）
        self.example_dirs, self.all_tables_map = self._scan_all_examples()
        
        logger.info(f"Scanned {len(self.example_dirs)} example directories")
        logger.info(f"Found {sum(len(tables) for tables in self.all_tables_map.values())} total candidate tables")
    
    def _scan_all_examples(self) -> Tuple[List[str], Dict[str, List[str]]]:
        """
        扫描examples_lite下的所有子目录
        
        Returns:
            (example_dirs, all_tables_map)
            example_dirs: 所有example目录的列表
            all_tables_map: {example_dir: [table1, table2, ...]}
        """
        
        example_dirs: List[str] = []
        all_tables_map: Dict[str, List[str]] = {}
        
        if not os.path.exists(self.input_path):
            raise FileNotFoundError(f"Input path not found: {self.input_path}")
        
        if self.dir_prefix:
            logger.info(f"Scanning examples in: {self.input_path} (filtering by prefix: '{self.dir_prefix}')")
        else:
            logger.info(f"Scanning examples in: {self.input_path}")
        
        # 遍历所有子目录
        for item in sorted(os.listdir(self.input_path)):
            item_path = os.path.join(self.input_path, item)
            
            if not os.path.isdir(item_path):
                continue
            
            # 应用前缀过滤
            if self.dir_prefix and not item.startswith(self.dir_prefix):
                logger.debug(f"Skipping {item} (does not match prefix '{self.dir_prefix}')")
                continue
            
            # 始终从 table_descriptions.json 中读取候选表（由离线流程预筛选）
            json_file = os.path.join(item_path, 'table_descriptions.json')
            if not os.path.exists(json_file):
                logger.debug(f"Skipping {item}: no table_descriptions.json")
                continue
            
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    table_descriptions = json.load(f)
                
                tables = list(table_descriptions.keys())
                if not tables:
                    logger.debug(f"Skipping {item}: empty table_descriptions.json")
                    continue
                
                # 记录表描述，后续用 LLM 选表 & 提问题
                self.table_descriptions_map[item_path] = table_descriptions
            
            except Exception as e:
                logger.warning(f"Failed to load {json_file}: {e}")
                continue
            
            if len(tables) > 0:
                example_dirs.append(item_path)
                all_tables_map[item_path] = tables
                logger.info(f"  - {item}: {len(tables)} tables")
            else:
                logger.debug(f"Skipping {item}: no tables found")
        
        return example_dirs, all_tables_map
    
    def _get_all_tables_from_sqlite(self, example_path: str) -> List[str]:
        """
        从SQLite文件读取所有表名（不使用预筛选的table_descriptions.json）
        
        Args:
            example_path: example目录路径
        
        Returns:
            表名列表
        """
        import sqlite3
        import glob
        
        # 查找SQLite文件
        sqlite_files = glob.glob(os.path.join(example_path, '*.sqlite')) + \
                       glob.glob(os.path.join(example_path, '*.db'))
        
        if not sqlite_files:
            return []
        
        sqlite_file = sqlite_files[0]
        tables = []
        
        try:
            connection = sqlite3.connect(sqlite_file)
            cursor = connection.cursor()
            
            # 获取所有表名（排除系统表）
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'")
            tables = [row[0] for row in cursor.fetchall()]
            
            connection.close()
            
            logger.debug(f"✅ Read {len(tables)} tables from {os.path.basename(sqlite_file)}")
            
        except Exception as e:
            logger.debug(f"Failed to read SQLite file {sqlite_file}: {e}")
        
        return tables
    
    def generate_qa_pairs(self, k_tables: int = 3, n_questions: int = 5, output_file: Optional[str] = None) -> List[QAPair]:
        """
        生成问题和SQL
        
        遍历所有example目录，每个目录生成n_questions个问题
        每个问题从该目录的表中随机选择k_tables个表
        
        Args:
            k_tables: 每个问题随机选择的表数量
            n_questions: 每个example生成的问题数量
            output_file: 输出JSON文件路径（如果提供，会实时写入）
        
        Returns:
            QAPair列表
        """
        
        qa_pairs = []
        
        # 如果指定了输出文件，初始化JSON文件
        if output_file:
            self._init_json_file(output_file)
            logger.info(f"[Save] Output file initialized: {output_file} (JSONL format)")
        
        # 遍历所有子目录模式
        logger.info(f"="*80)
        logger.info(f"Starting QA generation: Traverse all {len(self.example_dirs)} examples")
        logger.info(f"  Questions per example: {n_questions}")
        logger.info(f"  Tables per question: {k_tables}")
        if output_file:
            logger.info(f"  Real-time save to: {output_file} (JSONL format)")
        logger.info(f"="*80)
        
        for example_idx, selected_example in enumerate(self.example_dirs, 1):
            example_name = os.path.basename(selected_example)
            logger.info(f"\n{'='*80}")
            logger.info(f"Processing example {example_idx}/{len(self.example_dirs)}: {example_name}")
            logger.info(f"{'='*80}")
            
            try:
                # 为每个example生成n_questions个问题（每个问题随机选择k_tables个表）
                example_qa_pairs = self._generate_for_single_example(
                    selected_example, example_name, k_tables, n_questions, output_file
                )
                qa_pairs.extend(example_qa_pairs)
                
                logger.info(f"✅ Completed {example_name}: Generated {len(example_qa_pairs)} QA pairs")
            
            except Exception as e:
                logger.error(f"[Error] Failed to process example {example_name}: {e}")
                import traceback
                logger.error(traceback.format_exc())
                # 继续处理下一个example，不终止程序
                continue
        
        success_count = sum(1 for q in qa_pairs if q.success)
        logger.info(f"\n{'='*80}")
        logger.info(f"Generation completed!")
        logger.info(f"Total generated: {len(qa_pairs)} QA pairs")
        logger.info(f"Successful: {success_count}")
        logger.info(f"Failed: {len(qa_pairs) - success_count}")
        logger.info(f"{'='*80}")
        
        # 如果指定了输出文件，确保最终格式正确
        if output_file:
            self._finalize_json_file(output_file)
            logger.info(f"[Save] Finalized JSONL file: {output_file}")
        
        return qa_pairs
    
    def _generate_for_single_example(self, 
                                     selected_example: str, 
                                     example_name: str,
                                     k_tables: int,
                                     n_questions: int,
                                     output_file: Optional[str] = None) -> List[QAPair]:
        """
        为单个example生成QA对（带错误处理，不会终止程序）
        
        Args:
            selected_example: example目录路径
            example_name: example名称
            k_tables: 每次选择的表数量
            n_questions: 生成的问题数量
        
        Returns:
            QAPair列表（即使出错也返回已生成的部分）
        """
        
        qa_pairs = []
        
        try:
            # 检查是否有可用表
            if selected_example not in self.all_tables_map:
                logger.warning(f"[Skip] {example_name}: No tables found in all_tables_map")
                return qa_pairs
            
            available_tables = self.all_tables_map[selected_example]
            if len(available_tables) == 0:
                logger.warning(f"[Skip] {example_name}: No available tables")
                return qa_pairs
            
            # 为这个example生成n_questions个问题
            for q_idx in range(n_questions):
                try:
                    logger.info(f"[{example_name}] Generating question {q_idx+1}/{n_questions}")
                    
                    # 1. 使用 LLM 从 table_descriptions.json 中选择 K 个表并提出问题
                    question_hint: Optional[str] = None
                    selected_tables: Optional[List[str]] = None
                    
                    table_descriptions = self.table_descriptions_map.get(selected_example)
                    if table_descriptions:
                        # 最多重试 2 次，不再回退到随机选表，以避免“超纲问题”
                        max_select_attempts = 2
                        attempt = 0
                        while attempt < max_select_attempts and not selected_tables:
                            attempt += 1
                            try:
                                logger.info(f"[{example_name}] LLM table selection attempt {attempt}/{max_select_attempts}...")
                                selected_tables, question_hint = self._select_tables_and_question_with_llm(
                                    table_descriptions, available_tables, k_tables, example_name
                                )
                            except Exception as e:
                                logger.warning(f"[{example_name}] LLM-based table selection failed on attempt {attempt}: {e}")
                                selected_tables, question_hint = None, None
                        
                        if not selected_tables:
                            logger.warning(
                                f"[{example_name}] LLM failed to select tables after {max_select_attempts} attempts; "
                                "skipping this question instead of falling back to random tables."
                            )
                            continue  # 直接跳过这一题，避免随机表导致超纲问题
                        
                        # 保证选表数量不超过 k_tables
                        if len(selected_tables) > k_tables:
                            selected_tables = selected_tables[:k_tables]
                        k_actual = len(selected_tables)
                    else:
                        # 如果没有 table_descriptions（极少数情况），只能退回到旧逻辑：随机选表
                        logger.warning(
                            f"[{example_name}] No table_descriptions found; falling back to random table selection."
                        )
                        k_actual = min(k_tables, len(available_tables))
                        selected_tables = random.sample(available_tables, k_actual)
                        question_hint = None
                    
                    logger.info(f"[{example_name}] Selected {k_actual} tables:")
                    for j, table in enumerate(selected_tables, 1):
                        logger.info(f"  {j}. {table}")
                    
                    # 2. 读取表的schema
                    logger.info(f"[{example_name}] Loading table schemas...")
                    table_schemas = self._load_table_schemas(selected_example, selected_tables)
                    logger.info(f"[{example_name}] Loaded {len(table_schemas)} schemas")
                    
                    if len(table_schemas) == 0:
                        logger.warning(f"[{example_name}] No schemas loaded, skipping this question")
                        continue
                    
                    # 3. 构建prompt（可选地注入 LLM 规划出的问题）
                    logger.info(f"[{example_name}] Building prompt...")
                    prompt = self._build_generation_prompt(
                        selected_tables,
                        table_schemas,
                        example_name,
                        n_questions=1,
                        question_hint=question_hint,
                    )
                    
                    logger.debug(f"[{example_name}] Prompt length: {len(prompt)} characters")
                    
                    # 4. 调用LLM生成
                    logger.info(f"[{example_name}] Calling LLM...")
                    qa_pairs_raw = self._call_llm_generate(prompt)
                    
                    if not qa_pairs_raw:
                        logger.warning(f"[{example_name}] No QA pairs generated by LLM for question {q_idx+1}")
                        continue
                    
                    # 5. 验证SQL并补充信息（带重试和验证机制）
                    for qa_raw in qa_pairs_raw:
                        try:
                            logger.info(f"[{example_name}] Raw QA pair received:")
                            logger.info(f"  Question: {qa_raw.get('question', 'N/A')[:100]}...")
                            logger.info(f"  SQL: {qa_raw.get('sql', 'N/A')[:200]}...")
                            
                            qa_pair = self._validate_and_enrich(qa_raw, table_schemas, example_name)
                            
                            # 如果SQL执行失败，尝试修复（最多3次）
                            if not qa_pair.success:
                                logger.warning(f"[{example_name}] SQL execution failed, attempting to fix...")
                                qa_pair = self._retry_fix_sql(qa_pair, qa_raw, table_schemas, example_name, max_retries=3)
                            
                            # # 如果SQL执行成功，让LLM验证结果
                            # if qa_pair.success and qa_pair.data:
                            #     logger.info(f"[{example_name}] SQL executed successfully, validating result with LLM...")
                            #     qa_pair = self._validate_result_with_llm(qa_pair, qa_raw, table_schemas, example_name)
                            
                            qa_pairs.append(qa_pair)
                            
                            # 实时写入JSONL文件
                            if output_file:
                                self._append_to_json_file(output_file, qa_pair)
                            
                            if qa_pair.success:
                                logger.info(f"[{example_name}] ✅ Question {q_idx+1} generated successfully")
                            else:
                                logger.warning(f"[{example_name}] ⚠️  Question {q_idx+1} SQL execution failed after retries: {qa_pair.error_message}")
                        
                        except Exception as e:
                            logger.error(f"[{example_name}] Error validating QA pair: {e}")
                            import traceback
                            logger.debug(traceback.format_exc())
                            continue
                
                except Exception as e:
                    logger.error(f"[{example_name}] Error generating question {q_idx+1}: {e}")
                    import traceback
                    logger.debug(traceback.format_exc())
                    continue  # 继续生成下一个问题
        
        except Exception as e:
            logger.error(f"[Error] Failed to process example {example_name}: {e}")
            import traceback
            logger.error(traceback.format_exc())
            # 返回已生成的部分，不抛出异常
        
        return qa_pairs
    
    def _load_table_schemas(self, example_path: str, table_names: List[str]) -> Dict[str, str]:
        """
        从DDL.csv和JSON文件读取表的schema
        
        参考 schema_linking_retrieve.py 的 get_table_info_from_directory 方法
        
        Args:
            example_path: example目录路径
            table_names: 表名列表（完整表名，如 bigquery-public-data.google_analytics_sample.ga_sessions_20170507）
        
        Returns:
            {table_name: schema_text}
        """
        
        schemas = {}
        
        # 遍历example目录下的所有子目录，查找DDL.csv
        # 目录结构可能是：example_path/item/schema_item/DDL.csv
        # 或者：example_path/item/DDL.csv（直接就是schema目录）
        
        def find_ddl_files(root_path, depth=0, max_depth=3):
            """递归查找所有DDL.csv文件"""
            ddl_files = []
            if depth > max_depth:
                return ddl_files
            
            try:
                for item in os.listdir(root_path):
                    item_path = os.path.join(root_path, item)
                    
                    # 跳过特殊目录和文件
                    if item in ['spider', 'output', '__pycache__', '.git']:
                        continue
                    
                    # 如果是目录，继续递归
                    if os.path.isdir(item_path):
                        ddl_files.extend(find_ddl_files(item_path, depth+1, max_depth))
                    # 如果是DDL.csv文件，记录
                    elif item == "DDL.csv":
                        ddl_files.append(root_path)
            
            except PermissionError:
                pass
            
            return ddl_files
        
        # 查找所有包含DDL.csv的目录
        schema_dirs = find_ddl_files(example_path)
        
        for schema_path in schema_dirs:
            ddl_path = os.path.join(schema_path, "DDL.csv")
            if not os.path.exists(ddl_path):
                continue
            
            # 读取 DDL.csv 获取表列表
            try:
                    with open(ddl_path, "r", newline="", encoding="utf-8", errors="ignore") as f:
                        reader = csv.reader(f)
                        header = next(reader)
                        
                        for row in reader:
                            if not row:
                                continue
                            
                            # 第一列是短表名（如 ga_sessions_20170127）
                            short_table_name = row[0].strip()
                            
                            # 构建完整表名（从目录结构推断）
                            schema_path_parts = schema_path.split(os.sep)
                            try:
                                example_idx = schema_path_parts.index(os.path.basename(example_path))
                                db_parts = schema_path_parts[example_idx+1:]
                                full_table_name = ".".join(db_parts + [short_table_name])
                            except ValueError:
                                # 如果找不到，尝试用目录名构建
                                parent_dir = os.path.basename(os.path.dirname(schema_path))
                                schema_dir = os.path.basename(schema_path)
                                full_table_name = f"{parent_dir}.{schema_dir}.{short_table_name}"
                            
                            # 检查是否是我们需要的表（支持完整表名或短表名匹配）
                            matched_full_name = None
                            for tn in table_names:
                                if tn == full_table_name or tn.endswith(f".{short_table_name}"):
                                    matched_full_name = tn
                                    break
                            
                            if matched_full_name is None:
                                continue
                            
                            # 使用短表名查找JSON文件
                            table_name_short = short_table_name
                            
                            # 尝试读取对应的 JSON 文件
                            schema_item_name = os.path.basename(schema_path)
                            json_files = [
                                os.path.join(schema_path, f"{table_name_short}.json"),
                                os.path.join(schema_path, f"{schema_item_name}.{table_name_short}.json")
                            ]
                            parent_dir = os.path.basename(os.path.dirname(schema_path))
                            if parent_dir:
                                json_files.append(
                                    os.path.join(schema_path, f"{parent_dir}.{table_name_short}.json")
                                )
                            
                            table_json = None
                            for json_file in json_files:
                                if os.path.exists(json_file):
                                    try:
                                        with open(json_file, 'r', encoding='utf-8') as jf:
                                            table_json = json.load(jf)
                                        break
                                    except Exception as e:
                                        logger.debug(f"Failed to load {json_file}: {e}")
                                        continue
                            
                            if table_json:
                                # 构建表信息字符串，格式与 prompts.txt 一致
                                schema_text = f"Table full name: {table_json.get('table_fullname', full_table_name)}\n"
                                
                                column_names = table_json.get('column_names', [])
                                column_types = table_json.get('column_types', [])
                                descriptions = table_json.get('description', [])
                                
                                for j in range(len(column_names)):
                                    col_name = column_names[j]
                                    col_type = column_types[j] if j < len(column_types) else "UNKNOWN"
                                    col_desc = descriptions[j] if j < len(descriptions) and descriptions[j] else ""
                                    
                                    if col_desc:
                                        schema_text += f"Column name: {col_name} Type: {col_type} Description: {col_desc}\n"
                                    else:
                                        schema_text += f"Column name: {col_name} Type: {col_type}\n"
                                
                                # 添加样本行（如果有）
                                if 'sample_rows' in table_json and table_json['sample_rows']:
                                    schema_text += f"Sample rows:\n{json.dumps(table_json['sample_rows'], ensure_ascii=False)}\n"
                                
                                schemas[matched_full_name] = schema_text.strip()
                            else:
                                logger.warning(f"JSON file not found for table {matched_full_name} (short: {table_name_short})")
                                # Fallback: 使用table_descriptions
                                schemas[matched_full_name] = self._get_fallback_schema(matched_full_name, example_path)
            
            except Exception as e:
                logger.error(f"Error reading DDL for {schema_path}: {e}")
                continue
        
        # 如果没有找到任何schema，尝试简单结构
        if not schemas:
            logger.info("No DDL.csv found, trying simple structure")
            schemas = self._load_schemas_simple_structure(example_path, table_names)
        
        # 对于没有找到schema的表，使用fallback
        for table_name in table_names:
            if table_name not in schemas:
                logger.warning(f"Schema not found for {table_name}, using fallback")
                schemas[table_name] = self._get_fallback_schema(table_name, example_path)
        
        return schemas
    
    def _build_schema_from_json(self, table_json: Dict, table_name: str) -> str:
        """
        从JSON构建schema字符串（参考 schema_linking_graph_1120.py 的格式）
        """
        # 构建表信息字符串
        schema_text = f"Table full name: {table_json.get('table_fullname', table_name)}\n"
        
        column_names = table_json.get('column_names', [])
        column_types = table_json.get('column_types', [])
        descriptions = table_json.get('description', [])
        
        for j in range(len(column_names)):
            col_name = column_names[j]
            col_type = column_types[j] if j < len(column_types) else "UNKNOWN"
            col_desc = descriptions[j] if j < len(descriptions) and descriptions[j] else ""
            
            if col_desc:
                schema_text += f"Column name: {col_name} Type: {col_type} Description: {col_desc}\n"
            else:
                schema_text += f"Column name: {col_name} Type: {col_type}\n"
        
        # 添加样本行（如果有，限制数量避免太长）
        if 'sample_rows' in table_json and table_json['sample_rows']:
            sample_rows = table_json['sample_rows']
            # 如果是list，只取前3行
            if isinstance(sample_rows, list):
                sample_rows = sample_rows[:3]
            schema_text += f"Sample rows:\n{sample_rows}\n"
        
        return schema_text.strip()
    
    def _load_schemas_simple_structure(self, example_path: str, table_names: List[str]) -> Dict[str, str]:
        """
        简单结构：从example_path根目录读取schema
        优先级：SQLite文件 → prompts.txt → JSON文件
        
        优先从SQLite读取原因：
        - prompts.txt是预筛选的，只包含部分表
        - 从SQLite读取可以访问所有表，生成更多样化的问题
        
        适用于local examples (local001, local002, etc.)
        """
        schemas = {}
        
        # 策略1: 从SQLite文件直接读取（优先！获取完整数据库信息）
        logger.info("Trying to read schemas from SQLite file...")
        schemas = self._load_schemas_from_sqlite(example_path, table_names)
        if schemas:
            logger.info(f"✅ Loaded {len(schemas)} schemas from SQLite file")
            return schemas
        
        # 策略2: 从prompts.txt解析（fallback，但这是预筛选的）
        prompts_file = os.path.join(example_path, 'prompts.txt')
        if os.path.exists(prompts_file):
            logger.warning("SQLite not found, falling back to prompts.txt (pre-filtered)")
            schemas = self._parse_schemas_from_prompts(prompts_file, table_names)
            if schemas:
                logger.info(f"✅ Loaded {len(schemas)} schemas from prompts.txt")
                return schemas
        
        # 策略3: 从JSON文件读取（最后的fallback）
        logger.warning("No SQLite or prompts.txt found, trying JSON files")
        for table_name in table_names:
            # 尝试不同的文件名格式
            short_name = table_name.split('.')[-1]  # 获取短表名
            json_files_to_try = [
                os.path.join(example_path, f"{table_name}.json"),
                os.path.join(example_path, f"{short_name}.json")
            ]
            
            table_json = None
            for json_file in json_files_to_try:
                if os.path.exists(json_file):
                    try:
                        with open(json_file, 'r', encoding='utf-8') as jf:
                            table_json = json.load(jf)
                        logger.debug(f"✅ Loaded schema for {table_name} from {os.path.basename(json_file)}")
                        break
                    except Exception as e:
                        logger.debug(f"Failed to load {json_file}: {e}")
                        continue
            
            if table_json:
                schema_text = self._build_schema_from_json(table_json, table_name)
                schemas[table_name] = schema_text
        
        return schemas
    
    def _load_schemas_from_sqlite(self, example_path: str, table_names: List[str]) -> Dict[str, str]:
        """
        从SQLite文件直接读取表结构（参考 schema_linking_graph_1120.py 的 get_table_info_from_sqlite）
        """
        import sqlite3
        import glob
        
        schemas = {}
        
        # 查找SQLite文件
        sqlite_files = glob.glob(os.path.join(example_path, '*.sqlite')) + \
                       glob.glob(os.path.join(example_path, '*.db'))
        
        if not sqlite_files:
            logger.warning(f"No SQLite files found in {example_path}")
            return schemas
        
        sqlite_file = sqlite_files[0]  # 使用第一个找到的SQLite文件
        logger.info(f"Reading table structures from {os.path.basename(sqlite_file)}")
        
        try:
            connection = sqlite3.connect(sqlite_file)
            cursor = connection.cursor()
            
            # 获取所有表
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
            all_tables = [row[0] for row in cursor.fetchall()]
            
            # 过滤出我们需要的表
            for table_name in table_names:
                short_name = table_name.split('.')[-1]
                
                # 查找匹配的表
                matched_table = None
                for db_table in all_tables:
                    if db_table == short_name or db_table == table_name:
                        matched_table = db_table
                        break
                
                if not matched_table or matched_table.startswith('sqlite_'):
                    continue
                
                try:
                    # 获取列信息
                    cursor.execute(f"PRAGMA table_info({matched_table})")
                    columns_info = cursor.fetchall()
                    
                    # 获取样本行
                    sample_rows = []
                    try:
                        cursor.execute(f"SELECT * FROM {matched_table} LIMIT 3")
                        sample_rows = cursor.fetchall()
                    except Exception as e:
                        logger.debug(f"Error fetching sample rows for {matched_table}: {e}")
                    
                    # 构建表信息字符串
                    table_info = f"Table full name: {matched_table}\n"
                    
                    for col in columns_info:
                        col_name = col[1]  # 列名
                        col_type = col[2]  # 列类型
                        table_info += f"Column name: {col_name} Type: {col_type}\n"
                    
                    # 添加样本行
                    if sample_rows:
                        table_info += f"Sample rows:\n{str(sample_rows[:3])}\n"
                    
                    schemas[table_name] = table_info.strip()
                    logger.debug(f"✅ Extracted schema for {table_name} from SQLite")
                    
                except Exception as e:
                    logger.error(f"Error processing table {matched_table}: {e}")
                    continue
            
            connection.close()
            
        except Exception as e:
            logger.error(f"Error reading SQLite file {sqlite_file}: {e}")
        
        return schemas

    def _select_tables_and_question_with_llm(
        self,
        table_descriptions: Dict[str, str],
        available_tables: List[str],
        k_tables: int,
        example_name: str,
    ) -> Tuple[Optional[List[str]], Optional[str]]:
        """
        使用 LLM 基于 table_descriptions.json 选出 K 个表，并设计一个可回答的问题。
        
        返回:
            (tables_selected, question_hint)
        """
        # 只在有足够表的情况下调用
        if not available_tables:
            return None, None

        # 构造表描述文本，只包含当前 example 下的表
        desc_blocks = []
        for tb_name, desc in table_descriptions.items():
            if tb_name in available_tables:
                desc_blocks.append(f"- {tb_name}: {desc}")
        if not desc_blocks:
            return None, None

        tables_desc_text = "\n".join(desc_blocks)

        prompt = f"""You are a database analyst. You must pick tables that can be joined WITHOUT guessing.

Below are ALL available tables in database "{example_name}" with short descriptions.

## Available Tables
{tables_desc_text}

## Task
Choose EXACTLY {k_tables} tables AND write ONE business/analytics question that is fully answerable using ONLY those tables.

## HARD JOIN RULES (STRICT)
- You may only assume joins when the relationship is obvious from table/field naming implied by the descriptions:
  - Same key name (e.g., order_id to order_id, seller_id to seller_id)
  - Foreign-key style naming (e.g., orders.customer_id to customers.customer_id)
- Do NOT invent bridge tables. If the join requires an intermediate table not selected, you must select it.
- Avoid disconnected domains (e.g., leads tables with ecommerce orders) unless there is a clear bridging key.
- If you cannot find a valid set of {k_tables} joinable tables, return tables_selected as [] and explain why in reason.

## METRIC SAFETY
- Prefer questions whose main metric can be computed at the grain of an obvious fact table (orders, order_items, payments, reviews, etc).
- Avoid questions that require summing pre-aggregated totals after 1-to-many joins.

## Output Format (STRICT JSON ONLY)
Return ONLY one JSON object:
{{
  "question": "one concrete question",
  "tables_selected": ["t1","t2", "... exactly {k_tables} ..."],
  "join_plan": "Describe how the tables join using key names (e.g., orders.order_id = order_items.order_id; order_items.seller_id = sellers.seller_id). If unknown, say UNKNOWN.",
  "reason": "If tables_selected is empty, explain missing join keys/bridge tables. Otherwise keep this short."
}}

Rules:
- tables_selected MUST contain ONLY table names from the list above.
- tables_selected MUST contain EXACTLY {k_tables} table names OR be [] (fail-safe).
- The question MUST be answerable using ONLY tables_selected.

Now output the JSON:
"""

        self.llm_client.clear_messages()
        response_text = self.llm_client.get_response(prompt)

        # 解析 JSON 响应
        try:
            # 尝试从响应中提取 JSON 对象
            json_match = re.search(r'\{.+\}', response_text, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
                data = json.loads(json_str)
            else:
                data = json.loads(response_text)
        except Exception as e:
            logger.warning(f"[{example_name}] Failed to parse table selection response: {e}")
            return None, None

        tables_selected = data.get("tables_selected") or data.get("tables") or []
        question_hint = data.get("question")

        if not isinstance(tables_selected, list) or not question_hint:
            return None, None

        # 过滤非法表名，只保留在 available_tables 中的
        tables_selected = [t for t in tables_selected if t in available_tables]
        # 去重并保留顺序
        seen = set()
        filtered = []
        for t in tables_selected:
            if t not in seen:
                seen.add(t)
                filtered.append(t)
        tables_selected = filtered[:k_tables]

        if len(tables_selected) == 0:
            return None, None

        logger.info(f"[{example_name}] LLM-selected tables: {tables_selected}")
        logger.info(f"[{example_name}] LLM-designed question: {question_hint[:120]}...")

        return tables_selected, question_hint
    
    def _parse_schemas_from_prompts(self, prompts_file: str, table_names: List[str]) -> Dict[str, str]:
        """
        从prompts.txt解析表schema信息
        prompts.txt格式：
        --------------------------------------------------
        Table full name: xxx
        Column name: xxx Type: xxx
        ...
        Sample rows: ...
        --------------------------------------------------
        """
        schemas = {}
        
        try:
            with open(prompts_file, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # 按分隔符分割成多个表
            table_blocks = content.split('--------------------------------------------------')
            
            for block in table_blocks:
                block = block.strip()
                if not block or 'Table full name:' not in block:
                    continue
                
                # 提取表名
                lines = block.split('\n')
                table_fullname = None
                for line in lines:
                    if line.startswith('Table full name:'):
                        table_fullname = line.replace('Table full name:', '').strip()
                        break
                
                if not table_fullname:
                    continue
                
                # 检查是否是我们需要的表
                matched_name = None
                for tn in table_names:
                    if tn == table_fullname or tn.split('.')[-1] == table_fullname:
                        matched_name = tn
                        break
                
                if matched_name:
                    # 保存整个block作为schema
                    schemas[matched_name] = block
                    logger.debug(f"✅ Parsed schema for {matched_name}")
            
            return schemas
            
        except Exception as e:
            logger.error(f"Error parsing prompts.txt: {e}")
            return {}
    
    def _get_fallback_schema(self, table_name: str, example_path: str) -> str:
        """Fallback: 从table_descriptions.json获取描述"""
        
        json_file = os.path.join(example_path, 'table_descriptions.json')
        if os.path.exists(json_file):
            with open(json_file, 'r', encoding='utf-8') as f:
                descriptions = json.load(f)
            desc = descriptions.get(table_name, "No description available")
            return f"Table full name: {table_name}\nDescription: {desc}"
        else:
            return f"Table full name: {table_name}\n(Schema not available)"
    
    def _load_schemas_from_descriptions(self, example_path: str, table_names: List[str]) -> Dict[str, str]:
        """从table_descriptions.json加载schema（作为fallback）"""
        
        json_file = os.path.join(example_path, 'table_descriptions.json')
        
        with open(json_file, 'r', encoding='utf-8') as f:
            descriptions = json.load(f)
        
        schemas = {}
        for table_name in table_names:
            desc = descriptions.get(table_name, "No description available")
            schemas[table_name] = f"Table full name: {table_name}\nDescription: {desc}"
        
        return schemas
    
    def _build_generation_prompt(self, 
                                 table_names: List[str], 
                                 table_schemas: Dict[str, str],
                                 example_name: str,
                                 n_questions: int = 1,
                                 question_hint: Optional[str] = None) -> str:
        """
        构建LLM生成prompt（每次生成1个问题）
        
        参考 run.py 和 prompt.py，根据数据库类型添加特定的提示
        """
        
        # 根据 example 名称推断数据库类型（参考 utils.py 的 get_api_name）
        try:
            api = get_api_name(example_name)
        except:
            # 如果无法推断，默认使用 bigquery
            api = "bigquery"
            logger.warning(f"Could not infer API type from {example_name}, defaulting to bigquery")
        
        logger.info(f"[Prompt] Detected API type: {api} for example: {example_name}")
        
        # 获取数据库特定的基本SQL格式示例
        dialect_basic = self.prompts.get_prompt_dialect_basic(api)
        
        # 获取数据库特定的嵌套字段处理提示
        dialect_nested = self.prompts.get_prompt_dialect_nested(api)
        
        # 获取数据库特定的字符串匹配提示
        dialect_string_matching = self.prompts.get_prompt_dialect_string_matching(api)
        
        # 获取数据库特定的UNION操作提示（如果有多个表）
        table_struct = ", ".join([t.split('.')[-1] for t in table_names])
        dialect_list_all_tables = self.prompts.get_prompt_dialect_list_all_tables(table_struct, api)
        
        # 其他通用提示
        prompt_decimal_places = self.prompts.get_prompt_decimal_places()
        prompt_convert_symbols = self.prompts.get_prompt_convert_symbols()
        prompt_knowledge = self.prompts.get_prompt_knowledge()
        
        # 构建基础 prompt
        prompt = f"""You are a SQL expert. Generate 1 realistic natural language question and corresponding SQL query based on the following database tables.

# Database Engine: {api.upper()}

# Database Tables and Schemas

"""
        
        # 添加每个表的详细schema
        for i, table_name in enumerate(table_names, 1):
            schema = table_schemas.get(table_name, "")
            
            # 限制schema长度（避免太长）
            if len(schema) > 5000:
                schema = schema[:5000] + "\n... (schema truncated for length)"
            
            prompt += f"\n## Table {i}: {table_name.split('.')[-1]}\n\n"
            prompt += schema + "\n"
        
        # 如果上游已经让 LLM 规划好了目标问题，这里显式告知模型“必须围绕该问题生成 SQL”
        if question_hint:
            prompt += f"""

# Target Question (MUST USE)

You must answer exactly the following business question (you may rephrase it slightly for naturalness, but do NOT change its meaning):

{question_hint}
"""
        
        # 根据提供的表数量调整要求
        if len(table_schemas) > 1:
            table_requirement = f"Use ALL {len(table_schemas)} tables provided above (you must use all of them in your SQL query)"
        else:
            table_requirement = "Use the table provided above"
        
        prompt += f"""

# Task

Generate exactly 1 diverse and realistic question-SQL pair that meets the following criteria:

1. Table Usage: {table_requirement}
2. Business Value: Represents a meaningful business or analytical insight (e.g., trend analysis, comparison, ranking, conversion)
3. SQL Dialect: Uses correct SQL syntax for the {api.upper()} dialect
4. Difficulty: Must be one of: easy, medium, hard
5. SQL Features: May include filtering, aggregation, sorting, JOINs, subqueries, or window functions
6. Result Quality: Returns actionable and interpretable results

# CRITICAL OUTPUT RULES (MUST FOLLOW)
- Output MUST be a valid JSON array with exactly one object.
- Output MUST contain NO extra text before or after the JSON (no explanations, no markdown).
- Do NOT wrap the JSON in code fences (no ```json).
- Do NOT include comments anywhere (no -- and no /* */) in the SQL or output.
- Ensure JSON strings are properly escaped (no unescaped newlines or quotes inside values).
- The "sql" field must be a single string.

# SQL Syntax Guidelines ({api.upper()} Dialect)

{dialect_basic}

# Critical Dialect-Specific Rules

{dialect_nested}

{dialect_string_matching}

{dialect_list_all_tables}

{prompt_convert_symbols}

{prompt_decimal_places}

{prompt_knowledge}

# DIALECT GUARDRAILS (IMPORTANT)
- NEVER use non-{api.upper()} operators/syntax such as: ->, ->>, @>, #>, ::, ILIKE, SIMILAR TO.
- Avoid angle-bracket type declarations in queries (e.g., ARRAY<...>, STRUCT<...>) unless the schema explicitly requires nested field access per the rules above.
- Do NOT use any placeholder identifiers like <TABLE> or <COLUMN>.
- Use only SQL that can run as-is in {api.upper()}.

# Key Requirements for Question & SQL

Question Quality:
- Must be natural, clear, and reflect real-world analytical needs
- Avoid vague terms like "data" or "information" — specify metrics (e.g., count, total revenue, average duration)
- When entity names are not specified, return both name and ID (if both exist in schema)

SQL Correctness & Executability:
- Must be syntactically correct in {api.upper()}
- Use fully qualified table names with backticks if required (e.g., `project.dataset.table`)
- Only use columns that exist in the provided schema (do not guess)
- Ensure all parentheses are properly closed
- GROUP BY must include all non-aggregated selected fields

Relevance & Simplicity:
- Focus on delivering meaningful insights, not unnecessary complexity
- Avoid deeply nested subqueries (max 2-3 levels) unless essential
- Prefer readability and correctness over cleverness
- Keep CTEs to 2-4 maximum

Aggregation & Metrics:
- Use appropriate aggregations (SUM, COUNT, AVG, etc.) with proper GROUP BY
- When calculating percentage decrease: return a positive number using ABS() if needed
- Always alias derived fields with clear names (e.g., AS total_revenue, AS session_count)
- IMPORTANT - Cumulative Fields:
  - Fields named with "total", "cumulative", "accumulated" are often pre-aggregated values
  - Do NOT SUM() cumulative/total fields that are already aggregated per row
  - When in doubt, use the field directly or use MAX() instead of SUM()

Common Pitfalls to Avoid:
- For BigQuery TIMESTAMP fields, use TIMESTAMP_TRUNC() or DATE() instead of TIMESTAMP_SUB() with YEAR/MONTH intervals
- Verify all columns in SELECT exist in the FROM/JOIN tables
- Do NOT double-aggregate pre-aggregated fields

# FAIL-SAFE (DO NOT HALLUCINATE)
- If the schema does not contain enough columns to answer a business question without guessing,
  output a JSON object with "sql" as an empty string ("") and put a short reason in "question".

# GROUP BY RULE (STRICT)
- If the query contains a GROUP BY clause:
  - Every SELECT expression that is NOT wrapped in an aggregate function (e.g., COUNT, SUM, AVG, MIN, MAX, APPROX_COUNT_DISTINCT)
    MUST appear in the GROUP BY clause (including expressions like DATE(ts), TIMESTAMP_TRUNC(ts, DAY), COALESCE(a,b)).
  - If you need to include a field in SELECT but do NOT want to group by it, wrap it with ANY_VALUE(field) and give it a clear alias.
- Never select non-aggregated fields alongside aggregated metrics without adding them to GROUP BY or using ANY_VALUE().

# SINGLE-SELECT ONLY (STRICT)
- The SQL MUST contain exactly ONE top-level SELECT statement.
- Do NOT use WITH/CTEs.
- Do NOT use UNION / UNION ALL.
- Do NOT use multiple SQL statements separated by semicolons.

# BIGQUERY SHARDED TABLE RULE (STRICT)
- If multiple ga_sessions_YYYYMMDD tables are provided, you MUST query them using a wildcard table:
  FROM `bigquery-public-data.google_analytics_sample.ga_sessions_*`
  and filter to the requested dates using:
  WHERE _TABLE_SUFFIX IN ('YYYYMMDD', 'YYYYMMDD', ...)
- Perform aggregation in this single SELECT (use GROUP BY as needed).

# Output Format

You must NOT invent joins. Only join on keys that clearly match (same name or FK naming like x_id=id).
Before writing SQL, verify all requested fields are reachable from one chosen fact table using valid joins.
If you cannot reach a field or apply a filter without guessing, output sql="" and question="Schema insufficient: <missing table/key>".
Never join unrelated identifiers (order_id=customer_id, product_id=category_name, city=category_name).
Avoid summing pre-aggregated totals after 1-to-many joins to prevent double counting.

Return ONLY a JSON array with exactly one object using this structure (do not copy the example values; generate new ones that match the schema):

[
  {{
    "question": "...",
    "sql": "...",
    "difficulty": "easy|medium|hard",
    "tables_used": ["..."],
    "relevant_columns": ["..."]
  }}
]
"""
        
        return prompt
    
    def _call_llm_generate(self, prompt: str) -> List[Dict]:
        """调用LLM生成问题和SQL（使用GPTChat）"""
        
        try:
            # 使用GPTChat的get_response方法
            # 注意：GPTChat会维护消息历史，每次调用前清空
            self.llm_client.clear_messages()
            
            logger.debug(f"[LLM Input] Sending prompt to LLM...")
            logger.debug(f"[LLM Input] Prompt length: {len(prompt)} chars")
            
            response_text = self.llm_client.get_response(prompt)
            
            logger.info(f"[LLM Output] Response received: {len(response_text)} characters")
            logger.info(f"[LLM Output] Response preview (first 500 chars):")
            logger.info(f"{response_text[:500]}...")
            logger.debug(f"[LLM Output] Full response: {response_text}")
            
            # 解析JSON
            logger.debug(f"[Processing] Parsing JSON from response...")
            qa_pairs = self._parse_json_response(response_text)
            
            logger.info(f"[Processing] Parsed {len(qa_pairs)} QA pair(s) from response")
            
            return qa_pairs
        
        except Exception as e:
            logger.error(f"[Error] LLM generation failed: {e}")
            import traceback
            logger.debug(traceback.format_exc())
            return []
    
    def _parse_json_response(self, response: str) -> List[Dict]:
        """从LLM响应中解析JSON"""
        
        logger.debug(f"[Parsing] Attempting to extract JSON from response...")
        
        # 尝试提取JSON block
        json_match = re.search(r'```json\s*\n(.+?)\n```', response, re.DOTALL)
        if json_match:
            json_str = json_match.group(1)
            logger.debug(f"[Parsing] Found JSON in code block")
        else:
            # 尝试直接提取数组
            json_match = re.search(r'\[.+\]', response, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
                logger.debug(f"[Parsing] Found JSON array in response")
            else:
                logger.warning(f"[Parsing] No JSON found in response")
                logger.debug(f"[Parsing] Response preview: {response[:500]}")
                return []
        
        logger.debug(f"[Parsing] Extracted JSON string length: {len(json_str)} chars")
        logger.debug(f"[Parsing] JSON preview: {json_str[:300]}...")
        
        try:
            qa_list = json.loads(json_str)
            if not isinstance(qa_list, list):
                qa_list = [qa_list]
                logger.debug(f"[Parsing] Converted single object to list")
            
            logger.info(f"[Parsing] Successfully parsed {len(qa_list)} QA pair(s)")
            return qa_list
        except json.JSONDecodeError as e:
            logger.error(f"[Parsing] JSON parse error: {e}")
            logger.debug(f"[Parsing] JSON string preview: {json_str[:500]}")
            logger.debug(f"[Parsing] Full JSON string: {json_str}")
            return []
    
    def _validate_and_enrich(self, qa_raw: Dict, table_schemas: Dict[str, str], example_name: str) -> QAPair:
        """验证SQL并补充信息"""
        
        question = qa_raw.get('question', '')
        sql = qa_raw.get('sql', '')
        tables_used = qa_raw.get('tables_used', [])
        relevant_columns = qa_raw.get('relevant_columns', [])
        
        logger.debug(f"[Validation] Question: {question}")
        logger.debug(f"[Validation] SQL: {sql}")
        logger.debug(f"[Validation] Tables used: {tables_used}")
        logger.debug(f"[Validation] Relevant columns: {relevant_columns}")
        
        # 计算size（所有使用表的schema总长度）
        size = sum(
            len(table_schemas.get(table, '')) 
            for table in tables_used 
            if table in table_schemas
        )
        logger.debug(f"[Validation] Tables size: {size} chars")

        if not sql or not sql.strip():
            logger.warning(f"[Validation] Empty SQL received; skipping execution")
            return QAPair(
                query=question,
                tables=tables_used,
                relevant_columns=relevant_columns,
                sql=sql,
                data=None,
                size=size,
                success=False,
                error_message="Empty SQL generated by fail-safe; execution skipped.",
                execution_time=0.0,
                source_example=example_name
            )
        
        # 验证SQL
        logger.debug(f"[Validation] Executing SQL with LIMIT 1 for validation...")
        success, data, error_msg, exec_time = self._execute_sql_with_limit(sql, example_name)
        
        logger.info(f"[Validation] SQL execution completed:")
        logger.info(f"  Success: {success}")
        logger.info(f"  Execution time: {exec_time:.3f}s")
        if success:
            logger.info(f"  Data type: {type(data)}")
            logger.info(f"  Data preview: {str(data)[:200]}...")
        else:
            logger.warning(f"  Error message: {error_msg}")
        
        return QAPair(
            query=question,
            tables=tables_used,
            relevant_columns=relevant_columns,
            sql=sql,
            data=data,
            size=size,
            success=success,
            error_message=error_msg,
            execution_time=exec_time,
            source_example=example_name
        )
    
    def _retry_fix_sql(self, qa_pair: QAPair, qa_raw: Dict, table_schemas: Dict[str, str], 
                       example_name: str, max_retries: int = 3) -> QAPair:
        """
        当SQL执行失败时，尝试根据错误信息修复SQL
        
        Args:
            qa_pair: 失败的QAPair
            qa_raw: 原始LLM输出
            table_schemas: 表schema信息
            example_name: example名称
            max_retries: 最大重试次数
        
        Returns:
            修复后的QAPair（如果成功）或原始QAPair（如果所有重试都失败）
        """
        
        for retry_idx in range(max_retries):
            logger.info(f"[{example_name}] Retry {retry_idx+1}/{max_retries}: Attempting to fix SQL...")
            
            try:
                # 构建修复提示
                fix_prompt = self._build_fix_prompt(
                    question=qa_pair.query,
                    failed_sql=qa_pair.sql,
                    error_message=qa_pair.error_message,
                    table_schemas=table_schemas,
                    example_name=example_name
                )
                
                # 调用LLM修复
                logger.info(f"[{example_name}] Calling LLM to fix SQL...")
                self.llm_client.clear_messages()
                response_text = self.llm_client.get_response(fix_prompt)
                logger.info(f"[{example_name}] LLM response received, length: {len(response_text)} chars")
                logger.debug(f"[{example_name}] Response preview: {response_text[:500]}...")
                
                # 解析修复后的SQL
                logger.info(f"[{example_name}] Parsing LLM response...")
                fixed_qa_list = self._parse_json_response(response_text)
                logger.info(f"[{example_name}] Parsing completed, found {len(fixed_qa_list) if fixed_qa_list else 0} QA pairs")
                if not fixed_qa_list:
                    logger.warning(f"[{example_name}] Retry {retry_idx+1}: Failed to parse LLM response")
                    continue
                
                fixed_qa_raw = fixed_qa_list[0]
                
                # 验证修复后的SQL
                fixed_qa_pair = self._validate_and_enrich(fixed_qa_raw, table_schemas, example_name)
                
                if fixed_qa_pair.success:
                    logger.info(f"[{example_name}] ✅ SQL fixed successfully on retry {retry_idx+1}")
                    return fixed_qa_pair
                else:
                    logger.warning(f"[{example_name}] Retry {retry_idx+1}: SQL still fails: {fixed_qa_pair.error_message}")
                    qa_pair = fixed_qa_pair  # 更新错误信息，继续重试
            
            except Exception as e:
                logger.error(f"[{example_name}] Retry {retry_idx+1}: Error during fix attempt: {e}")
                import traceback
                logger.debug(traceback.format_exc())
                continue
        
        logger.error(f"[{example_name}] ❌ Failed to fix SQL after {max_retries} retries")
        return qa_pair
    
    def _validate_result_with_llm(self, qa_pair: QAPair, qa_raw: Dict, table_schemas: Dict[str, str],
                                   example_name: str, validation_depth: int = 0) -> QAPair:
        """
        让LLM验证SQL执行结果是否正确
        
        Args:
            qa_pair: 成功执行的QAPair
            qa_raw: 原始LLM输出
            table_schemas: 表schema信息
            example_name: example名称
            validation_depth: 当前验证深度（防止无限递归）
        
        Returns:
            验证后的QAPair（可能包含修正的SQL）
        """
        
        # 限制验证深度，避免无限递归
        MAX_VALIDATION_DEPTH = 2
        if validation_depth >= MAX_VALIDATION_DEPTH:
            logger.warning(f"[{example_name}] Reached max validation depth ({MAX_VALIDATION_DEPTH}), stopping validation")
            return qa_pair
        
        try:
            logger.info(f"[{example_name}] Asking LLM to validate SQL result (depth: {validation_depth})...")
            
            # 构建验证提示
            validation_prompt = self._build_validation_prompt(
                question=qa_pair.query,
                sql=qa_pair.sql,
                result_data=qa_pair.data,
                table_schemas=table_schemas,
                example_name=example_name
            )
            
            # 调用LLM验证
            self.llm_client.clear_messages()
            response_text = self.llm_client.get_response(validation_prompt)
            
            # 解析验证结果
            logger.debug(f"[{example_name}] Validation response: {response_text[:500]}...")
            
            # 检查是否需要修正
            if "CORRECT" in response_text.upper() or "NO ISSUES" in response_text.upper():
                logger.info(f"[{example_name}] ✅ LLM validated: SQL result is correct")
                return qa_pair
            elif "INCORRECT" in response_text.upper() or "ISSUE" in response_text.upper():
                logger.warning(f"[{example_name}] ⚠️  LLM found issues, attempting to fix...")
                
                # 尝试从响应中提取修正的SQL
                fixed_qa_list = self._parse_json_response(response_text)
                if fixed_qa_list:
                    fixed_qa_raw = fixed_qa_list[0]
                    fixed_qa_pair = self._validate_and_enrich(fixed_qa_raw, table_schemas, example_name)
                    
                    if fixed_qa_pair.success and fixed_qa_pair.data:
                        logger.info(f"[{example_name}] ✅ SQL corrected based on LLM feedback, re-validating...")
                        # 再次验证修复后的结果，确保真的正确了（递归调用，深度+1）
                        final_qa_pair = self._validate_result_with_llm(
                            fixed_qa_pair, fixed_qa_raw, table_schemas, example_name, 
                            validation_depth=validation_depth + 1
                        )
                        return final_qa_pair
                    else:
                        logger.warning(f"[{example_name}] ❌ Corrected SQL still fails, marking as unsuccessful")
                        # 修复后的SQL仍然失败，标记为失败（使用修复后的信息）
                        return replace(
                            fixed_qa_pair,
                            success=False, 
                            error_message=f"LLM validation failed: Corrected SQL still fails. {fixed_qa_pair.error_message or ''}"
                        )
                else:
                    logger.warning(f"[{example_name}] ❌ Could not parse corrected SQL, marking as unsuccessful")
                    # 无法修复，标记为失败
                    return replace(
                        qa_pair,
                        success=False,
                        error_message="LLM validation failed: Result does not correctly answer the question and could not be fixed."
                    )
            else:
                logger.warning(f"[{example_name}] ⚠️  LLM validation response unclear, marking as unsuccessful")
                # LLM响应不明确，保守起见标记为失败
                return replace(
                    qa_pair,
                    success=False,
                    error_message="LLM validation unclear: Could not determine if result is correct."
                )
        
        except Exception as e:
            logger.error(f"[{example_name}] Error during LLM validation: {e}")
            import traceback
            logger.debug(traceback.format_exc())
            return qa_pair
    
    def _build_fix_prompt(self, question: str, failed_sql: str, error_message: str,
                          table_schemas: Dict[str, str], example_name: str) -> str:
        """构建SQL修复提示"""
        
        # 确定数据库类型
        api = self._get_api_from_example_name(example_name)
        
        # 获取数据库特定指令
        dialect_basic = self.prompts.get_prompt_dialect_basic(api)
        
        # 检测是否是超时错误
        is_timeout = "timed out" in str(error_message).lower() or "timeout" in str(error_message).lower()
        
        timeout_warning = ""
        if is_timeout:
            timeout_warning = """
⚠️  **CRITICAL: QUERY TIMEOUT DETECTED**

Your SQL query took too long to execute (>60 seconds). You MUST simplify the query:

**Required Simplifications:**
1. Remove CTEs (WITH clauses) - use simple SELECT statements instead
2. Reduce the number of JOINs - use only 2-3 tables maximum
3. Avoid complex subqueries - flatten the query structure
4. Remove DISTINCT if possible - it's computationally expensive
5. Add WHERE clauses to filter data early
6. Remove unnecessary calculations or aggregations

**Example of simplification:**
- BAD: `WITH ... JOIN ... JOIN ... GROUP BY ... HAVING ...`
- GOOD: `SELECT simple_columns FROM table1 JOIN table2 ON ... WHERE ... LIMIT 100`

"""
        
        prompt = f"""# SQL Error Fixing Task

You generated a SQL query that failed to execute. Please fix the SQL based on the error message.

{timeout_warning}

## Original Question
{question}

## Your Previous SQL (FAILED)
```sql
{failed_sql}
```

## Error Message
```
{error_message}
```

## Available Tables and Schema

"""
        
        for i, (table_name, schema) in enumerate(table_schemas.items(), 1):
            prompt += f"### Table {i}: {table_name.split('.')[-1]}\n\n"
            prompt += schema + "\n\n"
        
        prompt += f"""
## SQL Dialect: {api.upper()}

{dialect_basic}

## Common SQL Errors and Fixes

1. **UNION ALL with mismatched columns**: Ensure all SELECT statements have the same number and types of columns
2. **GROUP BY missing**: All non-aggregated columns in SELECT must be in GROUP BY
3. **Column not found**: Verify column names match the schema exactly
4. **TIMESTAMP functions**: Use TIMESTAMP_TRUNC() or DATE() for BigQuery timestamps
5. **Parentheses mismatch**: Check all opening parentheses have matching closing ones
6. **Double aggregation**: Do NOT use SUM() on fields like totals.X that are already aggregated per row
7. **Query Timeout**: Simplify complex queries - avoid CTEs, reduce JOINs, add WHERE filters

## Task

Fix the SQL query to resolve the error. {"**IMPORTANT: Simplify the query significantly to avoid timeout!**" if is_timeout else ""} Return ONLY a JSON array with one object:

```json
[
  {{
    "question": "{question}",
    "sql": "CORRECTED SQL HERE",
    "difficulty": "medium",
    "tables_used": ["table_name"],
    "relevant_columns": ["column_name"]
  }}
]
```

Generate the corrected SQL now:
"""
        
        return prompt
    
    def _build_validation_prompt(self, question: str, sql: str, result_data: str,
                                 table_schemas: Dict[str, str], example_name: str) -> str:
        """构建SQL结果验证提示"""
        
        # 限制result_data长度
        max_result_len = 2000
        if len(result_data) > max_result_len:
            result_data = result_data[:max_result_len] + "\n... (truncated)"
        
        prompt = f"""# SQL Result Validation Task

Please validate if the SQL query correctly answers the question based on the execution result.

## Question
{question}

## SQL Query
```sql
{sql}
```

## Execution Result
```
{result_data}
```

## Available Tables and Schema

"""
        
        for i, (table_name, schema) in enumerate(table_schemas.items(), 1):
            prompt += f"### Table {i}: {table_name.split('.')[-1]}\n\n"
            prompt += schema + "\n\n"
        
        prompt += """
## Validation Criteria

1. **Correctness**: Does the SQL correctly implement the question's logic?
2. **Completeness**: Does the result answer all parts of the question?
3. **Data Quality**: Does the result make sense given the question?
4. **Common Issues to Check**:
   - Wrong aggregation (e.g., SUM on already-aggregated totals.X fields)
   - Missing filters or conditions
   - Incorrect JOIN conditions
   - Wrong GROUP BY columns
   - Double counting or incorrect calculations

## Task

Analyze the SQL and result, then respond:

1. If the SQL is CORRECT and the result makes sense:
   - Respond with: "CORRECT - The SQL correctly answers the question."

2. If you find issues:
   - Respond with: "INCORRECT - [explain the issue]"
   - Then provide a corrected SQL in JSON format:

```json
[
  {
    "question": "same question",
    "sql": "CORRECTED SQL HERE",
    "difficulty": "medium",
    "tables_used": ["table_name"],
    "relevant_columns": ["column_name"]
  }
]
```

Your validation:
"""
        
        return prompt
    
    def _get_api_from_example_name(self, example_name: str) -> str:
        """从example名称推断数据库API类型"""
        
        if example_name.startswith('bq'):
            return 'bigquery'
        elif example_name.startswith('sf') or example_name.startswith('snowflake'):
            return 'snowflake'
        elif example_name.startswith('local'):
            return 'sqlite'
        else:
            return 'sqlite'  # 默认
    
    def _execute_sql_with_limit(self, sql: str, example_name: str) -> tuple:
        """
        执行SQL验证（加LIMIT测试）
        
        参考 eval.py 和 agent.py 的验证方式
        
        Args:
            sql: SQL查询语句
            example_name: example名称（用于推断数据库类型）
        
        Returns:
            (success, data, error_message, execution_time)
        """
        
        logger.debug(f"[SQL Execution] Original SQL: {sql}")
        
        # 确定数据库类型
        api = self._get_api_from_example_name(example_name)
        logger.info(f"[SQL Execution] Detected API type: {api} for example: {example_name}")
        
        # 获取sqlite路径（如果是sqlite）
        sqlite_path = None
        if api == 'sqlite':
            example_path = os.path.join(self.input_path, example_name)
            # 查找.sqlite或.db文件
            if os.path.exists(example_path):
                for file in sorted(os.listdir(example_path)):
                    if file.endswith(('.sqlite', '.db')):
                        sqlite_path = os.path.join(example_path, file)
                        logger.info(f"[SQL Execution] Found SQLite file: {sqlite_path}")
                        break
                if not sqlite_path:
                    logger.warning(f"[SQL Execution] No .sqlite/.db file found in {example_path}")
        
        # 添加LIMIT
        test_sql = self._add_limit_to_sql(sql, limit=10)
        logger.debug(f"[SQL Execution] SQL with LIMIT: {test_sql}")
        
        start_time = time.time()
        
        try:
            logger.debug(f"[SQL Execution] Calling execute_sql_api with api={api}, sqlite_path={sqlite_path}...")
            # execute_sql_api 签名: execute_sql_api(sql_query, ex_id, save_path=None, api="sqlite", max_len=30000, sqlite_path=None, timeout=300)
            # 参考 agent.py 第51行和 eval.py 第142行的调用方式
            result = self.sql_env.execute_sql_api(
                test_sql,     # sql_query (位置参数)
                "test",       # ex_id (位置参数)
                None,         # save_path (None表示不保存，返回CSV字符串)
                api,          # api (动态确定：bigquery/snowflake/sqlite)
                30000,        # max_len
                sqlite_path,  # sqlite_path (sqlite数据库文件路径)
                30           # timeout (180秒=3分钟，给复杂查询足够时间)
            )
            
            execution_time = time.time() - start_time
            logger.debug(f"[SQL Execution] API returned type: {type(result)}")
            logger.debug(f"[SQL Execution] API returned value: {str(result)[:500]}...")
            logger.debug(f"[SQL Execution] Execution time: {execution_time:.3f}s")
            
            # 参考 agent.py 第53行的判断逻辑：成功时返回字符串且不是空结果
            # 参考 sql.py 第212-215行：失败时返回包含 "##ERROR##" 的字符串或字典
            if isinstance(result, dict):
                # 错误情况：返回了错误字典
                error_msg = result.get('error_msg', str(result))[:500]
                logger.warning(f"[SQL Execution] ❌ SQL execution failed (dict): {error_msg}")
                return (False, None, error_msg, execution_time)
            elif isinstance(result, str):
                # 检查是否包含错误标记
                if "##ERROR##" in result:
                    error_msg = result[:500]
                    logger.warning(f"[SQL Execution] ❌ SQL execution failed (error string): {error_msg}")
                    return (False, None, error_msg, execution_time)
                elif result == "" or result == "0":
                    # 空结果或保存成功（save_path不为None时返回"0"）
                    data = "Execution successful (empty result or saved to file)"
                    logger.info(f"[SQL Execution] ✅ SQL executed successfully (empty/saved)")
                    return (True, data, None, execution_time)
                else:
                    # 成功：返回了CSV字符串结果
                    # 限制数据长度，避免太大
                    data = result[:10000] if len(result) > 10000 else result
                    logger.info(f"[SQL Execution] ✅ SQL executed successfully (returned {len(result)} chars)")
                    logger.debug(f"[SQL Execution] Data preview: {data[:500]}...")
                    return (True, data, None, execution_time)
            else:
                # 未知返回类型
                error_msg = f"Unexpected return type: {type(result)}"
                logger.warning(f"[SQL Execution] ❌ {error_msg}")
                return (False, None, error_msg, execution_time)
        
        except Exception as e:
            execution_time = time.time() - start_time
            error_msg = str(e)[:500]
            logger.error(f"[SQL Execution] ❌ Exception occurred: {error_msg}")
            logger.debug(f"[SQL Execution] Exception details: {e}")
            import traceback
            logger.debug(f"[SQL Execution] Traceback: {traceback.format_exc()}")
            return (False, None, error_msg, execution_time)
    
    def _add_limit_to_sql(self, sql: str, limit: int = 10) -> str:
        """智能添加LIMIT"""
        
        if not sql or not sql.strip():
            return sql

        sql = sql.strip().rstrip(';')
        sql_upper = sql.upper()
        
        # 移除已有的LIMIT
        if 'LIMIT' in sql_upper:
            limit_pos = sql_upper.rfind('LIMIT')
            limit_match = re.search(r'LIMIT\s+\d+', sql[limit_pos:], re.IGNORECASE)
            if limit_match:
                sql = sql[:limit_pos + limit_match.start()].strip()
        
        # 添加新LIMIT
        return f"{sql} LIMIT {limit}"
    
    def _init_json_file(self, output_file: str):
        """初始化JSONL文件（JSON Lines格式，每行一个JSON对象）"""
        
        # 如果文件已存在，先备份
        if os.path.exists(output_file):
            backup_file = output_file + ".backup"
            import shutil
            shutil.copy2(output_file, backup_file)
            logger.info(f"[Save] Backed up existing file to: {backup_file}")
        
        # JSONL格式：文件可以为空，或者每行一个JSON对象
        # 不需要初始化，直接追加即可
        logger.debug(f"[Save] Initialized JSONL file: {output_file}")
    
    def _append_to_json_file(self, output_file: str, qa_pair: QAPair):
        """实时追加单个QA对到JSONL文件（JSON Lines格式）"""
        
        try:
            import threading
            # 使用文件锁确保线程安全
            if not hasattr(self, '_file_lock'):
                self._file_lock = threading.Lock()
            
            with self._file_lock:
                # JSONL格式：每行一个JSON对象，直接追加即可
                qa_dict = asdict(qa_pair)
                with open(output_file, 'a', encoding='utf-8') as f:
                    json.dump(qa_dict, f, ensure_ascii=False)
                    f.write('\n')  # 每行一个JSON对象
            
            logger.debug(f"[Save] Appended QA pair to {output_file}")
        
        except Exception as e:
            logger.error(f"[Save] Failed to append to JSONL file: {e}")
            import traceback
            logger.debug(traceback.format_exc())
    
    def _finalize_json_file(self, output_file: str):
        """确保JSONL文件格式正确（验证每行都是有效的JSON）"""
        
        try:
            if os.path.exists(output_file):
                # 验证JSONL格式：检查每行是否是有效的JSON
                line_count = 0
                with open(output_file, 'r', encoding='utf-8') as f:
                    for line_num, line in enumerate(f, 1):
                        line = line.strip()
                        if line:  # 跳过空行
                            try:
                                json.loads(line)
                                line_count += 1
                            except json.JSONDecodeError as e:
                                logger.warning(f"[Save] Invalid JSON at line {line_num}: {e}")
                
                logger.debug(f"[Save] Finalized JSONL file: {output_file} ({line_count} valid lines)")
        
        except Exception as e:
            logger.warning(f"[Save] Failed to finalize JSONL file: {e}")
    
    def save_results(self, qa_pairs: List[QAPair], output_file: str):
        """保存结果到JSONL文件（如果已经实时写入，则只打印统计）"""
        
        # 检查文件是否已存在（可能已经实时写入）
        if os.path.exists(output_file):
            file_size = os.path.getsize(output_file)
            # 统计行数
            line_count = sum(1 for _ in open(output_file, 'r', encoding='utf-8') if _.strip())
            logger.info(f"[Save] Results already saved to {output_file}")
            logger.info(f"[Save] File size: {file_size / 1024:.2f} KB")
            logger.info(f"[Save] Total lines: {line_count}")
            # 确保格式正确
            self._finalize_json_file(output_file)
        else:
            # 如果没有实时写入，现在保存为JSONL格式
            logger.info(f"[Save] Saving {len(qa_pairs)} QA pairs to {output_file} (JSONL format)")
            
            logger.debug(f"[Save] Converting to JSONL format...")
            with open(output_file, 'w', encoding='utf-8') as f:
                for qa in qa_pairs:
                    qa_dict = asdict(qa)
                    json.dump(qa_dict, f, ensure_ascii=False)
                    f.write('\n')
            
            file_size = os.path.getsize(output_file)
            logger.info(f"[Save] ✅ Saved {len(qa_pairs)} QA pairs to {output_file}")
            logger.info(f"[Save] File size: {file_size / 1024:.2f} KB")
        
        # 打印统计
        self._print_statistics(qa_pairs)
    
    def _print_statistics(self, qa_pairs: List[QAPair]):
        """打印统计信息"""
        
        total = len(qa_pairs)
        if total == 0:
            print("\n⚠️  No QA pairs generated")
            return
        
        successful = sum(1 for qa in qa_pairs if qa.success)
        
        print("\n" + "="*80)
        print("Generation Statistics")
        print("="*80)
        print(f"Total QA pairs: {total}")
        print(f"Successful SQL: {successful} ({successful/total*100:.1f}%)")
        print(f"Failed SQL: {total - successful} ({(total-successful)/total*100:.1f}%)")
        
        if successful > 0:
            avg_exec_time = sum(qa.execution_time for qa in qa_pairs if qa.execution_time and qa.success) / successful
            print(f"Avg execution time: {avg_exec_time:.2f}s")
        
        avg_size = sum(qa.size for qa in qa_pairs) / total
        print(f"Avg size: {avg_size:.0f} chars")
        print("="*80 + "\n")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate QA pairs from examples_lite directory')
    parser.add_argument('--input_path', type=str, 
                       default='/path/to/examples_lite',
                       help='Path to examples_lite directory')
    parser.add_argument('--k_tables', type=int, default=3,
                       help='Number of random tables to select')
    parser.add_argument('--n_questions', type=int, default=5,
                       help='Number of questions to generate')
    parser.add_argument('--output', type=str, default='generated_qa.jsonl',
                       help='Output JSONL file (JSON Lines format, one JSON object per line)')
    parser.add_argument('--dir_prefix', type=str, default=None,
                       help='Filter subdirectories by prefix (e.g., "local" to only process local001, local002, etc.)')
    parser.add_argument('--llm_model', type=str, default='Qwen3-235B-A22B-Instruct-2507-FP8',
                       help='LLM model name (for GPTChat)')
    parser.add_argument('--azure', action='store_true',
                       help='Use Azure OpenAI')
    parser.add_argument('--temperature', type=float, default=0.7,
                       help='Temperature for LLM')
    
    args = parser.parse_args()
    
    # 初始化LLM客户端（使用GPTChat）
    try:
        llm_client = GPTChat(
            azure=args.azure,
            model=args.llm_model,
            temperature=args.temperature
        )
        logger.info(f"LLM client initialized: model={args.llm_model}, azure={args.azure}, temperature={args.temperature}")
    except Exception as e:
        logger.error(f"Failed to initialize LLM client: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # 初始化SQL环境
    try:
        sql_env = SqlEnv()
        logger.info("SQL environment initialized")
    except Exception as e:
        logger.error(f"Failed to initialize SQL environment: {e}")
        return
    
    # 创建生成器
    try:
        generator = TableBasedQuestionGenerator(
            input_path=args.input_path,
            llm_client=llm_client,
            sql_env=sql_env,
            dir_prefix=args.dir_prefix
        )
    except Exception as e:
        logger.error(f"Failed to create generator: {e}")
        return
    
    # 生成QA对（带全局错误处理，实时写入）
    qa_pairs = []
    try:
        qa_pairs = generator.generate_qa_pairs(
            k_tables=args.k_tables,
            n_questions=args.n_questions,
            output_file=args.output  # 传入输出文件路径，实现实时写入
        )
    except Exception as e:
        logger.error(f"[Fatal Error] Failed to generate QA pairs: {e}")
        import traceback
        logger.error(traceback.format_exc())
        # 即使出错，也尝试保存已生成的部分
        if qa_pairs:
            logger.info(f"Saving {len(qa_pairs)} QA pairs generated before error...")
        else:
            return
    
    if not qa_pairs:
        logger.warning("No QA pairs generated")
        return
    
    # 保存结果
    try:
        generator.save_results(qa_pairs, args.output)
    except Exception as e:
        logger.error(f"Failed to save results: {e}")
        return
    
    # 打印示例
    successful_pairs = [qa for qa in qa_pairs if qa.success]
    if successful_pairs:
        print("\n📋 Sample Successful Question:")
        print("="*80)
        sample = successful_pairs[0]
        print(f"Q: {sample.query}")
        print(f"\nSource: {sample.source_example}")
        print(f"Tables: {', '.join([t.split('.')[-1] for t in sample.tables])}")
        print(f"Columns: {', '.join(sample.relevant_columns[:5])}")
        print(f"\nSQL:\n{sample.sql}")
        print(f"\nSize: {sample.size} chars")
        print(f"Execution: {'✅ Success' if sample.success else '❌ Failed'} ({sample.execution_time:.2f}s)")
        print("="*80)
    
    # 如果有失败的，也打印一个示例
    failed_pairs = [qa for qa in qa_pairs if not qa.success]
    if failed_pairs:
        print("\n⚠️  Sample Failed Question:")
        print("="*80)
        sample = failed_pairs[0]
        print(f"Q: {sample.query}")
        print(f"\nSQL:\n{sample.sql}")
        print(f"\nError: {sample.error_message}")
        print("="*80)


if __name__ == "__main__":
    main()

