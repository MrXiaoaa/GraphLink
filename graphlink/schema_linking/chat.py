from openai import OpenAI, AzureOpenAI
import httpx
from utils import extract_all_blocks
import os
import time
import json
from datetime import datetime

class GPTChat:
    def __init__(self, azure=False, model="gpt-4o", temperature=1) -> None:
        request_timeout = float(os.environ.get("GRAPHLINK_API_TIMEOUT", "180"))
        max_retries = int(os.environ.get("GRAPHLINK_API_MAX_RETRIES", "0"))
        http_timeout = httpx.Timeout(request_timeout, connect=min(30.0, request_timeout))
        if not azure:
            if model in ["o1-preview", "o1-mini"]:
                self.client = OpenAI(
                    api_key=os.environ.get("OPENAI_API_KEY"),
                    api_version="2024-12-01-preview",
                    timeout=request_timeout,
                    max_retries=max_retries,
                    http_client=httpx.Client(trust_env=False, timeout=http_timeout),
                )
            elif model in ["deepseek-reasoner"]:
                self.client = OpenAI(
                    base_url="https://api.deepseek.com",
                    api_key=os.environ.get("DS_API_KEY"),
                    timeout=request_timeout,
                    max_retries=max_retries,
                    http_client=httpx.Client(trust_env=False, timeout=http_timeout),
                )         
            elif model in ["Qwen3-Coder-480B-A35B-Instruct-FP8"]:
                self.client = OpenAI(
                    base_url=os.environ.get("OPENAI_BASE_URL"),
                    api_key=os.environ.get("OPENAI_API_KEY"),
                    timeout=request_timeout,
                    max_retries=max_retries,
                    http_client=httpx.Client(trust_env=False, timeout=http_timeout),
                )             
            else: 
                self.client = OpenAI(
                    base_url=os.environ.get("OPENAI_BASE_URL"),
                    api_key=os.environ.get("OPENAI_API_KEY"),
                    timeout=request_timeout,
                    max_retries=max_retries,
                    http_client=httpx.Client(trust_env=False, timeout=http_timeout),
                )
            # else:
            #     raise NotImplementedError("Unsupported API Key")
        else:
            if model in ["o1-preview", "o1-mini", "o3", "o4-mini"]:
                self.client = AzureOpenAI(
                    azure_endpoint = os.environ.get("AZURE_ENDPOINT"),
                    api_key=os.environ.get("AZURE_OPENAI_KEY"),
                    api_version="2024-12-01-preview",
                    timeout=request_timeout,
                    max_retries=max_retries,
                    http_client=httpx.Client(trust_env=False, timeout=http_timeout),
                )
            elif model in ["o3-pro"]:
                self.client = AzureOpenAI(
                    azure_endpoint = os.environ.get("AZURE_ENDPOINT"),
                    api_key=os.environ.get("AZURE_OPENAI_KEY"),
                    api_version="2025-03-01-preview",
                    timeout=request_timeout,
                    max_retries=max_retries,
                )             
            else:
                self.client = AzureOpenAI(
                    azure_endpoint = os.environ.get("AZURE_ENDPOINT"),
                    api_key=os.environ.get("AZURE_OPENAI_KEY"),
                    api_version="2024-05-01-preview",
                    timeout=request_timeout,
                    max_retries=max_retries,
                )

        self.messages = []
        self.model = model
        self.temperature = float(temperature)
        max_tokens_raw = os.environ.get("GRAPHLINK_MAX_TOKENS")
        self.max_tokens = int(max_tokens_raw) if max_tokens_raw else None
        
        # 📊 统计信息
        self.call_count = 0  # 总调用次数
        self.call_history = []  # 每次调用的详细信息
        self.total_prompt_chars = 0  # 总 prompt 字符数
        self.total_response_chars = 0  # 总 response 字符数

    def get_response(self, prompt) -> str:
        self.messages.append({"role": "user", "content": prompt})
        prompt_preview = prompt[:150].replace('\n', ' ') if len(prompt) > 150 else prompt.replace('\n', ' ')
        
        # 📊 记录调用开始时间和 prompt 信息
        call_start_time = time.time()
        prompt_length = len(prompt)
        prompt_tokens_estimate = prompt_length // 4  # 粗略估算 tokens
        
        try:
            if self.model in ["o3-pro"]:
                response = self.client.responses.create(
                    model=self.model,
                    input=self.messages,
                    temperature=self.temperature
                )
                main_content = response.output_text
            else:
                chat_kwargs = {
                    "model": self.model,
                    "messages": self.messages,
                    "temperature": self.temperature,
                }
                if self.max_tokens is not None:
                    chat_kwargs["max_tokens"] = self.max_tokens
                response = self.client.chat.completions.create(**chat_kwargs)
                main_content = response.choices[0].message.content
            self.messages.append({"role": "assistant", "content": main_content})
            
            # 📊 记录成功的调用统计
            call_end_time = time.time()
            response_length = len(main_content)
            response_tokens_estimate = response_length // 4
            
            self.call_count += 1
            self.total_prompt_chars += prompt_length
            self.total_response_chars += response_length
            
            # 提取真实 token 计数（来自 API 返回的 usage 字段）
            call_info = {
                "call_id": self.call_count,
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "prompt_length": prompt_length,
                "prompt_tokens_estimate": prompt_tokens_estimate,
                "response_length": response_length,
                "response_tokens_estimate": response_tokens_estimate,
                "total_tokens_estimate": prompt_tokens_estimate + response_tokens_estimate,
                "duration_seconds": round(call_end_time - call_start_time, 2),
                "model": self.model,
                "temperature": self.temperature,
                "prompt_preview": prompt[:200] + "..." if len(prompt) > 200 else prompt
            }
            # 优先使用 API 返回的真实 token 计数（vLLM / OpenAI 均支持 response.usage）
            try:
                usage = response.usage
                if usage is not None:
                    call_info["prompt_tokens"]     = usage.prompt_tokens
                    call_info["response_tokens"]   = usage.completion_tokens
                    call_info["total_tokens"]      = usage.total_tokens
            except Exception:
                pass  # usage 字段不可用时静默降级到 estimate
            self.call_history.append(call_info)
            
            return main_content
        except Exception as e:
            # 🚀 修复：如果 API 调用失败，回滚刚添加的 user message，避免重试时累积
            removed_msg = self.messages.pop()
            print(f"⚠️  API 调用失败，回滚消息:")
            print(f"  - 错误类型: {type(e).__name__}")
            print(f"  - 错误信息: {str(e)[:200]}")
            print(f"  - 回滚的 Prompt 长度: {len(removed_msg['content']):,} 字符 (≈{len(removed_msg['content'])//4:,} tokens)")
            print(f"  - Prompt 预览: {prompt_preview}...")
            print(f"  - 当前消息历史: {len(self.messages)} 条消息")
            raise e

    def get_model_response(self, prompt, code_format=None) -> list:
        code_blocks = []
        max_try = int(os.environ.get("GRAPHLINK_MAX_TRY", "3"))
        while code_blocks == [] and max_try > 0:
            max_try -= 1
            try:
                response = self.get_response(prompt)
            except Exception as e:
                print(f"max_try: {max_try}, exception: {e}")
                if max_try > 0:
                    print(f"Waiting 5 seconds before retry...")
                    time.sleep(5)
                continue
            code_blocks = extract_all_blocks(response, code_format)
        if code_blocks == []:
            print(f"get_model_response() exit, max_try: {max_try}, code_blocks: {code_blocks}")
            # 🚀 修复：不直接退出，而是抛出异常让调用方处理
            raise RuntimeError(f"Failed to get valid response after 3 tries. code_blocks: {code_blocks}")
            
        return code_blocks

    def get_model_response_txt(self, prompt):
        max_try = int(os.environ.get("GRAPHLINK_MAX_TRY", "3"))
        response = None
        while max_try > 0:
            max_try -= 1
            try:
                response = self.get_response(prompt)
            except Exception as e:
                print(f"max_try: {max_try}, exception: {e}")
                if max_try > 0:
                    print(f"Waiting 5 seconds before retry...")
                    time.sleep(5)
                continue
            break
        if response is None:
            print(f"get_model_response_txt() exit, max_try: {max_try}")
            # 🚀 修复：不直接退出，而是抛出异常让调用方处理
            raise RuntimeError(f"Failed to get valid response after 3 tries")
        
        return response

    def get_message_len(self):
        return {'prompt_len': sum(len(msg['content']) for msg in self.messages if msg['role'] == 'user'),
                'response_len': sum(len(msg['content']) for msg in self.messages if msg['role'] == 'assistant'),
                'num_calls': len([msg for msg in self.messages if msg['role'] == 'user'])}
    
    def get_message_stats(self):
        """获取消息统计信息（别名方法）"""
        return self.get_message_len()
    
    def clear_messages(self):
        """🚀 清理消息历史，释放内存"""
        self.messages.clear()
    
    def init_messages(self):
        """🚀 初始化/清理消息"""
        self.messages = []
    
    def get_call_statistics(self):
        """📊 获取调用统计信息"""
        if self.call_count == 0:
            return {
                "total_calls": 0,
                "total_prompt_chars": 0,
                "total_response_chars": 0,
                "total_tokens_estimate": 0,
                "average_prompt_chars": 0,
                "average_response_chars": 0,
                "average_tokens_estimate": 0
            }
        
        total_tokens = self.total_prompt_chars // 4 + self.total_response_chars // 4
        
        return {
            "total_calls": self.call_count,
            "total_prompt_chars": self.total_prompt_chars,
            "total_response_chars": self.total_response_chars,
            "total_tokens_estimate": total_tokens,
            "average_prompt_chars": self.total_prompt_chars // self.call_count,
            "average_response_chars": self.total_response_chars // self.call_count,
            "average_tokens_estimate": total_tokens // self.call_count,
            "model": self.model,
            "temperature": self.temperature
        }
    
    def save_statistics(self, output_file="gptchat_statistics.json", include_history=True):
        """📊 保存调用统计信息到文件
        
        Args:
            output_file: 输出文件路径
            include_history: 是否包含每次调用的详细历史（默认True）
        """
        stats = self.get_call_statistics()
        
        output_data = {
            "summary": stats,
            "generation_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        
        if include_history and self.call_history:
            output_data["call_history"] = self.call_history
        
        # 保存为JSON文件
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        
        print(f"📊 统计信息已保存到: {output_file}")
        print(f"📞 总调用次数: {stats['total_calls']}")
        print(f"📏 总输入字符数: {stats['total_prompt_chars']:,} (≈{stats['total_prompt_chars']//4:,} tokens)")
        print(f"📝 总输出字符数: {stats['total_response_chars']:,} (≈{stats['total_response_chars']//4:,} tokens)")
        print(f"🔢 总 tokens 估算: {stats['total_tokens_estimate']:,}")
        
        return output_file
    
    def print_statistics(self):
        """📊 打印调用统计信息"""
        stats = self.get_call_statistics()
        
        print("\n" + "=" * 80)
        print("📊 GPTChat 调用统计")
        print("=" * 80)
        print(f"🤖 模型: {stats.get('model', 'N/A')}")
        print(f"🌡️  温度: {stats.get('temperature', 'N/A')}")
        print(f"📞 总调用次数: {stats['total_calls']}")
        print(f"📏 总输入字符数: {stats['total_prompt_chars']:,} chars")
        print(f"   └─ 估算 tokens: ≈{stats['total_prompt_chars']//4:,} tokens")
        print(f"📝 总输出字符数: {stats['total_response_chars']:,} chars")
        print(f"   └─ 估算 tokens: ≈{stats['total_response_chars']//4:,} tokens")
        print(f"🔢 总 tokens 估算: {stats['total_tokens_estimate']:,} tokens")
        
        if stats['total_calls'] > 0:
            print(f"\n📊 平均值:")
            print(f"   - 平均 Prompt 长度: {stats['average_prompt_chars']:,} chars (≈{stats['average_prompt_chars']//4:,} tokens)")
            print(f"   - 平均 Response 长度: {stats['average_response_chars']:,} chars (≈{stats['average_response_chars']//4:,} tokens)")
            print(f"   - 平均每次调用 tokens: {stats['average_tokens_estimate']:,} tokens")
        
        print("=" * 80 + "\n")
