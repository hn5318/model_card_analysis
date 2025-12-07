from config import API_KEY, BASE_URL, MODEL_NAME
from loguru import logger
import json
import os
import openai
import pandas as pd
from typing import List, Dict, Any
from datetime import datetime


CLASSIFY_PROMPT_TEMPLATE = """
    Role:
    You are an expert on the task of ModelCard quality assessment.

    Task:
    Given the full text of a single ModelCard, classify it as either "High Quality" or "Low Quality" according to the criteria below.

    Classification Criteria:

    High Quality: The ModelCard contains ALL THREE core content themes:
    1. Model Overview: Provides an overview of the model's basic information, such as key specifications, functionality, and overall characteristics. Section titles may include 'Model Description', 'Model Details', 'Introduction', etc.
    2. Model Evaluation: Reports evaluation metrics and corresponding results. Section titles may include 'Evaluation' and 'Evaluation Results', etc.
    3. Model Usage Guide: Provides at least one runnable code example showing how to load or use the model. Section titles may include 'Usage', 'How to Use', 'How to Run', 'Example Code', etc.

    Low Quality: ANY core theme is missing, OR a theme is present only as a heading or contains only vague statements without the concrete elements listed above.

    Output Requirements:
    Respond with EXACTLY one of the two strings, case-sensitive, with no additional text, punctuation, or whitespace:
    - "High Quality"
    - "Low Quality"

    Input and Response:
    ModelCard text:{query}
    Selected category:
    """

# 配置日志系统
def setup_logger():
    """配置日志系统，仅输出到文件"""
    # 移除默认的日志处理器
    logger.remove()
    
    # 创建日志文件路径
    log_file_path = f"D:/Research/RQ2/llm_classification-main/classifier_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    
    # 添加文件输出
    logger.add(
        sink=log_file_path,
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {message}",
        level="INFO",
        rotation="100 MB",  # 当文件大小超过10MB时轮转
        retention="7 days",  # 保留7天的日志
        encoding="utf-8"
    )
    
    logger.info(f"日志系统初始化完成，日志文件: {log_file_path}")
    return log_file_path

class LLM():
    def __init__(self, api_key: str = None, base_url: str = None, model_name: str = None) -> None:
        try:
            self.api_key = api_key
            if not self.api_key:
                raise ValueError("API密钥未提供")
            
            self.model_name = model_name
            if not model_name:
                raise ValueError("模型名称未提供")
            
            self.base_url = base_url
            if not self.base_url:
                raise ValueError("API基础URL未提供")
            
            # 初始化 OpenAI 客户端
            self.client = openai.OpenAI(
                api_key = self.api_key,
                base_url = self.base_url
            )
            
            logger.info(f"模型初始化完成 - 模型: {model_name}")
        except Exception as e:
            logger.error(f"模型初始化失败: {e}")
            raise

    def chat(self, 
             messages: List[Dict[str, str]],
             ) -> str:

        try:
            # 调用API
            response = self.client.chat.completions.create(
                model = self.model_name,
                messages = messages,
                # max_tokens = 2
                # temperature = 0
            )
            
            # 提取回复内容
            reply_content = response.choices[0].message.content
            
            return reply_content
            
        except Exception as e:
            logger.error(f"API调用失败: {e}")
            raise

class LLMClassifier:
    def __init__(self) -> None:
        self.llm = LLM(api_key=API_KEY, base_url=BASE_URL, model_name=MODEL_NAME)

    def load_data_from_json(self, json_file_path: str) -> list:
        data = pd.read_json(json_file_path, lines=True, orient='records')
        return data

    def classify(self, query: str) -> str:
        task_description = CLASSIFY_PROMPT_TEMPLATE
        
        # LLM
        logger.info('大模型进行推理........')
        output = self.llm.chat([{
            'role': 'user', 
            'content': task_description.format(query = query)}])
        return output
    
if __name__ == '__main__':
    # 初始化日志系统
    log_file_path = setup_logger()
    logger.info("开始执行分类任务")
    
    try:
        lc = LLMClassifier()

        # 读取数据
        json_file_path = "D:/Research/RQ2/llm_classification-main/modelcard_data (update).json"
        # json_file_path = "D:/Research/RQ2/llm_classification-main/quality_1_data.json"
        data = lc.load_data_from_json(json_file_path)

        # data = data.head(10)

        # data = data[9:10]

        # data = data.tail(30)
        
        # # 从第664条记录开始分类
        # data = data[663:]

        logger.info(f"成功读取JSON文件，包含 {len(data)} 条记录")

        # 初始化结果列表
        results = []

        # 大模型分类
        logger.info("开始分类...")
        total_records = len(data)
        
        for index, row in data.iterrows():
            try:
                modelcard_text = row['modelcard_text']
                # modelcard_text = row['modelcard_text']

                logger.info(f"正在处理第 {index + 1}/{total_records} 条记录...")
                
                # 调用分类方法
                category = lc.classify(modelcard_text)
                
                # 构建结果字典
                result_item = {
                    'modelId': row['modelId'],
                    'modelcard_text_clean': modelcard_text,
                    'category': category
                }
                
                # 添加到结果列表
                results.append(result_item)
                
            except Exception as e:
                logger.error(f"处理第 {index + 1} 条记录时出错: {e}")
                # 即使出错也要记录，标记为错误
                result_item = {
                    'modelId': row['modelId'],
                    'modelcard_text_clean': row['modelcard_text_clean'],
                    'category': f"ERROR: {str(e)}"
                }
                results.append(result_item)
        
        # 保存结果到JSON文件
        result_file_path = "D:/Research/RQ2/llm_classification-main/result.json"
        
        # 将结果列表转换为DataFrame并保存为JSON
        result_df = pd.DataFrame(results)
        result_df.to_json(result_file_path, orient='records', lines=True)
        
        logger.success(f"分类完成！结果已保存到: {result_file_path}")
        logger.info(f"总共处理了 {len(results)} 条记录")
        
        # 统计分类结果
        category_counts = result_df['category'].value_counts()
        logger.info("分类结果统计:")
        logger.info(f"分类结果: {category_counts.to_dict()}")
        
    except Exception as e:
        logger.critical(f"程序执行过程中发生严重错误: {e}")
        raise
    finally:
        logger.info(f"程序执行完成，详细日志已保存到: {log_file_path}")

    
