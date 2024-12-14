# app.py

import gradio as gr
import pandas as pd
from rag_system.core import RAGSystem
from rag_system.config import Config
import logging
from rag_system.loaders import load_documents
# 初始化RAG系统实例
import os

# Set no_proxy environment variable
# 
logging.basicConfig(
    level=logging.ERROR,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("app.log"),
        logging.StreamHandler()  # This will also print logs to the console
    ]
)
rag_system = RAGSystem(proxy="http://localhost:7890")

# 初始化系统（默认索引类型）
rag_system.initialize_system(index_type=Config.VECTOR_STORE_TYPE, num_clusters=Config.NUM_CLUSTERS)

def upload_csv_and_update(file, index_type, num_clusters):
    if file is None:
        return "请上传一个CSV文件。", None, None
    try:
        # 读取CSV文件
        df = pd.read_csv(file.name)
        documents=load_documents(file.name, Config.FILE_EXTENSION, Config.ENCODING)
        
        # 添加新文档到系统
        rag_system.add_documents(documents)
        
        # 重新初始化向量存储（如果需要更改索引类型）
        if index_type != Config.VECTOR_STORE_TYPE:
            rag_system.vector_store = None  # 重置向量存储
            rag_system.initialize_system(index_type=index_type, num_clusters=num_clusters)
        
        return "CSV文件已成功上传并更新向量存储。", rag_system, rag_system.llm
    except Exception as e:
        return f"上传CSV文件时出错: {str(e)}", None, None

def perform_query(query):
    logging.debug(f"Query: {query}")
    if not rag_system.qa_chain:
        return "系统尚未初始化。请先上传CSV文件并选择索引类型。", ""
    try:
        logging.debug("Querying...")
        answer, sources = rag_system.query(query)
        logging.debug(f"Answer: {answer}")
        return answer, sources
    except Exception as e:
        return f"查询时出错: {str(e)}", ""

def add_record(id, name, description, price):
    try:
        # 创建新记录的字典
        new_record = {
            "id": id,
            "name": name,
            "description": description,
            "price": price
            # 根据实际数据集添加更多字段
        }
        rag_system.add_documents([new_record])
        return "记录已成功添加。"
    except Exception as e:
        return f"添加记录时出错: {str(e)}"

def delete_record(criteria):
    try:
        rag_system.delete_documents(criteria)
        return "记录已成功删除。"
    except Exception as e:
        return f"删除记录时出错: {str(e)}"

def update_record(criteria, new_values):
    try:
        rag_system.update_documents(criteria, new_values)
        return "记录已成功更新。"
    except Exception as e:
        return f"更新记录时出错: {str(e)}"
os.environ["no_proxy"] = "localhost,127.0.0.1,::1"
with gr.Blocks() as demo:
    gr.Markdown("# LLM + RAG 数据库管理界面")
    
    with gr.Tab("上传CSV文件"):
        with gr.Row():
            csv_upload = gr.File(label="上传CSV文件", file_types=["csv"])
            index_choice = gr.Radio(choices=["Flat", "IVF", "HNSW", "IVFPQ"], label="选择索引类型", value=Config.VECTOR_STORE_TYPE)
            num_clusters = gr.Number(label="聚类数（仅IVF有效）", value=Config.NUM_CLUSTERS)
        upload_button = gr.Button("上传并更新向量存储")
        upload_status = gr.Textbox(label="状态", interactive=False)
        upload_button.click(upload_csv_and_update, inputs=[csv_upload, index_choice, num_clusters], outputs=[upload_status, gr.State(), gr.State()])
    
    with gr.Tab("执行查询"):
        with gr.Row():
            query_input = gr.Textbox(label="输入查询", placeholder="请输入您的问题...")
            query_button = gr.Button("执行查询")
        query_output = gr.Textbox(label="回答", interactive=False)
        source_output = gr.Textbox(label="相关文档来源", interactive=False)
        query_button.click(perform_query, inputs=query_input, outputs=[query_output, source_output])
    
    # with gr.Tab("增删改查"):
    #     with gr.Accordion("添加记录", open=False):
    #         with gr.Column():
    #             # 根据实际数据集调整字段
    #             id_input = gr.Textbox(label="ID")
    #             name_input = gr.Textbox(label="名称")
    #             description_input = gr.Textbox(label="描述")
    #             price_input = gr.Number(label="价格")
    #             add_button = gr.Button("添加记录")
    #             add_status = gr.Textbox(label="状态", interactive=False)
    #             add_button.click(add_record, inputs=[id_input, name_input, description_input, price_input], outputs=add_status)
        
    #     with gr.Accordion("删除记录", open=False):
    #         with gr.Column():
    #             criteria_input = gr.Textbox(label="删除条件（例如: id=123）")
    #             delete_button = gr.Button("删除记录")
    #             delete_status = gr.Textbox(label="状态", interactive=False)
    #             delete_button.click(delete_record, inputs=criteria_input, outputs=delete_status)
        
    #     with gr.Accordion("更新记录", open=False):
    #         with gr.Column():
    #             criteria_update = gr.Textbox(label="更新条件（例如: id=123）")
    #             new_values_input = gr.Textbox(label="新值（例如: price=99.99）")
    #             update_button = gr.Button("更新记录")
    #             update_status = gr.Textbox(label="状态", interactive=False)
    #             update_button.click(update_record, inputs=[criteria_update, new_values_input], outputs=update_status)

    # gr.Markdown("""
    # ---
    # **注意**：CRUD 操作需要在 `RAGSystem` 类中实现具体的逻辑。本示例仅提供接口框架，请根据实际需求补充 `delete_documents` 和 `update_documents` 方法的实现。
    # """)

demo.launch(debug=True,share=False, server_port=8600)
