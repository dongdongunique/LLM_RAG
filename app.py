# app.py

import gradio as gr
import pandas as pd
from rag_system.core import RAGSystem
from rag_system.config import Config
import logging
from rag_system.loaders import load_documents, Document
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
        file_num = len(df)
        documents=load_documents(file.name, Config.FILE_EXTENSION, Config.ENCODING)
        
        # 添加新文档到系统
        rag_system.add_documents(documents)
        
        # 重新初始化向量存储（如果需要更改索引类型）
        if index_type != Config.VECTOR_STORE_TYPE:
            rag_system.vector_store = None  # 重置向量存储
            rag_system.initialize_system(index_type=index_type, num_clusters=num_clusters)
        
        return "CSV文件已成功上传并更新向量存储。", rag_system, rag_system.llm, file_num
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

def add_record(id_input, date_inpu, title_input, description_input, name_input, main_categories_input, categories_input, store_input, ave_rating_input, rating_num_input, price_input):
    try:
        # 创建新记录的字典
        new_record = {
            "parent_asin": id_input,
            "date_first_available": date_inpu,
            "title": title_input,
            "description": description_input,
            "filename": name_input,
            "main_category": main_categories_input,
            "categories": categories_input,
            "store": store_input,
            "average_rating": ave_rating_input,
            "rating_number": rating_num_input,
            "price": price_input,
        }
        row_text = ' | '.join([f"{key}: {value}" for key, value in new_record.items() if value is not None])
        metadata = {
            "source": "./amazon_products.csv",
            "row": rag_system.total_num,  # 这里的 'row' 取固定值，如果动态生成，请传入实际的行号
        }
        rag_system.total_num = rag_system.total_num + 1
        new_record = Document(page_content=row_text, metadata=metadata)
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

# 核心 UI 构建
os.environ["no_proxy"] = "localhost,127.0.0.1,::1"
with gr.Blocks() as demo:
    # 顶栏样式
    gr.HTML("""
    <style>
        #header {
            background: linear-gradient(90deg, #4CAF50, #008CBA);  /* 渐变色 */
            padding: 10px 0;
            text-align: center;
            color: white;
            font-size: 32px;
            font-weight: bold;
            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);  /* 阴影效果 */
        }
        
    </style>
    """)

    # 顶栏
    gr.Markdown("<div id='header'>LLM + RAG 数据库管理界面</div>", elem_id="header")

    # 上传 CSV Tab
    with gr.Tab("上传csv文件", elem_id="tab-upload"):
        with gr.Row():
            csv_upload = gr.File(label="上传CSV文件", file_types=["csv"], elem_id="csv-upload")
            index_choice = gr.Radio(choices=["Flat", "IVF", "HNSW", "IVFPQ"], label="选择索引类型", value=Config.VECTOR_STORE_TYPE, elem_id="index-choice")
            num_clusters = gr.Number(label="聚类数（仅IVF有效）", value=Config.NUM_CLUSTERS, elem_id="num-clusters")
        upload_button = gr.Button("上传并更新向量存储", variant="primary", elem_id="upload-button")
        upload_status = gr.Textbox(label="状态", interactive=False, elem_id="upload-status")
        upload_button.click(upload_csv_and_update, inputs=[csv_upload, index_choice, num_clusters], outputs=[upload_status, gr.State(), gr.State()])

    # 执行查询 Tab
    with gr.Tab("执行查询", elem_id="tab-query"):
        with gr.Row():
            query_input = gr.Textbox(label="输入查询", placeholder="请输入您的问题...", elem_id="query-input")
            query_button = gr.Button("执行查询", variant="primary", elem_id="query-button")
        query_output = gr.Textbox(label="回答", interactive=False, elem_id="query-output")
        source_output = gr.Textbox(label="相关文档来源", interactive=False, elem_id="source-output")
        query_button.click(perform_query, inputs=query_input, outputs=[query_output, source_output])

    # CRUD 操作 Tab
    with gr.Tab("增删改查", elem_id="tab-crud"):
        with gr.Accordion("添加记录", open=False):
            with gr.Column():
                id_input = gr.Textbox(label="ID", elem_id="id-input")
                date_inpu = gr.Textbox(label="日期", elem_id="date-input")
                title_input = gr.Textbox(label="标题", elem_id="title-input")
                description_input = gr.Textbox(label="描述", elem_id="description-input")
                name_input = gr.Textbox(label="名称", elem_id="name-input")
                main_categories_input = gr.Textbox(label="主要类别", elem_id="main-category-input")
                categories_input = gr.Textbox(label="类别", elem_id="category-input")
                store_input = gr.Textbox(label="商店", elem_id="store-input")
                ave_rating_input = gr.Textbox(label="平均评分", elem_id="rating-input")
                rating_num_input = gr.Textbox(label="评分数量", elem_id="rating-num-input")
                price_input = gr.Textbox(label="价格", elem_id="price-input")
                add_button = gr.Button("添加记录", variant="primary", elem_id="add-record-button")
                add_status = gr.Textbox(label="状态", interactive=False)
                add_button.click(add_record, inputs=[id_input, date_inpu, title_input, description_input, name_input, main_categories_input, categories_input, store_input, ave_rating_input, rating_num_input, price_input], outputs=add_status)
        
        with gr.Accordion("删除记录", open=False):
            with gr.Column():
                criteria_input = gr.Textbox(label="删除条件（例如: id=123）", elem_id="criteria-input")
                delete_button = gr.Button("删除记录", variant="secondary", elem_id="delete-record-button")
                delete_status = gr.Textbox(label="状态", interactive=False)
                delete_button.click(delete_record, inputs=criteria_input, outputs=delete_status)
        
        with gr.Accordion("更新记录", open=False):
            with gr.Column():
                criteria_update = gr.Textbox(label="更新条件（例如: id=123）", elem_id="update-criteria-input")
                new_values_input = gr.Textbox(label="新值（例如: price=99.99）", elem_id="update-values-input")
                update_button = gr.Button("更新记录", variant="tertiary", elem_id="update-record-button")
                update_status = gr.Textbox(label="状态", interactive=False)
                update_button.click(update_record, inputs=[criteria_update, new_values_input], outputs=update_status)

    # 最后展示注意事项
    gr.Markdown("""
    ---
    **注意**：CRUD 操作需要在 `RAGSystem` 类中实现具体的逻辑。本示例仅提供接口框架，请根据实际需求补充 `delete_documents` 和 `update_documents` 方法的实现。
    """)

# 启动应用
demo.launch(debug=True, share=False, server_port=8600)