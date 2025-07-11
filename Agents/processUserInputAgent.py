import sys
from pathlib import Path
import json

# Add root project directory to sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent))



from typing import Dict, List, Optional, Any, TypedDict, Annotated, Union
from datetime import datetime

from utilities.modelRelated import invoke_model, invoke_model_with_tools
from utilities.processFiles import detect_and_process_file_paths, store_uploaded_files

from pathlib import Path
# Create an interactive chatbox using gradio
import gradio as gr
from dotenv import load_dotenv


from langgraph.graph import StateGraph, END, START
from langgraph.graph.message import add_messages
# from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import Command, interrupt
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool


class ProcessUserInputState(TypedDict):
    message: Annotated[List[BaseMessage], add_messages]
    user_input: str
    user_uploaded_files: List[str]
    text_input_validation: str
    previous_messages: List[BaseMessage]





class ProcessUserInputAgent:

    def __init__(self):
        self.memory = MemorySaver()
        self.graph = self._build_graph(self.memory)


    def _create_initial_state(self, session_id: str, previous_messages: List[BaseMessage]) -> ProcessUserInputState:
        return {
            "session_id": session_id,
            "previous_messages": previous_messages,
            "user_input": "",
            "user_uploaded_files": [],
            "text_input_validation": "",
        }
    

    def _build_graph(self, memory):
        graph = StateGraph(ProcessUserInputState)
        graph.add_node("collect_user_input", self._collect_user_input)
        graph.add_node("analyze_user_input_text", self._analyze_user_input_text)
    
        graph.add_edge(START, "collect_user_input")
        graph.add_conditional_edges("collect_user_input", self._route_after_collect_user_input)
        graph.add_edge("analyze_user_input_text", END)

        return graph.compile(memory)

    def _collect_user_input(self, state: ProcessUserInputState) -> ProcessUserInputState:
        """收集用户信息"""
        print("\n🔍 开始执行: _collect_user_input")
        print("=" * 50)
        print("⌨️ 等待用户输入...")


        user_input = interrupt("请输入用户信息")
        detcted_files = detect_and_process_file_paths(user_input)
        if detcted_files:
            print(f"📂 检测到用户上传的文件: {detcted_files}")
            stored_files = store_uploaded_files(detcted_files, state["session_id"])
            state["user_uploaded_files"] = stored_files



        print(f"📥 接收到用户输入: {user_input[:100]}{'...' if len(user_input) > 100 else ''}")
        print("✅ _collect_user_input 执行完成")
        print("=" * 50)
        

        return {"user_input": user_input}
    


    def _route_after_collect_user_input(self, state: ProcessUserInputState) -> ProcessUserInputState:
        if state["user_uploaded_files"]:
            return "process_uploaded_files"
        else:
            return "analyze_user_input_text"



    def _analyze_user_input_text(self, state: ProcessUserInputState) -> ProcessUserInputState:
        """This node performs a safety check on user text input when all uploaded files are irrelevant.
        It validates if the user input contains meaningful table/Excel-related content.
        Returns [Valid] or [Invalid] based on the analysis."""
        
        print("\n🔍 开始执行: _analyze_user_input_text")
        print("=" * 50)
        
        user_input = state["user_input"]
        print(f"📝 正在分析用户文本输入: {user_input[:100]}{'...' if len(user_input) > 100 else ''}")
        
        if not user_input or user_input.strip() == "":
            print("❌ 用户输入为空")
            print("✅ _analyze_text_input 执行完成")
            print("=" * 50)
            return {
                "text_input_validation": "[Invalid]",
                "process_user_input_messages": [SystemMessage(content="❌ 用户输入为空，验证失败")]
            }
        
        # Create validation prompt for text input safety check
        # Get the previous AI message content safely
        previous_ai_content = ""
        try:
            if state.get("previous_AI_messages"):
                previous_ai_messages = state["previous_AI_messages"]
                print(f"🔍 previous_AI_messages 类型: {type(previous_ai_messages)}")
                
                # Handle both single message and list of messages
                if isinstance(previous_ai_messages, list):
                    if len(previous_ai_messages) > 0:
                        latest_message = previous_ai_messages[-1]
                        if hasattr(latest_message, 'content'):
                            previous_ai_content = latest_message.content
                        else:
                            previous_ai_content = str(latest_message)
                        print(f"📝 从消息列表提取内容，长度: {len(previous_ai_content)}")
                    else:
                        print("⚠️ 消息列表为空")
                else:
                    # It's a single message object
                    if hasattr(previous_ai_messages, 'content'):
                        previous_ai_content = previous_ai_messages.content
                    else:
                        previous_ai_content = str(previous_ai_messages)
                    print(f"📝 从单个消息提取内容，长度: {len(previous_ai_content)}")
            else:
                print("⚠️ 没有找到previous_AI_messages")
                
        except Exception as e:
            print(f"❌ 提取previous_AI_messages内容时出错: {e}")
            previous_ai_content = ""
            
        print(f"上一轮ai输入内容：=========================================\n{previous_ai_content}")
        system_prompt = f"""
你是一位专业的输入验证专家，任务是判断用户的文本输入是否与**表格生成或 Excel 处理相关**，并且是否在当前对话上下文中具有实际意义。

你将获得以下两部分信息：
- 上一轮 AI 的回复（用于判断上下文是否连贯）
- 当前用户的输入内容

请根据以下标准进行判断：

【有效输入 [Valid]】满足以下任一条件即可视为有效：
- 明确提到生成表格、填写表格、Excel 处理、数据整理等相关操作
- 提出关于表格字段、数据格式、模板结构等方面的需求或提问
- 提供表格相关的数据内容、字段说明或规则
- 对上一轮 AI 的回复作出有意义的延续或回应（即使未直接提到表格）
- 即使存在错别字、语病、拼写错误，只要语义清晰合理，也视为有效

【无效输入 [Invalid]】符合以下任一情况即视为无效：
- 内容与表格/Excel 完全无关（如闲聊、情绪表达、与上下文跳脱）
- 明显为测试文本、随机字符或系统调试输入（如 "123"、"测试一下"、"哈啊啊啊" 等）
- 仅包含空白、表情符号、标点符号等无实际内容

【输出要求】
请你根据上述标准，**仅输出以下两种结果之一**（不添加任何其他内容）：
- [Valid]
- [Invalid]

【上一轮 AI 的回复】
{previous_ai_content}
"""



        
        try:
            print("📤 正在调用LLM进行文本输入验证...")
            # Get LLM validation
            user_input = "用户输入：" + user_input
            print("analyze_text_input时调用模型的输入: \n" + user_input)              
            validation_response = invoke_model(model_name="Pro/deepseek-ai/DeepSeek-V3", messages=[SystemMessage(content=system_prompt), HumanMessage(content=user_input)])
            # validation_response = self.llm_s.invoke([SystemMessage(content=system_prompt)])
            
            print(f"📥 验证响应: {validation_response}")
            
            if "[Valid]" in validation_response:
                validation_result = "[Valid]"
                status_message = "用户输入验证通过 - 内容与表格相关且有意义"
            elif "[Invalid]" in validation_response:
                validation_result = "[Invalid]"
                status_message = "用户输入验证失败 - 内容与表格无关或无意义"
            else:
                # Default to Invalid for safety
                validation_result = "[Invalid]"
                status_message = "用户输入验证失败 - 无法确定输入有效性，默认为无效"
                print(f"⚠️ 无法解析验证结果，LLM响应: {validation_response}")
            
            print(f"📊 验证结果: {validation_result}")
            print(f"📋 状态说明: {status_message}")
            
            # Create validation summary
            summary_message = f"""文本输入安全检查完成:
            
            **用户输入**: {user_input[:100]}{'...' if len(user_input) > 100 else ''}
            **验证结果**: {validation_result}
            **状态**: {status_message}"""
            
            print("✅ _analyze_text_input 执行完成")
            print("=" * 50)
            
            return {
                "text_input_validation": validation_result,
                "messages": [SystemMessage(content=summary_message)]
            }
                
        except Exception as e:
            print(f"❌ 验证文本输入时出错: {e}")
            
            # Default to Invalid for safety when there's an error
            error_message = f"""❌ 文本输入验证出错: {e}
            
            📄 **用户输入**: {user_input[:100]}{'...' if len(user_input) > 100 else ''}
            🔒 **安全措施**: 默认标记为无效输入"""
            
            print("✅ _analyze_text_input 执行完成 (出错)")
            print("=" * 50)
            
            return {
                "text_input_validation": "[Invalid]",
                "process_user_input_messages": [SystemMessage(content=error_message)]
            }

    def _route_after_analyze_user_input_text(self, state: ProcessUserInputState) -> ProcessUserInputState:
        if state["text_input_validation"] == "[Valid]":
            return END
        else:
            return "collect_user_input"

    def run_process_user_input(self, session_id: str, previous_messages: List[BaseMessage]) -> ProcessUserInputState:
        """运行用户输入处理流程"""
        print("\n🚀 开始运行 ProcessUserInputAgent")
        print("=" * 60)

        inital_state = self._create_initial_state(session_id, previous_messages)
        config = {"configurable": {"thread_id": session_id}}

        print(f"📋 会话ID: {session_id}")
        print(f"📝 初始状态已创建")
        print("🔄 正在执行用户输入处理工作流...")
        try:
            while True:
                finial_state =  self.graph.invoke(inital_state, config )
                if "__interrupted__" in finial_state:
                    interrupt_value = finial_state["__interrupted__"][0].value
                    print(f"💬 智能体: {interrupt_value}")
                    user_response = input("请输入用户响应: ")
                    inital_state = Command(resume=user_response)
                    continue

                print("🎉执行完毕")
                return finial_state
            
        except Exception as e:
            print(f"❌ 运行用户输入处理流程时出错: {e}")

            return None
        
    

        
        
        
        
    
    

