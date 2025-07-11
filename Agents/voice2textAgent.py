import sys
from pathlib import Path
import json

# Add root project directory to sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent))



from typing import Dict, List, Optional, Any, TypedDict, Annotated, Union
from datetime import datetime

from utilities.modelRelated import invoke_model, invoke_model_with_tools

from pathlib import Path
# Create an interactive chatbox using gradio
import gradio as gr
from dotenv import load_dotenv


from langgraph.graph import StateGraph, END, START
from langgraph.graph.message import add_messages
# from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import Command
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool

# Import other agents
from Agents.processUserInputAgent import ProcessUserInputAgent

class Voice2TextState(TypedDict):
    audio_file_path: str
    user_input: str
    session_id: str
    previous_messages: List[BaseMessage]






class Voice2TextAgent:
    def __init__(self):
        self.graph = self._build_graph()

    def _build_graph(self, memory = MemorySaver() ):
        graph = StateGraph(Voice2TextState)
        graph.add_node("collect_user_input", self._collect_user_input)
        graph.add_node("transcribe_audio", self._transcribe_audio)
        graph.add_node("analyze_transcribed_audio", self._analyze_transcribed_audio)
        graph.add_node("chat_with_user", self._chat_with_user)

        graph.add_edge(START, "collect_user_input")
        graph.add_edge("collect_user_input", "transcribe_audio")
        graph.add_edge("transcribe_audio", "process_transcribed_audio")
        graph.add_edge("process_transcribed_audio", "chat_with_user")
        graph.add_edge("chat_with_user", END)
        return graph.compile(memory)
    
    def _create_initial_state(self, session_id: str, previous_messages: List[BaseMessage] = None) -> Voice2TextState:
        return {
            "session_id": session_id,
            "previous_messages": previous_messages,
            "user_input": "",
            "audio_file_path": "",
        }

    def _collect_user_input(self, state: Voice2TextState) -> Voice2TextState:
        processUserInputAgent = ProcessUserInputAgent()
        finial_state = processUserInputAgent.run_process_user_input(state["session_id"], state["previous_messages"])
        user_input = finial_state["user_input"]
        user_uploaded_files = finial_state["user_uploaded_files"]
        return {"user_input": user_input, "audio_file_path": user_uploaded_files}

    def _transcribe_audio(self, state: Voice2TextState) -> Voice2TextState:
        system_prompt = """"aa"""

        invoke_model(
            model_name="whisper-1",
            messages=[
                SystemMessage(content=system_prompt),
                HumanMessage(content=state["audio_file_path"])
            ]
        )

    def _analyze_transcribed_audio(self, state: Voice2TextState) -> Voice2TextState:
        pass

    def _chat_with_user(self, state: Voice2TextState) -> Voice2TextState:
        pass
