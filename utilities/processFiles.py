from __future__ import annotations
from bs4 import BeautifulSoup
from pathlib import Path
import re
import os
import json
from pathlib import Path
import subprocess
import chardet
from typing import Union, List, Dict
import pandas as pd
from datetime import datetime
import shutil

from utilities.modelRelated import invoke_model

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

def detect_and_process_file_paths(user_input: str) -> list:
    """检测用户输入中的音频文件路径并验证文件是否存在，返回结果为用户上传的音频文件路径组成的数列"""
    file_paths = []
    processed_paths = set()  # Track already processed paths to avoid duplicates
    
    # 音频文件扩展名
    audio_extensions = r'(?:mp3|wav|flac|aac|ogg|m4a|wma|opus|mp4|avi|mov|mkv|webm|3gp)'
    
    # 改进的音频文件路径检测模式，支持中文字符
    # Windows路径模式 (C:\path\file.ext 或 D:\path\file.ext) - 仅匹配音频文件
    windows_pattern = rf'[A-Za-z]:[\\\\/](?:[^\\\\/\s\n\r]+[\\\\/])*[^\\\\/\s\n\r]+\.{audio_extensions}'
    # 相对路径模式 (./path/file.ext 或 ../path/file.ext) - 仅匹配音频文件
    relative_pattern = rf'\.{{1,2}}[\\\\/](?:[^\\\\/\s\n\r]+[\\\\/])*[^\\\\/\s\n\r]+\.{audio_extensions}'
    # 简单文件名模式 (filename.ext) - 仅匹配音频文件
    filename_pattern = rf'\b[a-zA-Z0-9_\u4e00-\u9fff\-\(\)（）]+\.{audio_extensions}\b'
    
    patterns = [windows_pattern, relative_pattern, filename_pattern]
    
    # Run the absolute path pattern first
    for match in re.findall(patterns[0], user_input, re.IGNORECASE):
        if match in processed_paths:
            continue
        processed_paths.add(match)
        _log_existence(match, file_paths)

    # Run the relative path pattern
    for match in re.findall(patterns[1], user_input, re.IGNORECASE):
        if match in processed_paths:
            continue
        processed_paths.add(match)
        _log_existence(match, file_paths)
        
    # Run the filename pattern if we didn't find any files
    if not file_paths:
        for match in re.findall(patterns[2], user_input, re.IGNORECASE):
            if match in processed_paths:
                continue
            processed_paths.add(match)
            _log_existence(match, file_paths)

    return file_paths


def store_uploaded_files(file_paths: list, session_id: str) -> list:
    """
    将上传的文件存储到指定的会话目录中
    
    Args:
        file_paths: 原始文件路径列表
        session_id: 会话ID
        
    Returns:
        存储后的文件路径列表
    """
    if not file_paths:
        return []
    
    # 创建目标目录
    target_dir = Path(f"conversations/{session_id}/user_uploaded_files")
    target_dir.mkdir(parents=True, exist_ok=True)
    
    stored_paths = []
    
    for file_path in file_paths:
        try:
            source_path = Path(file_path)
            if not source_path.exists():
                print(f"⚠️ 源文件不存在: {file_path}")
                continue
            
            # 生成目标文件路径
            target_path = target_dir / source_path.name
            
            # 如果目标文件已存在，添加时间戳避免覆盖
            if target_path.exists():
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                stem = source_path.stem
                suffix = source_path.suffix
                target_path = target_dir / f"{stem}_{timestamp}{suffix}"
            
            # 复制文件
            shutil.copy2(source_path, target_path)
            stored_paths.append(str(target_path))
            print(f"✅ 文件已存储: {source_path.name} -> {target_path}")
            
        except Exception as e:
            print(f"❌ 存储文件失败 {file_path}: {e}")
    
    return stored_paths


# -- 小工具函数 ------------------------------------------------------------
def _log_existence(path: str, container: list):
    if os.path.exists(path):
        container.append(path)
        print(f"✅ 检测到音频文件: {path}")
    else:
        print(f"⚠️ 音频文件路径无效或文件不存在: {path}")
