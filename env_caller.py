# env_caller.py
import os
import subprocess


def call_spacy_env(script_path, *args):
    """调用spacy环境中的脚本"""
    spacy_python = os.path.join("venv_spacy", "Scripts", "python.exe")
    cmd = [spacy_python, script_path] + list(args)
    result = subprocess.run(cmd, capture_output=True, text=True)
    return result.stdout, result.stderr, result.returncode


def call_langchain_env(script_path, *args):
    """调用langchain环境中的脚本"""
    langchain_python = os.path.join("venv_langchain", "Scripts", "python.exe")
    cmd = [langchain_python, script_path] + list(args)
    result = subprocess.run(cmd, capture_output=True, text=True)
    return result.stdout, result.stderr, result.returncode
