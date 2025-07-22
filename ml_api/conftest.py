# conftest.py
import os, sys
# Insere o diretório raiz (onde está este conftest.py) no início do sys.path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))