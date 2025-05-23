import sys
from pathlib import Path

# Adiciona o diretório src ao sys.path
root = Path(__file__).resolve().parent / 'src'
sys.path.insert(0, str(root))

from models import multivariate
