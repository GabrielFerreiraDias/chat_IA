import os
import re
from pathlib import Path

def read_txt(path):
    return Path(path).read_text(encoding="utf-8", errors="ignore")

def read_pdf(path):
    from PyPDF2 import PdfReader
    reader = PdfReader(path)
    text = ""
    for page in reader.pages:
        extracted = page.extract_text()
        if extracted:
            text += extracted + "\n"
    return text

def read_docx(path):
    from docx import Document
    doc = Document(path)
    text = ""
    for para in doc.paragraphs:
        if para.text.strip():
            text += para.text + "\n"
    return text

def _looks_like_toc(line: str) -> bool:
    # Ex.: "1. Proprietário 2", "2. 1. Âmbito 3"
    if re.search(r'\s\d+\s*$', line):
        return True
    if re.match(r'^(\d+(\.\d+)+)\s+', line):
        return True
    if line.count('.') > 6:
        return True
    return False

def _looks_like_reference(line: str) -> bool:
    # Remove referências, citações, notas de rodapé
    if re.match(r'^(Ref|Referência|Fonte|Figura|Fig|Tabela|Table|Image|Imagem|Apêndice|Anexo)[\s:.]', line, re.IGNORECASE):
        return True
    if re.match(r'^\[\d+\]', line):  # [1], [2], etc
        return True
    if re.match(r'^http', line):  # URLs
        return True
    if re.match(r'^www\.', line):  # Websites
        return True
    if 'mailto:' in line.lower():
        return True
    return False

def _is_metadata(line: str) -> bool:
    # Remove metadados: data, versão, página, etc
    if re.match(r'^(Versão|Version|Data|Date|Autor|Author|Páginas?|Pages?|Página)[\s:.]', line, re.IGNORECASE):
        return True
    if re.match(r'^\d{1,2}/\d{1,2}/\d{4}', line):  # Datas
        return True
    if re.match(r'^v\d+\.\d+', line):  # Versions
        return True
    if 'página' in line.lower() or 'page' in line.lower():
        if re.search(r'\d+\s*(?:de|of)?\s*\d+', line):
            return True
    return False

def _is_figure_caption(line: str) -> bool:
    # Remove legendas de figuras/tabelas
    if re.match(r'^(Figura|Figure|Fig\.|Tabela|Table|Gráfico|Chart|Imagem|Image)[\s\d:.]', line, re.IGNORECASE):
        return True
    return False

def clean_text(text):
    lines = text.split('\n')
    cleaned = []

    for line in lines:
        line = line.strip()
        if not line:
            continue

        # Remove linhas tipo índice/sumário
        if _looks_like_toc(line):
            continue

        # Remove referências
        if _looks_like_reference(line):
            continue

        # Remove metadados
        if _is_metadata(line):
            continue

        # Remove legendas de figuras
        if _is_figure_caption(line):
            continue

        # Remove linhas muito curtas (menos de 20 caracteres)
        if len(line) < 20:
            continue

        # Remove linhas com muitos números/símbolos (provavelmente tabelas)
        if sum(c.isdigit() for c in line) > len(line) * 0.5:
            continue

        # Remove linhas com poucos caracteres alfabéticos
        alpha_ratio = sum(c.isalpha() for c in line) / max(len(line), 1)
        if alpha_ratio < 0.5:
            continue

        # Remove linhas que parecem rodapé ou cabeçalho
        if len(line) < 30 and (line.isupper() or line.count(' ') < 2):
            continue

        cleaned.append(line)

    text = " ".join(cleaned)
    # Remove espaços múltiplos
    text = re.sub(r'\s+', ' ', text).strip()
    # Remove caracteres especiais desnecessários
    text = re.sub(r'[\[\]\{\}]', '', text)
    
    return text

class DocumentProcessor:
    def __init__(self, docs_path="aprendizado"):
        self.docs_path = docs_path
        self.chunks = []

    def process_all_documents(self):
        self.chunks = []
        for root, _, files in os.walk(self.docs_path):
            for name in files:
                path = os.path.join(root, name)
                ext = os.path.splitext(name)[1].lower()

                try:
                    if ext == ".txt":
                        text = read_txt(path)
                    elif ext == ".pdf":
                        text = read_pdf(path)
                    elif ext in [".docx", ".doc"]:
                        text = read_docx(path)
                    else:
                        continue

                    text = clean_text(text)

                    if len(text) > 100:
                        self._add_chunks(text, source=path)
                        print(f"✓ Processado: {name}")

                except Exception as e:
                    print(f"✗ Erro em {name}: {e}")
                    continue

        return self.chunks

    def _add_chunks(self, text, source, chunk_size=1200, overlap=200):
        # Quebra por tamanho (caracteres), mantendo contexto
        i = 0
        while i < len(text):
            chunk = text[i:i+chunk_size]
            if len(chunk) > 200:
                self.chunks.append({"text": chunk, "source": source})
            i += chunk_size - overlap

    def get_chunks(self):
        return self.chunks