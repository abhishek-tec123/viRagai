# # with file uploader class test with exl and upload with fast api===========================================================

import os
import pdfplumber
from docx import Document as DocxDocument
import tiktoken
import pandas as pd
import requests
from bs4 import BeautifulSoup
from io import BytesIO

class FileUploader:
    def __init__(self, model="cl100k_base"):
        self.model = model
        os.environ["TOKENIZERS_PARALLELISM"] = "false"

    def count_tokens(self, text):
        encoder = tiktoken.get_encoding(self.model)
        return len(encoder.encode(text))

    def extract_text_from_upload(self, filename: str, file_obj):
        ext = os.path.splitext(filename)[1].lower()

        if ext == ".txt":
            return file_obj.read().decode("utf-8")

        elif ext == ".pdf":
            text = ""
            with pdfplumber.open(BytesIO(file_obj.read())) as pdf:
                for page in pdf.pages:
                    text += page.extract_text() or ""
            return text

        elif ext == ".docx":
            return "\n".join([para.text for para in DocxDocument(BytesIO(file_obj.read())).paragraphs])

        elif ext in [".xlsx", ".xls"]:
            df = pd.read_excel(BytesIO(file_obj.read()), sheet_name=None)
            text = ""
            for sheet_name, sheet_df in df.items():
                text += f"--- Sheet: {sheet_name} ---\n"
                text += sheet_df.to_string(index=False) + "\n\n"
            return text

        else:
            raise ValueError(f"Unsupported file format: {ext}")


    def extract_text_from_url(self, url):
        print(f"Scraping URL: {url}")
        headers = {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/121.0.0.0 Safari/537.36"
            )
        }
        response = requests.get(url, headers=headers)
        response.raise_for_status()

        soup = BeautifulSoup(response.text, "html.parser")

        # Remove script and style elements
        for tag in soup(["script", "style"]):
            tag.decompose()

        text = soup.get_text(separator="\n")
        cleaned_text = "\n".join(line.strip() for line in text.splitlines() if line.strip())

        print(f"Extracted {self.count_tokens(cleaned_text)} tokens from URL.")
        return cleaned_text


    def load_file(self, file_path_or_url):
        if file_path_or_url.startswith("http://") or file_path_or_url.startswith("https://"):
            return self.extract_text_from_url(file_path_or_url)

        if not os.path.exists(file_path_or_url):
            raise FileNotFoundError(f"{file_path_or_url} does not exist.")

        with open(file_path_or_url, "rb") as f:
            filename = os.path.basename(file_path_or_url)
            text = self.extract_text_from_upload(filename, f)
            print(f"Loaded '{file_path_or_url}' with {self.count_tokens(text)} tokens.")
            return text
