import os
import pdfplumber

def extract_text(pdf_path):
    text = ""
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text(x_tolerance=1.5, y_tolerance=3)
            if page_text:
                text += page_text + "\n"
    return text

# divide the text in chunks of 500 words
# the overlap is used to not truncate the meaning of the sentences
def chunk_text(text, chunk_size=500, overlap=50):
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = start + chunk_size
        chunk = " ".join(words[start:end])
        chunks.append(chunk)
        start += chunk_size - overlap
    return chunks

def process_folder(folder_path):
    all_chunks = []
    for filename in os.listdir(folder_path):
        if filename.endswith(".pdf"):
            text = extract_text(os.path.join(folder_path, filename))
            chunks = chunk_text(text)
            for i, chunk in enumerate(chunks):
                all_chunks.append({
                    "chunk_id": f"{filename}_{i}",
                    "text": chunk,
                    "source": filename
                })
    return all_chunks