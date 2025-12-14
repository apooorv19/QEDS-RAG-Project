import json
import os
import time
import textwrap
import threading
import queue
from langchain_community.chat_models import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# --- CONFIGURATION ---
llm = ChatOllama(
    model="llama3", 
    temperature=0.1, 
    num_ctx=4096,
)

CLEANING_TEMPLATE = """
SYSTEM INSTRUCTION:
Rewrite the OCR text to make it readable.
1. FIX typos.
2. MERGE broken lines.
3. LEAVE MATH/EQUATIONS ALONE.
4. OUTPUT ONLY THE CLEANED TEXT.

INPUT:
{text}

CLEANED:
"""

prompt = ChatPromptTemplate.from_template(CLEANING_TEMPLATE)
chain = prompt | llm | StrOutputParser()

def extract_text_from_json(json_path):
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        all_pages_text = []
        if isinstance(data, dict):
            for key, pages in data.items():
                if isinstance(pages, list):
                    for page in pages:
                        page_lines = []
                        if 'text_lines' in page:
                            for line_obj in page['text_lines']:
                                if 'text' in line_obj:
                                    page_lines.append(line_obj['text'])
                        if page_lines:
                            all_pages_text.append("\n".join(page_lines))
        return "\n\n".join(all_pages_text)
    except Exception:
        return ""

def run_llm_thread_safe(text_chunk):
    
    result_queue = queue.Queue()

    def worker():
        try:
            cleaned = chain.invoke({"text": text_chunk})
            if "Here is" in cleaned:
                cleaned = cleaned.split("\n", 1)[-1].strip()
            result_queue.put(cleaned)
        except Exception:
            result_queue.put(None)

    # Start the thread
    t = threading.Thread(target=worker)
    t.daemon = True
    t.start()

    t.join(timeout=20)

    if t.is_alive():
        return None 
    else:
        try:
            return result_queue.get_nowait()
        except queue.Empty:
            return None

def process_single_file(input_path, output_path):
    base_name = os.path.basename(os.path.dirname(input_path))
    if os.path.exists(output_path):
        print(f"⏩ Skipping {base_name} (Exists)")
        return

    print(f"\n📂 Processing: {base_name}")
    raw_text = extract_text_from_json(input_path)
    if not raw_text: return

    primary_chunks = raw_text.split('\n\n')
    cleaned_final_parts = []
    total_chunks = len(primary_chunks)

    print(f"   🔹 Blocks to process: {total_chunks}")

    for i, chunk in enumerate(primary_chunks):
        chunk = chunk.strip()
        if len(chunk) < 5: continue

        # UI Update
        print(f"      Processing Block {i+1}/{total_chunks}...", end="\r")

        # 1. LARGE CHUNK LOGIC
        if len(chunk) > 1000:
            sub_segments = textwrap.wrap(chunk, width=800, break_long_words=False)
            processed_segments = []
            
            for sub in sub_segments:
                result = run_llm_thread_safe(sub)
                if result is None:
                    processed_segments.append(sub) # Keep original
                else:
                    processed_segments.append(result)
            
            cleaned_final_parts.append(" ".join(processed_segments))

        # 2. NORMAL CHUNK LOGIC
        else:
            result = run_llm_thread_safe(chunk)
            if result is None:
                # TIMEOUT HAPPENED HERE
                print(f"      ⚠️ SKIPPED Block {i+1} (Timeout 20s) - Keeping Original")
                cleaned_final_parts.append(chunk)
            else:
                cleaned_final_parts.append(result)

    # Save
    final_text = "\n\n".join(cleaned_final_parts)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(final_text)
    print(f"\n   ✅ Done: {os.path.basename(output_path)}")

def traverse_and_clean(source_root, dest_root):
    print(f"🚀 Starting traversal from: {source_root}\n")
    for root, dirs, files in os.walk(source_root):
        if "results.json" in files:
            relative_path = os.path.relpath(root, source_root)
            target_dir = os.path.join(dest_root, relative_path)
            input_file = os.path.join(root, "results.json")
            module_name = os.path.basename(root)
            output_file = os.path.join(target_dir, f"{module_name}_CLEANED.txt")
            
            process_single_file(input_file, output_file)

if __name__ == "__main__":
    # --- CONFIGURATION ---
    START_DIR = r"D:\QEDS RAG Project\notes-llm\data\processed_notes"
    OUTPUT_DIR = r"D:\QEDS RAG Project\notes-llm\data\OCR text"
    
    if os.path.exists(START_DIR):
        traverse_and_clean(START_DIR, OUTPUT_DIR)
        print("\n🎉 All files processed!")
    else:
        print(f"❌ Error: Start path not found: {START_DIR}")