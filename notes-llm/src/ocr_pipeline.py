from pathlib import Path
import json
import subprocess


# ----- Paths -----

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"

RAW_DIR = DATA_DIR / "raw_notes"         # Semester folders
SURYA_DIR = DATA_DIR / "processed_notes" # Surya JSON output
OCR_TEXT_DIR = DATA_DIR / "ocr_text"     # Final plain text

SURYA_DIR.mkdir(parents=True, exist_ok=True)
OCR_TEXT_DIR.mkdir(parents=True, exist_ok=True)


# ----- Helpers -----

def run_surya_on_file(pdf_path: Path, semester_name: str, course_name: str):

    pdf_stem = pdf_path.stem

    # Base output dir
    course_out_dir = SURYA_DIR / semester_name / course_name
    course_out_dir.mkdir(parents=True, exist_ok=True)

    pdf_results_dir = course_out_dir / pdf_stem
    results_json = pdf_results_dir / "results.json"

    if results_json.exists():
        print(f"   ↪ Skipping OCR for {pdf_stem} (results.json already exists).")
        return

    cmd = [
        "surya_ocr",                
        "--task_name", "ocr_without_boxes",
        str(pdf_path),
        "--output_dir",
        str(course_out_dir),
    ]

    print(f"\n🔎 Running Surya on: {pdf_path}")
    print("CMD:", " ".join(cmd))

    completed = subprocess.run(cmd, capture_output=True, text=True)

    if completed.returncode != 0:
        print("❌ Surya OCR failed.")
        print("STDOUT:", completed.stdout)
        print("STDERR:", completed.stderr)
        raise RuntimeError(f"surya_ocr failed on {pdf_path}")

    if results_json.exists():
        print(f"✅ Surya finished. JSON -> {results_json}")
    else:
        raise FileNotFoundError(
            f"Expected {results_json} after Surya run, but it was not created."
        )


def extract_text_from_results(
    semester_name: str,
    course_name: str,
):

    course_processed_dir = SURYA_DIR / semester_name / course_name
    if not course_processed_dir.exists():
        print(f"  ⚠ No processed_notes folder for {course_name}, skipping text extraction.")
        return

    for pdf_dir in sorted(course_processed_dir.iterdir()):
        if not pdf_dir.is_dir():
            continue

        pdf_stem = pdf_dir.name
        results_json = pdf_dir / "results.json"
        if not results_json.exists():
            print(f"   ⚠ No results.json in {pdf_dir}, skipping.")
            continue

        # target text file
        text_out_dir = OCR_TEXT_DIR / semester_name / course_name
        text_out_dir.mkdir(parents=True, exist_ok=True)
        out_txt = text_out_dir / f"{pdf_stem}.txt"

        if out_txt.exists():
            print(f"   ↪ Text for {pdf_stem} already exists, skipping.")
            continue

        print(f"📝 Extracting text from {results_json} -> {out_txt}")

        with results_json.open("r", encoding="utf-8") as f:
            results = json.load(f)

        if not results:
            print(f"   ⚠ {results_json} is empty ({{}}). Skipping.")
            continue

        all_lines = []
        # Surya JSON: { "doc_name": [pages...] }
        for _, pages in results.items():
            for page in pages:
                for line in page.get("text_lines", []):
                    text = line.get("text", "").strip()
                    if text:
                        all_lines.append(text)

        if not all_lines:
            print(f"   ⚠ No text lines found in {results_json}, skipping.")
            continue

        out_txt.write_text("\n".join(all_lines), encoding="utf-8")
        print(f"   💾 Saved text -> {out_txt}")


# ----- Main pipeline -----


def main():
    if not RAW_DIR.exists():
        raise FileNotFoundError(f"{RAW_DIR} does not exist. Check your data path.")

    for semester_folder in RAW_DIR.iterdir():
        if not semester_folder.is_dir():
            continue

        semester_name = semester_folder.name
        print(f"\n==============================")
        print(f"🎓 Processing {semester_name}")
        print(f"==============================")

        # If you only want Semester 3 for now, uncomment:
        # if semester_name != "Semester 3":
        #     continue

        for course_folder in semester_folder.iterdir():
            if not course_folder.is_dir():
                continue

            course_name = course_folder.name
            print(f"\n📚 Course: {course_name}")

            pdf_files = sorted(course_folder.glob("*.pdf"))
            if not pdf_files:
                print(f"  ⚠ No PDFs found in {course_folder}, skipping course.")
                continue

            # 1) Run OCR for each PDF that doesn't have results.json yet
            for pdf_path in pdf_files:
                run_surya_on_file(pdf_path, semester_name, course_name)

            # 2) Extract text from all processed JSONs for this course
            extract_text_from_results(semester_name, course_name)

    print("\n🎉 OCR pipeline complete.")
    print(f"JSON output is under: {SURYA_DIR}")
    print(f"Plain text is under: {OCR_TEXT_DIR}")


if __name__ == "__main__":
    main()
