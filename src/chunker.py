import re
import json
import os


def clean_standard_id(raw_id):
    """Normalizes the ID to exactly match the hackathon's eval script."""
    clean = re.sub(r'\s+', ' ', raw_id).strip()
    clean = re.sub(r'\s*:\s*', ': ', clean)
    # Fix the uppercase/lowercase "Part" issue so the JSON matches exactly
    clean = clean.upper().replace('PART', 'Part')
    return clean


def clean_content(text):
    """Removes useless tokens (images, page headers) to make the LLM faster."""
    # 1. Remove all markdown image links
    text = re.sub(r'!\[image\]\(.*?\)', '', text)

    # 2. Remove the repetitive handbook title that appears on every page
    text = re.sub(r'SP 21 : 2005', '', text)
    text = re.sub(r'Title Page', '', text)

    # 3. Clean up excessive blank lines
    text = re.sub(r'\n{3,}', '\n\n', text)

    return text.strip()


def chunk_bis_markdown(file_path, output_json):
    print(f"Reading {file_path}...")
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()

    # The ultimate regex pattern to find "IS 383 : 1970" or "IS 2185 (Part 1) : 1979"
    pattern = r"(IS\s+\d+(?:\s*\(\s*Part\s*\d+\s*\))?\s*:\s*\d{4})"

    print("Slicing document by Standard ID...")
    # Add flags=re.IGNORECASE to ensure we catch "(PART 2)" from the OCR
    parts = re.split(pattern, content, flags=re.IGNORECASE)

    chunks = []

    # parts[0] is the Index/Table of Contents. We skip it.
    # parts[1] is the first ID, parts[2] is its text body, etc.
    for i in range(1, len(parts), 2):
        raw_id = parts[i]
        text_body = clean_content(parts[i+1])

        # Only save chunks that have actual specifications in them
        if len(text_body) > 50:
            clean_id = clean_standard_id(raw_id)
            chunks.append({
                "standard_id": clean_id,
                # We prepend the ID back into the content so the Vector DB
                # and LLM know exactly what standard they are reading
                "content": f"{clean_id} {text_body}"
            })

    print(f"Successfully created {len(chunks)} distinct standard flashcards!")

    with open(output_json, 'w', encoding='utf-8') as f:
        json.dump(chunks, f, indent=4)

    print(f"Saved to {output_json}")


if __name__ == "__main__":
    # Ensure this targets the files inside the 'data' folder
    input_file = os.path.join("data", "dataset.md")
    output_file = os.path.join("data", "bis_chunks.json")

    chunk_bis_markdown(input_file, output_file)
