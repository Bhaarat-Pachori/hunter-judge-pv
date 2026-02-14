import os
import re
import csv
import spacy

def preprocess_cadec_v2(data_path, output_path, n_sentences=10):
    """
    Preprocesses the CADEC v2 dataset to generate a CSV for sentence-level
    ADE detection, including metadata.

    Args:
        data_path (str): The parent directory containing 'text' and 'original' subfolders.
        output_path (str): The directory to save the generated CSV file.
        n_sentences (int): The maximum number of sentences to extract.
    """

    text_folder = os.path.join(data_path, "text")
    ann_folder = os.path.join(data_path, "original")

    if not os.path.exists(text_folder) or not os.path.exists(ann_folder):
        raise ValueError(f"'{text_folder}' or '{ann_folder}' does not exist. "
                         f"Ensure the data path is correct and contains these subfolders.")

    nlp = spacy.load("en_core_web_sm")

    os.makedirs(output_path, exist_ok=True)
    csv_file = os.path.join(output_path, "cadec_v2.csv")

    extracted_sentences = 0
    with open(csv_file, 'w', newline='', encoding='utf-8') as csvfile:
        fieldnames = ['id', 'text', 'label', 'label_text', 'meta_doc_drugs',
                      'meta_prev_context', 'span_offsets', 'source_file']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        for filename in os.listdir(text_folder):
            if not filename.endswith(".txt"):
                continue

            base_filename = filename[:-4]
            text_file = os.path.join(text_folder, filename)
            ann_file = os.path.join(ann_folder, base_filename + ".ann")

            if not os.path.exists(ann_file):
                print(f"Warning: Annotation file not found for {filename}")
                continue

            with open(text_file, 'r', encoding='utf-8') as f:
                text = f.read()

            # 1. Extract Document-Level Drugs
            doc_drugs = extract_document_drugs(ann_file, base_filename)

            # 2. Extract ADEs
            ades = extract_ades(ann_file)

            # 3. Sentence Segmentation
            doc = nlp(text)
            sentences = list(doc.sents)

            prev_context = ""
            for i, sentence in enumerate(sentences):
                # if extracted_sentences >= n_sentences:
                #     break

                sentence_text = sentence.text
                span_start = sentence.start_char
                span_end = sentence.end_char

                # Check for overlap with ADEs
                label = 0
                label_text = "Not-Related"
                for start, end in ades:
                    if span_start <= end and span_end >= start:
                        label = 1
                        label_text = "Related"
                        break

                # Format document drugs
                formatted_doc_drugs = "; ".join(sorted(list(set(doc_drugs))))

                writer.writerow({
                    'id': f"{base_filename}_{i}",
                    'text': sentence_text,
                    'label': label,
                    'label_text': label_text,
                    'meta_doc_drugs': formatted_doc_drugs,
                    'meta_prev_context': prev_context,
                    'span_offsets': f"{span_start}, {span_end}",
                    'source_file': filename
                })
                extracted_sentences += 1
                prev_context = sentence_text

    print(f"Successfully created {csv_file} with {extracted_sentences} sentences.")


def extract_document_drugs(ann_file, base_filename):
    drugs = []
    try:
        with open(ann_file, 'r', encoding='utf-8') as f:
            for line in f:
                if line.startswith("T") and "Drug" in line:
                    parts = line.split('\t')
                    drugs.append(parts[2].strip())
    except:
        drugs = [base_filename] #If .ann file not found, use the parent folder name (e.g., "Lipitor").
    
    if not drugs:
        drugs = [base_filename]  # Use filename if no drugs found in .ann
    return drugs


def extract_ades(ann_file):
    ades = []
    with open(ann_file, 'r', encoding='utf-8') as f:
        for line in f:
            if line.startswith("T") and ("ADR" in line or "ADE" in line.upper()):
                parts = line.split('\t')
                if len(parts) < 2:
                    continue
                try:
                    entity_info = parts[1].split()
                    if len(entity_info) < 3:  # Ensure entity_info has enough elements
                        continue
                    start = int(entity_info[1])
                    end = int(entity_info[2])
                    ades.append((start, end))
                except ValueError:
                    print(f"Warning: Invalid entity info in line: {line.strip()}")
    return ades


if __name__ == "__main__":
    data_path = "hunter-judge-pv/data/cadec_v2"
    output_path = "hunter-judge-pv/data/cadec_v2"
    preprocess_cadec_v2(data_path, output_path)
