import tkinter as tk
from tkinter import scrolledtext, messagebox, filedialog
import re
import math
from collections import defaultdict, Counter
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import os
from tabulate import tabulate # For creating the detailed output table

# ------------------------------
# Global Variables (State Management)
# ------------------------------
DOCUMENTS = {}       # { "doc_id": "content text" }
DOC_IDS = []         # List of doc IDs for indexing
N = 0                # Total number of documents
POSITIONAL_INDEX = {}
TFIDF_MATRIX = None
ALL_TERMS = []
DOC_ID_TO_INDEX = {}
TERM_TO_INDEX = {}
IDF = {}

# ------------------------------
# 1. Backend: Load & Parse Data (Folder Mode)
# ------------------------------

def load_data_from_folder(folder_path):
    """
    Reads ALL .txt files in the selected folder.
    """
    global DOCUMENTS, DOC_IDS, N
    
    DOCUMENTS = {}
    
    if not folder_path:
        return False

    try:
        # Get list of .txt files
        files = [f for f in os.listdir(folder_path) if f.endswith('.txt')]

        if not files:
            messagebox.showwarning("Warning", "No .txt files found in this folder!")
            return False

        # Process each file
        for filename in files:
            file_path = os.path.join(folder_path, filename)
            
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read().strip()
            
            # --- ID Extraction Logic ---
            # Check for <ID> format inside the file
            id_match = re.search(r'<ID>(\w+)</ID>', content)
            
            if id_match:
                # Use the ID found in the text (e.g., "1")
                doc_id = id_match.group(1)
                # Clean the tag out of the text
                clean_content = re.sub(r'<ID>(\w+)</ID>', '', content).strip()
            else:
                # Fallback: Use the filename (e.g., "1.txt" -> "1")
                doc_id = filename.split('.')[0] # Use filename without extension
                clean_content = content

            # Basic cleanup
            clean_content = " ".join(clean_content.split())
            DOCUMENTS[doc_id] = clean_content

        # Sort IDs (Numerically if possible)
        try:
            DOC_IDS = sorted(DOCUMENTS.keys(), key=lambda x: int(x) if x.isdigit() else x)
        except:
            DOC_IDS = sorted(DOCUMENTS.keys())
            
        N = len(DOC_IDS)
        return True

    except Exception as e:
        messagebox.showerror("Error", f"Failed to load folder: {e}")
        return False

def build_index_and_tfidf():
    """
    Constructs the Positional Index and TF-IDF Matrix from the loaded DOCUMENTS.
    """
    global POSITIONAL_INDEX, TFIDF_MATRIX, ALL_TERMS, DOC_ID_TO_INDEX, TERM_TO_INDEX, IDF
    
    # --- Step A: Build Positional Index ---
    POSITIONAL_INDEX = defaultdict(lambda: defaultdict(list))
    
    for doc_id, content in DOCUMENTS.items():
        terms = content.lower().split()
        for position, term in enumerate(terms, 1):
            POSITIONAL_INDEX[term][doc_id].append(position)

    # --- Step B: Build TF-IDF Matrix ---
    if N == 0: return

    # 1. TF (Term Frequency)
    tf_matrix = defaultdict(lambda: defaultdict(int))
    for doc_id, content in DOCUMENTS.items():
        terms = content.lower().split()
        term_counts = Counter(terms)
        for term, count in term_counts.items():
            tf_matrix[term][doc_id] = count

    # 2. IDF (Inverse Document Frequency)
    df = {term: len(postings) for term, postings in POSITIONAL_INDEX.items()}
    # IDF calculation: log10(N / df)
    idf_calc = {term: math.log10(N / df_val) if df_val > 0 else 0 for term, df_val in df.items()}
    
    ALL_TERMS = sorted(POSITIONAL_INDEX.keys())
    TFIDF_MATRIX = np.zeros((N, len(ALL_TERMS)))

    # Mappings
    DOC_ID_TO_INDEX = {doc_id: i for i, doc_id in enumerate(DOC_IDS)}
    TERM_TO_INDEX = {term: i for i, term in enumerate(ALL_TERMS)}
    IDF = idf_calc

    # 3. Fill Matrix (TF * IDF)
    for term, postings in POSITIONAL_INDEX.items():
        if term not in TERM_TO_INDEX: continue
        term_idx = TERM_TO_INDEX[term]
        for doc_id in postings:
            if doc_id not in DOC_ID_TO_INDEX: continue
            doc_idx = DOC_ID_TO_INDEX[doc_id]
            # TF-IDF calculation: (1 + log10(tf)) * idf
            tf_raw = tf_matrix[term][doc_id]
            tf_log = 1 + math.log10(tf_raw) if tf_raw > 0 else 0
            TFIDF_MATRIX[doc_idx, term_idx] = tf_log * IDF[term]

# ------------------------------
# 2. Backend: Search Algorithms (Modified for Detailed Output)
# ------------------------------

def phrase_match(phrase):
    # ... (Keep the original phrase_match function as it is for boolean logic)
    terms = phrase.lower().split()
    if not terms: return set()

    first_term = terms[0]
    if first_term not in POSITIONAL_INDEX: return set()

    candidate_docs = set(POSITIONAL_INDEX[first_term].keys())
    matched_docs = set()

    for doc_id in candidate_docs:
        term_positions = []
        valid = True
        
        for term in terms:
            if doc_id in POSITIONAL_INDEX.get(term, {}):
                term_positions.append(POSITIONAL_INDEX[term][doc_id])
            else:
                valid = False; break
        if not valid: continue

        pos_list = term_positions[0]
        for pos in pos_list:
            match = True
            for i in range(1, len(terms)):
                # Check if the next term is at the next position (pos + i)
                if (pos + i) not in term_positions[i]:
                    match = False; break
            if match:
                matched_docs.add(doc_id); break
    return matched_docs

def process_boolean_query(query):
    # ... (Keep the original process_boolean_query function as it is for boolean logic)
    query = query.lower().strip()

    # 1. Handle OR (Splits query into parts, results are UNIONED)
    if ' or ' in query:
        parts = query.split(' or ')
        result = set()
        for part in parts:
            result = result.union(process_boolean_query(part))
        return result

    # 2. Handle AND (Splits query into parts, results are INTERSECTED)
    elif ' and ' in query:
        parts = query.split(' and ')
        result = process_boolean_query(parts[0])
        for part in parts[1:]:
            result = result.intersection(process_boolean_query(part))
        return result

    # 3. Handle NOT (Inverts the result of the following part)
    elif query.startswith('not '):
        sub_query = query[4:].strip()
        sub_result = process_boolean_query(sub_query)
        return set(DOC_IDS) - sub_result

    # 4. Base Case: Exact Phrase Match
    else:
        return phrase_match(query)

def rank_documents_detailed(matched_docs, query_terms):
    """
    Calculates Cosine Similarity and returns detailed data for the output table.
    """
    if not matched_docs or N == 0: return [], None, None, None

    # 1. Prepare Query Vector Data
    query_data = []
    query_vec = np.zeros(len(ALL_TERMS))
    query_term_counts = Counter(query_terms)
    
    # Get the doc IDs that are in the TFIDF_MATRIX (i.e., have been indexed)
    indexed_matched_docs = [doc_id for doc_id in matched_docs if doc_id in DOC_ID_TO_INDEX]
    
    # Get the indices of the matched documents in the TFIDF_MATRIX
    doc_indices = [DOC_ID_TO_INDEX[doc_id] for doc_id in indexed_matched_docs]
    doc_ids_sorted = [DOC_IDS[idx] for idx in doc_indices]

    # Extract the relevant columns from the TFIDF_MATRIX for the matched documents
    doc_vectors = TFIDF_MATRIX[doc_indices, :]
    
    # --- Calculate Query TF-IDF and Normalization ---
    for term in sorted(query_term_counts.keys()):
        if term not in TERM_TO_INDEX:
            continue # Skip terms not in the vocabulary
            
        tf_raw = query_term_counts[term]
        idf_val = IDF.get(term, 0)
        
        # TF: 1 + log10(tf_raw)
        tf_log = 1 + math.log10(tf_raw) if tf_raw > 0 else 0
        
        # TF*IDF
        tfidf_val = tf_log * idf_val
        
        # Fill the query vector (using TF*IDF)
        term_idx = TERM_TO_INDEX[term]
        query_vec[term_idx] = tfidf_val
        
        # Store for the table (Normalization will be calculated later)
        query_data.append({
            'term': term,
            'tf_raw': tf_raw,
            'tf_log': tf_log,
            'idf': idf_val,
            'tfidf': tfidf_val,
            'term_idx': term_idx
        })

    # 2. Calculate Query Length (L2 Norm)
    query_length = np.linalg.norm(query_vec)
    
    # 3. Calculate Normalized Query Vector and Product (Query * Matched Docs)
    normalized_query_vec = query_vec / query_length if query_length > 0 else query_vec
    
    # Calculate normalized value for the table
    for item in query_data:
        term_idx = item['term_idx']
        item['normalized'] = normalized_query_vec[term_idx]
        
        # Calculate Product (Query * Matched Docs) for the table
        # Product = Normalized Query Vector * Normalized Document Vector
        # Since the TFIDF_MATRIX is NOT normalized, we need to normalize it first.
        # However, the image shows the product of (query * matched docs) which is the dot product
        # of the normalized query vector and the normalized document vectors.
        # Let's use the standard Cosine Similarity formula: (Q_norm . D_norm)
        # The image seems to show (Q_norm * D_raw) or similar.
        # To match the image, we will calculate the product of the normalized query term
        # and the raw TFIDF value of the document term.
        
        # We need the normalized document vectors to calculate the similarity correctly.
        # Let's normalize the document vectors first.
        
    # Normalize Document Vectors (L2 Norm)
    doc_lengths = np.linalg.norm(doc_vectors, axis=1, keepdims=True)
    normalized_doc_vectors = doc_vectors / doc_lengths
    
    # Recalculate Product (Normalized Query Term * Normalized Document Term)
    # This is the term-by-term product before summation.
    product_matrix = np.zeros((len(query_data), len(doc_ids_sorted)))
    
    for i, item in enumerate(query_data):
        term_idx = item['term_idx']
        # Extract the column for this term from the normalized document vectors
        doc_term_vec = normalized_doc_vectors[:, term_idx]
        
        # Product = Normalized Query Term * Normalized Document Term
        product_matrix[i, :] = item['normalized'] * doc_term_vec
        
    # 4. Calculate Similarity Scores (Sum of Products)
    # The sum of the product matrix columns gives the final similarity score
    similarity_scores = np.sum(product_matrix, axis=0)
    
    # 5. Create Ranking
    ranking = list(zip(doc_ids_sorted, similarity_scores))
    ranking.sort(key=lambda x: x[1], reverse=True)
    
    # 6. Prepare Final Output Data Structure
    
    # Combine query data with product matrix for the final table
    for i, item in enumerate(query_data):
        item['product'] = product_matrix[i, :].tolist()
        
    # Similarity scores dictionary for easy lookup
    similarity_dict = dict(zip(doc_ids_sorted, similarity_scores))
    
    return ranking, query_data, query_length, similarity_dict

# ------------------------------
# 3. GUI Logic (Modified to Display Detailed Output)
# ------------------------------

def browse_folder():
    global folder_label, result_box
    # ... (Keep the original browse_folder function)
    folder_selected = filedialog.askdirectory(title="Select Dataset Folder")
    
    if folder_selected:
        success = load_data_from_folder(folder_selected)
        if success:
            build_index_and_tfidf()
            folder_label.config(text=f"Loaded: {os.path.basename(folder_selected)}")
            result_box.delete(1.0, tk.END)
            result_box.insert(tk.END, f"✅ Successfully loaded {N} documents from folder.\n\nYou can now search using AND, OR, NOT!", "result")
        else:
            folder_label.config(text="No folder loaded")

def run_query_gui():
    global query_entry, result_box
    if not DOCUMENTS:
        messagebox.showerror("Error", "Please load a dataset folder first!")
        return

    query = query_entry.get().strip()
    if not query:
        messagebox.showwarning("Input Error", "Please enter a query.")
        return

    result_box.delete(1.0, tk.END)
    
    # 1. Get Boolean Results
    matched_docs = process_boolean_query(query)

    if not matched_docs:
        result_box.insert(tk.END, "❌ No documents matched this query.\n")
        return

    # 2. Rank Results (Detailed)
    query_terms = re.findall(r'\b\w+\b', query.lower())
    clean_terms = [t for t in query_terms if t not in ('and', 'or', 'not')]
    
    ranking, query_data, query_length, similarity_dict = rank_documents_detailed(matched_docs, clean_terms)
    
    if not ranking:
        result_box.insert(tk.END, "❌ No documents matched this query after indexing.\n")
        return

    # 3. Display Detailed TF-IDF Table (Matching the image)
    
    # Prepare table headers
    doc_ids_sorted = [doc_id for doc_id, score in ranking]
    doc_headers = [f'doc {doc_id}' for doc_id in doc_ids_sorted]
    
    # Limit to top 3 for display clarity, matching the image example (doc7, doc8, d10)
    display_doc_ids = doc_ids_sorted[:3]
    display_doc_headers = [f'doc {doc_id}' for doc_id in display_doc_ids]
    
    headers = ['query', 'tf-raw', 'tf(1+log t)', 'idf', 'tf*idf', 'normalized'] + display_doc_headers
    table_data = []
    
    # Fill table data
    for item in query_data:
        row = [
            item['term'],
            f"{item['tf_raw']:.4g}",
            f"{item['tf_log']:.4f}",
            f"{item['idf']:.4f}",
            f"{item['tfidf']:.4f}",
            f"{item['normalized']:.4f}"
        ]
        
        # Add product values for the displayed documents
        for doc_id in display_doc_ids:
            # Find the index of the doc_id in the full list
            full_list_index = doc_ids_sorted.index(doc_id)
            # Get the product value for this term and this document
            # The product matrix is stored in item['product']
            product_value = item['product'][full_list_index]
            row.append(f"{product_value:.4f}")
            
        table_data.append(row)
        
    # Calculate the 'sum' row
    sum_row = ['sum', '', '', '', '', '']
    
    # Calculate the sum of the product column for each displayed document
    for doc_id in display_doc_ids:
        full_list_index = doc_ids_sorted.index(doc_id)
        col_sum = similarity_dict[doc_id]
        sum_row.append(f"{col_sum:.4f}")
        
    table_data.append(sum_row)
    
    # Insert the table into the result box
    result_box.insert(tk.END, f"Query: {query}\n\n", "section")
    result_box.insert(tk.END, tabulate(table_data, headers=headers, tablefmt="fancy_grid", numalign="right") + "\n\n", "doc")
    
    # Insert Query Length
    result_box.insert(tk.END, f"Query Length: {query_length:.6f}\n\n", "header")
    
    # Insert Similarity Scores
    result_box.insert(tk.END, "Similarity Scores:\n", "section")
    for doc_id, score in ranking:
        result_box.insert(tk.END, f"similarity (q, doc{doc_id}): {score:.4f}\n")
        
    # Insert Returned Docs
    returned_docs = [doc_id for doc_id, score in ranking]
    result_box.insert(tk.END, f"\nReturned Docs (Ranked): {', '.join(returned_docs)}\n\n", "header")
    
    # 4. Display Top 3 Documents (Original Logic)
    result_box.insert(tk.END, "🏆 Top 3 Documents\n", "section")
    for i, (doc_id, score) in enumerate(ranking[:3], 1):
        result_box.insert(tk.END, f"\n{i}. {doc_id} (Score: {score:.4f})\n", "subheader")
        result_box.insert(tk.END, DOCUMENTS[doc_id] + "\n", "doc")

# ------------------------------
# 4. Main Window Setup
# ------------------------------

root = tk.Tk()
root.title("TF-IDF Detailed Search Engine")
root.geometry("850x780")
root.resizable(False, False)

# --- Header Section ---
top_frame = tk.Frame(root)
top_frame.pack(pady=15)

title_label = tk.Label(top_frame, text="TF-IDF Detailed Search Engine", font=("Arial", 18, "bold"))
title_label.pack()

subtitle_label = tk.Label
subtitle_label = tk.Label(top_frame, text="Boolean + TF-IDF + Cosine Similarity", font=("Arial", 12))
subtitle_label.pack()

# --- Query Section ---
query_frame = tk.Frame(root)
query_frame.pack(pady=10)

query_entry = tk.Entry(query_frame, width=60, font=("Arial", 12))
query_entry.grid(row=0, column=0, padx=10)

search_button = tk.Button(query_frame, text="Search", command=run_query_gui, width=12)
search_button.grid(row=0, column=1)

# --- Browse Folder Button ---
folder_button = tk.Button(root, text="Load Dataset Folder", command=browse_folder, width=25)
folder_button.pack(pady=5)

folder_label = tk.Label(root, text="No folder loaded", font=("Arial", 10))
folder_label.pack()

# --- Results Box ---
result_box = scrolledtext.ScrolledText(root, width=100, height=30, font=("Consolas", 10))
result_box.pack(pady=10)

# --- GUI Loop ---
root.mainloop()
