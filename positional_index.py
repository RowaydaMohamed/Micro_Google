from pyspark import SparkContext
import os
import re
import sys

sc = SparkContext(appName="PositionalInvertedIndex")

input_folder = sys.argv[1]

documents = []

file_names = sorted(os.listdir(input_folder), key=lambda x: int(x.split(".")[0])) 

for file_name in file_names:
    file_path = os.path.join(input_folder, file_name)
    with open(file_path, "r", encoding="utf-8") as f:
        text = f.read()
        documents.append((file_name, text)) 

files_rdd = sc.parallelize(documents)

# Exception dictionary for irregular forms
IRREGULARS = {
    "worser": "worse",
}

# Proper nouns to keep intact
PROPER_NOUNS = {"brutus"}  

# Stemmer function
def stem_word(word):
    if word in PROPER_NOUNS:
        return word
    if word in IRREGULARS:
        return IRREGULARS[word]
    # Regular plural stemming
    if word.endswith('s') and not word.endswith('ss'): 
        return word[:-1]
    return word

# Tokenization with positions and stemming
def tokenize(doc):
    doc_id, text = doc 
    words = re.findall(r"\w+", text.lower()) #[brutus, cleopatra, antony, antony, brutus, cleopatra]
    tokens = []
    for i, word in enumerate(words): 
        stemmed = stem_word(word)
        tokens.append(((stemmed, doc_id), i + 1))
    return tokens

term_doc_pos = files_rdd.flatMap(tokenize)

# Group positions per (term, doc)
grouped_positions = term_doc_pos.groupByKey().mapValues(list)

term_doc_positions = grouped_positions.map(
    lambda x: (x[0][0], (x[0][1], sorted(x[1])))
)

term_index = term_doc_positions.groupByKey().mapValues(list)


result = term_index.collect()


with open("positional_index.txt", "w") as f:
    for term, postings in result:
        f.write(f"< {term}\n")
        for doc_id, positions in sorted(postings, key=lambda x: x[0]):  
            pos_str = ", ".join(map(str, positions)) #converts list to string 
            f.write(f"  {doc_id}: {pos_str} ;\n")
        f.write(">\n\n")

sc.stop()
