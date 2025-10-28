import fitz  # PyMuPDF
import os
import pandas as pd

pdf_path = "GrandVitara 1.pdf"
doc = fitz.open(pdf_path)

output_dir = "output_dataset"
os.makedirs(output_dir, exist_ok=True)

data = []

for page_num in range(len(doc)):
    page = doc[page_num]
    text = page.get_text()

    data.append({
        "page": page_num + 1,
        "text": text.strip()
    })

df = pd.DataFrame(data)
df.to_csv(os.path.join(output_dir, "grand_vitara_dataset.csv"), index=False)

