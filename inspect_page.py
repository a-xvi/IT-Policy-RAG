import pdfplumber 
 
with pdfplumber.open("data/dummy_it_policy.pdf") as pdf: 
    for i, page in enumerate(pdf.pages): 
        text = page.extract_text() 
        if text: 
            lines = text.split("\n") 
            weird_lines = [l for l in lines if len(l.strip()) <= 3 and l.strip()] 
            if len(weird_lines) > 5: 
                print(f"Page {i+1}: {len(weird_lines)} short/junk-looking lines") 
