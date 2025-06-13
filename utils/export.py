import csv
import json
import pandas as pd
from fpdf import FPDF

def export_to_csv(results, filename="sentiment_results.csv"):
    keys = results[0].keys() if results else []
    with open(filename, 'w', newline='', encoding='utf-8') as output_file:
        dict_writer = csv.DictWriter(output_file, fieldnames=keys)
        dict_writer.writeheader()
        dict_writer.writerows(results)

def export_to_json(results, filename="sentiment_results.json"):
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=4)

def export_to_pdf(results, filename="sentiment_results.pdf"):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    
    line_height = pdf.font_size * 2.5
    col_width = pdf.epw / 3  # equal width columns
    
    # Header
    pdf.cell(col_width, line_height, "Text", border=1)
    pdf.cell(col_width, line_height, "Sentiment", border=1)
    pdf.cell(col_width, line_height, "Confidence", border=1)
    pdf.ln(line_height)
    
    # Data rows
    for item in results:
        pdf.multi_cell(col_width, line_height, item['text'], border=1, ln=3, max_line_height=pdf.font_size)
        pdf.set_xy(pdf.get_x() + col_width, pdf.get_y() - line_height)
        pdf.cell(col_width, line_height, item['sentiment'], border=1)
        pdf.cell(col_width, line_height, f"{item['confidence']:.2f}", border=1)
        pdf.ln(line_height)
    
    pdf.output(filename)
