from fpdf import FPDF
import pandas as pd

def export_to_csv(results, filename="sentiment_results.csv"):
    df = pd.DataFrame(results)
    df.to_csv(filename, index=False)

def export_to_json(results, filename="sentiment_results.json"):
    df = pd.DataFrame(results)
    df.to_json(filename, orient="records", lines=True)

def export_to_pdf(results, filename="sentiment_results.pdf"):
    df = pd.DataFrame(results)

    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=10)

    line_height = pdf.font_size * 2.5
    page_width = pdf.w - 2 * pdf.l_margin
    col_width = page_width / 3

    # Header
    headers = ["Text", "Sentiment", "Confidence"]
    for header in headers:
        pdf.cell(col_width, line_height, header, border=1)
    pdf.ln(line_height)

    # Rows
    for _, row in df.iterrows():
        text = str(row["text"])[:40]
        pdf.cell(col_width, line_height, text, border=1)
        pdf.cell(col_width, line_height, row["sentiment"], border=1)
        pdf.cell(col_width, line_height, f"{row['confidence']:.2f}", border=1)
        pdf.ln(line_height)

    pdf.output(filename)
