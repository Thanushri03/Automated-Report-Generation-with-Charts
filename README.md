# Automated-Report-Generation-with-Charts


---

This Streamlit application allows users to upload multiple internal documents (PDF, DOCX, or CSV) and automatically generate a structured report using Google’s Gemini model. The generated report includes an executive summary, insights, data visualizations, and recommendations, which can be downloaded as a Word (`.docx`) file.

---

## Features

* Upload multiple file types: **PDF**, **DOCX**, **CSV**
* Automatic text extraction and cleaning
* Smart chunking with configurable size and overlap
* Integration with **Google Gemini (genai)** for structured report generation
* Automatic **chart creation** using Matplotlib
* Optional inclusion of **external context** or market notes
* Downloadable **DOCX** report output
* Streamlit-based **interactive web UI**

---

## Requirements

* Python 3.9 or later
* Google Gemini API key
* Internet connection (for Gemini API calls)

---

## Installation

1. **Clone this repository:**

   ```bash
   git clone https://github.com/yourusername/report-generator.git
   cd report-generator
   ```

2. **Create a virtual environment (recommended):**

   ```bash
   python -m venv venv
   source venv/bin/activate   # On macOS/Linux
   venv\Scripts\activate      # On Windows
   ```

3. **Install dependencies:**

   ```bash
   pip install -r requirements.txt
   ```

---

## Dependencies

Add these to `requirements.txt`:

```
streamlit
pandas
matplotlib
python-docx
PyPDF2
google-genai
```

---

## Configuration

You must provide your **Google Gemini API key** before using the app.

There are two ways to do this:

1. Set it in the sidebar when the app is running
   or
2. Export it as an environment variable:

   ```bash
   export GEMINI_API_KEY="your_api_key_here"
   ```

You can also customize:

* Model name (default: `gemini-2.5-flash`)
* Chunk size and overlap
* External notes or additional text context

---

## Usage

1. Run the app:

   ```bash
   streamlit run app.py
   ```

2. Open your browser at the provided local URL (usually `http://localhost:8501`).

3. In the sidebar:

   * Enter your Gemini API key.
   * Adjust settings as needed.

4. Upload your files (PDF, DOCX, CSV).

5. Click **"Generate Report"**.

6. View chart previews (optional) and download the generated **DOCX** report.

---

## Output Example

The generated report includes:

* Title and report date
* Executive Summary
* Key Insights (bulleted)
* Data Visualizations (charts or tables)
* Recommendations (numbered list)
* References or citations

---

## File Structure

```
.
├── app.py                # Main Streamlit application
├── requirements.txt       # Python dependencies
├── README.md              # Documentation (this file)
```

---

## Notes

* The application relies on the **Google Gemini API** to generate structured reports. Ensure your API key has the correct permissions and usage quota.
* For large files, adjust chunk size and overlap for best performance.
* The report generator supports automatic chart creation for numerical data (bar, line, or pie).


---


