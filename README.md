# AI-for-ESG

## Phase 0 – Data Collection 📁

This project begins with the manual construction of a high-quality document corpus based on **publicly available corporate information** from companies listed in the **Swiss Market Index (SMI)**. To ensure data consistency and feasibility, we focus on the **top 10 SMI companies by market capitalization**, over the **2021–2023** period. These three years provide a sufficiently recent and rich dataset, with wide availability of sustainability and governance disclosures.

### 🔍 Selected Sources of Information

The dataset includes the following document types, which are commonly used in ESG assessments and financial analysis:

- **Annual Reports**  
- **Half-Year Reports**  
- **Sustainability Reports**  
- **Integrated Reports** (when available)  
- **Corporate Governance Reports**  
- **Earnings Call Transcripts & Fireside Chats**

The goal is to rely on **complete, public, and comparable documents** as the foundation for all subsequent analyses.

### ⚠️ Why the Collection Was Done Manually

Although web scraping was initially considered, it quickly proved unreliable and inefficient due to the following issues:

- Each company has a unique website structure and navigation logic.
- Report titles and formats vary widely (e.g. *“Integrated Report”*, *“Creating Shared Value Report”*, *“Sustainability Highlights”*, etc.).
- Some websites load documents dynamically via JavaScript, making traditional scraping tools ineffective.
- Links to PDF files are often not directly accessible in the initial HTML.
- Document classification is inconsistent (e.g. a 2023 report might be listed under a 2024 tab).
- Multilingual content (e.g. Holcim: English titles, German content).
- Certain companies split reports across multiple categories (e.g. Swiss Re separates Business, Financial, and Sustainability reports).

✅ **Conclusion**: Manual collection ensured better reliability, completeness, and clarity.

### 🔊 Earnings Calls and Transcripts

To complement the reports, I manually downloaded **earnings call transcripts and conference discussions** (fireside chats, Q&A sessions) for each selected company. These documents offer valuable insight due to their more conversational and analyst-driven format.

- Sources: **Seeking Alpha** (using a 28-day free account)
- Presentations were **excluded**: they tend to be scripted and marketing-oriented
- Focus was placed on interactive formats (Q&A, analyst calls)

Due to **legal and technical restrictions**, scraping from Seeking Alpha was not feasible (explicitly forbidden in the platform’s Terms of Use, and actively blocked).

### 🗂️ File Organization & Metadata

All documents were stored in a structured folder on **Google Drive**.

To manage and track the collection process, I created two complementary metadata tables:

- **Sheet 1 – Presence Matrix**: overview of document availability by `Company × Year`, using Boolean indicators.
- **Sheet 2 – Document Register**: detailed information per file, including file name, type, and path.

The two sheets are linked by a unique `(Company, Year)` key.

### 📈 Coverage Summary

The table below summarizes the coverage status for the top 10 SMI companies (2021–2023). Most reports and transcripts were successfully collected. A few gaps remain, particularly for some quarterly earnings calls (e.g. Roche Q3 2021, Lonza Q3 2023).

| Company   | Reports Collected | Earnings Calls Collected |
|-----------|--------------------|---------------------------|
| Nestlé    | ✅                 | ✅                        |
| Novartis  | ✅                 | ✅                        |
| Roche     | ✅                 | ⚠️ Missing Q3 2021, Q1 2022 |
| Richemont | ✅                 | ⚠️ Missing Q1/Q3 all years |
| ABB       | ✅                 | ✅                        |
| Zurich    | ✅                 | ⚠️ Missing Q3 2021        |
| UBS       | ✅                 | ✅                        |
| Holcim    | ✅                 | ⚠️ Missing Q1/Q2/Q3 2021 and Q1 2022 |
| Swiss Re  | ✅                 | ✅                        |
| Lonza     | ✅                 | ⚠️ Missing Q1/Q3 for 2021, 2022, 2023 |

---

This manual collection phase lays the foundation for all subsequent analysis. The next step involves organizing these files into a structured dataframe with standardized metadata.

👉 Proceed to [Phase 1 – Dataset Construction](#phase-1--dataset-construction-🧱)


## Phase 1 – Dataset Construction 🧱

> 📁 **Note on data availability**  
Due to file size limitations and copyright considerations, the raw PDF documents (annual reports, earnings call transcripts, etc.) are **not included in this repository**.  
However, all files used in this project are publicly available online on the official investor relations websites of the selected companies.  
For convenience and reproducibility, **copies of all documents are stored in a private Google Drive folder** and are accessed programmatically (see paths in the code).

In this first notebook, I construct the core dataset used for analysis by combining two sources:

1. A manually downloaded collection of PDF documents (annual reports, sustainability reports, transcripts, etc.) stored on Google Drive.
2. An Excel file containing structured metadata for the top 10 SMI companies.

### 🗂️ File Parsing and Metadata Extraction

I programmatically traverse each company's folder in Drive and extract metadata for every `.pdf` file:
- **Company** (from folder structure)
- **Year** (from file name or path)
- **Document Type** (inferred from filename keywords)
- **Document Title**
- **File Path**

A preview of the resulting dataset:

| Company                    | Year | Document Type     | Document Title                     | Path                                                                                      |
|----------------------------|------|-------------------|------------------------------------|-------------------------------------------------------------------------------------------|
| Zurich Insurance Group AG  | 2023 | Annual Report     | Zurich_Annual_Report_2023.pdf      | /content/drive/MyDrive/Thèse Master/Data/Zurich Insurance Group AG/Zurich_Annual_Report_2023.pdf |
| Zurich Insurance Group AG  | 2023 | Half-Year Report  | Zurich_Half_Year_Report_2023.pdf   | /content/drive/MyDrive/Thèse Master/Data/Zurich Insurance Group AG/Zurich_Half_Year_Report_2023.pdf |
| Zurich Insurance Group AG  | 2022 | Annual Report     | Zurich_Annual_Report_2022.pdf      | /content/drive/MyDrive/Thèse Master/Data/Zurich Insurance Group AG/Zurich_Annual_Report_2022.pdf |

This structured DataFrame is used to match each document with financial and ESG metadata (tickers, industry classification) in the next step.

---

### 📄 Complementary Metadata Table

Each document is also described in a second table that includes external metadata, such as tickers, industry classification, and download information.

| Company     | Year | Ticker SMI | Ticker Seeking Alpha (US) | Ranking per Cap | SASB Industry     | Document Type     | Document Title       | Source        | Source URL                                                        | Format | Scrapable via Google | Saved Local |
|-------------|------|-------------|----------------------------|------------------|--------------------|--------------------|------------------------|----------------|-------------------------------------------------------------------|--------|-----------------------|-------------|
| Nestlé SA   | 2023 | NESN        | NSRGY                     | 1                | Food & Beverage    | Annual Report      | Annual Review         | Nestlé Website | https://www.nestle.com/investors/publications                    | PDF    | No                    | Yes         |
| Nestlé SA   | 2023 | NESN        | NSRGY                     | 1                | Food & Beverage    | Half-Year Report   | Half-Year Report      | Nestlé Website | https://www.nestle.com/investors/publications                    | PDF    | No                    | Yes         |

---

### 🔗 Metadata Merge

The extracted document data is then merged with the Excel file, matching each `(Company, Year)` pair.  
The Excel metadata includes:
- Company tickers (SMI and Seeking Alpha)
- SASB industry classification

Before merging, I performed several standardization steps to ensure consistency and avoid mismatches:
- Standardized company names (e.g. `Nestle` vs `Nestlé`)
- Converted year values to strings
- Normalized accents (e.g. `é` → `e`)

After merging, I removed perfect duplicates based on core metadata fields to ensure a clean dataset.

📄 **Output file**: `df_merged_clean.csv`  
This file is saved in Drive and serves as the input for the next phase of the project:  
**[Phase 2 – PDF Text Extraction and Preprocessing](#phase-2--text-extraction--pre-processing-🧪)**

> 💡 The full source code for this metadata cleaning and merge process is available in the notebook:  
> [`1_Thesis.ipynb`](./Notebooks/1_Thesis.ipynb)

