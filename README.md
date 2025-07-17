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

### 🔊 Investor Communications and Earnings Call Transcripts

To complement the reports, I manually downloaded a diverse set of **transcripts from investor relations materials**, including **earnings calls**, **fireside chats**, and **Q&A sessions**. These materials are typically less polished and less controlled than formal sustainability or annual reports, making them particularly useful for more **objective ESG analysis**.

Unlike sustainability reports — which are often heavily curated for branding and compliance — investor communications respond directly to questions from analysts and stakeholders. This interactive nature makes them **less susceptible to greenwashing or corporate spin**, providing a more transparent view of a company’s priorities and risk management.

- Primary source: each company’s **Investor Relations** webpage (when available)
- Secondary source: **Seeking Alpha** (via a 28-day trial account) to supplement missing transcripts
- **Presentations were excluded**: they tend to be scripted and overly promotional
- Focus was placed on unscripted, **conversational formats** such as Q&A exchanges and analyst briefings

Due to **legal and technical limitations**, transcripts were collected manually. Automated scraping was not feasible: Seeking Alpha explicitly prohibits it in their [Terms of Use](https://seekingalpha.com/page/terms-of-use), and they implement active anti-scraping protections.

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

For the missing transcripts, I thoroughly searched across all available sources, including Investor Relations websites, Seeking Alpha, and other financial platforms. I also contacted someone with **Bloomberg Terminal access** (which has extensive coverage), but even there, the transcripts were unavailable. My working hypothesis is that **these earnings calls simply did not take place** — in **Switzerland**, unlike the **United States**, listed companies are **not legally required to hold four earnings calls per year**. Quarterly disclosures are common, but not mandatory or standardized.

➡️ *Source: SIX Swiss Exchange – [Periodic Reporting Requirements]([https://www.ser-ag.com/en/resources/regulations/directives/listing-rules.html](https://www.ser-ag.com/en/topics/regular-reporting.html))*

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

## Phase 2 – Dataset Exploration 🔍

In this phase, I perform an exploratory analysis of the merged dataset created in [Phase 1](#phase-1--dataset-construction-🧱). The goal is to verify data completeness, detect missing entries, and understand the distribution of document types.

### 🧮 Key Analyses

- **Total number of documents** per company and year  
- **Distribution of document types** (e.g. Annual Report, Sustainability Report, etc.)  
- **Pivot table** to visualize which types are available for each company-year combination  
- **Gap detection** to identify missing reports or transcripts  

A sample of the pivot table below shows how many documents of each type were collected for ABB Ltd between 2021 and 2023:

| Company | Year | Annual Report | Earnings Call Transcript | Governance Report | Half-Year Report | Integrated Report | Sustainability Report |
|---------|------|----------------|----------------------------|--------------------|-------------------|--------------------|------------------------|
| ABB Ltd | 2021 | 1              | 4                          | 0                  | 0                 | 0                  | 1                      |
| ABB Ltd | 2022 | 0              | 4                          | 1                  | 0                 | 1                  | 1                      |
| ABB Ltd | 2023 | 0              | 4                          | 1                  | 0                 | 1                  | 1                      |

This overview ensures the corpus is both **comprehensive and well-documented** before proceeding to the text extraction phase.

👉 For details and full visualizations, see the notebook [`2_Thesis.ipynb`](Notebooks/2_Thesis.ipynb).

### 🧠 Phase 3 – Sentence Extraction (NLP-ready)

This step is the most **crucial foundation** for the NLP classification phase. It involves extracting clean, meaningful, and self-contained sentences more than 200 corporate documents (Annual Reports, ESG Reports, etc.).

To achieve this, I built a sentence extraction pipeline using:

- [**PyMuPDF**](https://pymupdf.readthedocs.io/en/latest/): a fast and lightweight PDF parser that allows page-by-page access and precise text extraction from complex layouts.
- [**spaCy**](https://spacy.io/): a powerful industrial-strength NLP library that handles sentence segmentation, tokenization, and linguistic filtering.

Each document is parsed **page by page**, applying a series of custom cleaning operations:

- **Removal of repeated headers and footers** (e.g. company name, year, page numbers)
- **Exclusion of index pages** (detected using pattern frequency heuristics)
- **Sentence segmentation** with spaCy
- **Filtering out** short, noisy, numeric-only, or symbol-heavy text chunks

The resulting sentences were then saved to CSV for further processing.

Due to the complexity of the documents and the amount of layout noise, **this step took over 30 minutes to run** and had to be repeated **entirely from scratch**. Initially, I proceeded with the pipeline, assuming the extraction quality was sufficient. However, at the NLP classification stage, I noticed that the results were poor — many "sentences" were in fact titles, footers, page numbers, or table of contents entries that had been incorrectly parsed as meaningful content.

This significantly degraded model performance and introduced semantic noise. As a result, I had to go back to this sentence extraction phase, rebuild the cleaning logic, and reprocess **all documents again**, which took time but drastically improved the output quality. This experience highlighted how **crucial and foundational** this stage is for the success of the entire NLP pipeline: if sentence quality is poor, no downstream analysis can be trusted.

➡️ For full implementation details, see the notebook: [`3_SentenceExtraction.ipynb`](Notebooks/3_SentenceExtraction.ipynb)

### 🧠 Phase 4 – ESG Sentence Classification Using Transformer Models

This notebook performs sentence-level classification across all extracted company reports to assign ESG labels — **Environmental**, **Social**, **Governance**, or **None** — to each sentence.

We follow the methodology from [Tobias Schimanski's tutorial on Medium](https://medium.com/@schimanski.tobi/analyzing-esg-with-ai-and-nlp-tutorial-1-report-analysis-towards-esg-risks-and-opportunities-8daa2695f6c5), which is based on his academic paper:

> *“Bridging the gap in ESG measurement: Using NLP to quantify environmental, social, and governance communication”*  
> *Tobias Schimanski et al., 2023*

We use the **ESGBERT transformer models**, available from HuggingFace ([ESGBERT models repository](https://huggingface.co/ESGBERT)), which are fine-tuned BERT-based classifiers trained specifically to detect ESG content. Three separate models are loaded:
- `EnvironmentalBERT` for environmental content,
- `SocialBERT` for social-related content,
- `GovernanceBERT` for governance themes.

Each sentence is passed through all three models. Each model returns:
- a **label** (either the target class or `"none"`),
- and a **confidence score** between `0` and `1`.

#### 🔍 How the classification works

For each sentence:
- If the model predicts a relevant ESG category (e.g. `"environmental"`, `"social"`...), it returns a confidence score for that label.
- If no category surpasses a **confidence threshold of 0.5**, the sentence is assigned the label `"none"`.
- A **majority label** (or more precisely, the label with the highest score above 0.5) is then computed across the three categories.

Example result:

| company                   | year | document_type       | sentence                                                                                                                                                       | label_env   | score_env | label_soc | score_soc | label_gov | score_gov |
|---------------------------|------|----------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------|-------------|-----------|-----------|-----------|-----------|-----------|
| Compagnie Financière Richemont | 2022 | Sustainability Report | Chaired by dunhill’s CEO, the newly appointed Sustainability Committee ensures the implementation of Richemont’s strategy across the business... | environmental | 0.989977  | social    | 0.996938  | none      | 0.774423  |

This approach ensures that **each sentence is independently evaluated** for its ESG relevance, allowing nuanced and overlapping classifications.

#### 🧱 Batching for Large-Scale Classification

With **over 200,000 sentences** to classify, we split the dataset into **batches of 10,000** sentences for processing. This prevents memory overflow and allows intermediate saving of results. The batch loop:
1. Loads a slice of the data.
2. Applies the three ESG classifiers.
3. Saves the result to a dedicated folder in Google Drive.

After all batches are processed, they are concatenated into a single file and a final label column is assigned based on dominant confidence scores.

#### ⚙️ Running on GPU to Save Time (and Money)

Running transformer models is computationally intensive. Fortunately, Google Colab occasionally offers **free GPU access**. I was able to access a GPU for this classification step, which brought the total runtime down to just over **1 hour**.

Without GPU, this task would likely take several **hours or even days**, depending on hardware. However, after using the GPU for one full classification session, it became unavailable for the rest of the day — highlighting the **budgetary and infrastructural constraints** of this kind of academic project.

#### 🧵 Full Code Available

The entire classification pipeline — loading models, batching, applying prediction, and saving results — is detailed in  
📓 [`4_Classification_ESG.ipynb`](Notebooks/4_Classification_ESG.ipynb)


