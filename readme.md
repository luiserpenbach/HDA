# 🚀 Rocket Data Ingest Tool: User Guide

**Version:** 1.0  
**Target Audience:** Test Engineers & Operators  
**Maintained By:** Data Systems Team

---

## 1. Overview

The **Rocket Data Ingest Tool** is a "Ground Control" dashboard designed to standardize how we save test data. Instead of manually creating folders and copying files, this tool automates the process to ensure:

1.  **Traceability:** Every test gets a unique ID and a "Metadata Passport" (`metadata.json`).
2.  **Consistency:** Folder structures are identical for every test.
3.  **Performance:** Large CSV sensor files are automatically converted to optimized Parquet format for faster analysis.

---

## 2. Installation & Startup

### First-Time Setup
Ensure your DAQ computer has Python installed. Open a terminal (Command Prompt/PowerShell) and run the following command to install the required tools:

```bash
pip install streamlit pandas pyarrow