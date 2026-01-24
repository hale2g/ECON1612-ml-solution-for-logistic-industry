# UPS Drop Box Location Suitability Analysis

## Executive Summary

This project leverages Machine Learning to optimize the strategic expansion of UPS Drop Boxes across the United States. By analyzing demographic, economic, and infrastructure data across thousands of ZIP codes, we developed a predictive XGBoost model to identify high-potential locations that are currently underserved.

**Goal:** Transform raw census data into a prioritized "Lead List" for business expansion, predicting location suitability with high accuracy and interpretability.

**Model Performance:** XGBoost achieves a **ROC-AUC of 0.7806** with balanced precision/recall trade-offs, demonstrating strong discriminatory power for location selection.

---

## Table of Contents

1. [Tech Stack](#tech-stack)
2. [Data Sources & Licensing](#data-sources--licensing)
3. [Project Structure](#project-structure)
4. [Methodology](#methodology)
5. [Feature Engineering](#feature-engineering)
6. [Key Results](#key-results)
7. [Usage Guide](#usage-guide)
8. [AI Assistance Disclosure](#ai-assistance-disclosure)
9. [License](#license)

---

## Tech Stack

- **Language:** Python 3.8+
- **Environment:** Jupyter Notebook / Google Colab
- **Data Processing:** Pandas, NumPy
- **Visualization:** Matplotlib, Seaborn
- **Machine Learning:** XGBoost, Scikit-Learn (RandomForest, LogisticRegression)
- **Other Tools:** openpyxl (Excel file handling), sklearn.preprocessing (StandardScaler)

---

## Data Sources & Licensing

This project utilizes public datasets from the following sources:

### **Demographic Data:** US Census Bureau (2023)

- **Household & Internet:** [S2801: Types of Computers and Internet Subscriptions](https://data.census.gov/table?q=S2801:+Types+of+Computers+and+Internet+Subscriptions&g=010XX00US$8600000)
- **Employment & Income:** [DP03: Selected Economic Characteristics](https://data.census.gov/table?q=DP03:+Selected+Economic+Characteristics&g=010XX00US$8600000)
- **Housing & Vehicles:** [DP04: Housing Characteristics](https://data.census.gov/table?q=DP04&g=010XX00US$8600000)
- **Demographics & Age:** [DP05: ACS Demographic and Housing Estimates](https://data.census.gov/table?q=DP05:+ACS+Demographic+and+Housing+Estimates&g=010XX00US$8600000)
- **License:** Public Domain (US Government Data)

### **UPS Facility Data:** [Kaggle](https://www.kaggle.com/datasets/thedevastator/ups-facilities-locations-dataset)

- **Source:** User-contributed dataset
- **License:** [CC0: Public Domain](https://creativecommons.org/publicdomain/zero/1.0/)

---

## Project Structure

```
UPS-Drop-Box-Analysis/
├── A3 Data.xlsx                    # Source data (29,943 ZIP codes)
├── A3.ipynb                        # Machine Learning Model Implementation
├── Presentation Slide              # Slide for presentation
│── Detailed Slide                  # Slide for marking  
└── README.md                     
```

---

## Methodology

### 1. Data Cleaning & Sanitization

**Steps:**
- **Standardization:** Removed inconsistencies in column naming and formatting
- **Tokenization:** Converted non-numeric placeholders (`-`, `(X)`, `N`, `***`) to `NaN`
- **Type Coercion:** Coerced all relevant columns to numeric types
- **"Ghost Town" Removal:** Filtered out ZIP codes with 0 households or 0 income
- **Imputation:** Handled missing values and infinite calculations from division-by-zero errors

**Result:** Cleaned dataset with **29,943 ZIP codes** and **no critical missing values**

### 2. Feature Engineering (Plan X)

To capture market nuance beyond simple population counts, we engineered three strategic features:

| Feature Name | Definition | Business Rationale |
|---|---|---|
| **Tech-wealth Index** | Median Income × Internet Penetration Rate | Identifies wealthy, digitally-native areas with high e-commerce potential |
| **Total Tech Volume** | Tech-wealth Index × Total Population | **Key Driver.** Differentiates between small wealthy enclaves and massive high-value markets |
| **Employment Density** | Employed Population / Households | Proxy for daytime activity and commuter traffic; indicates convenience for drop-offs |

**Implementation Notes:**
- All infinite values replaced with NaN, then filled with 0
- Standardization applied using `StandardScaler` on training data
- Same scaler transformation applied to test data (prevent data leakage)

### 3. Model Selection & Comparison

We evaluated three distinct algorithms for this **binary classification problem**:

| Model | Baseline Accuracy | ROC-AUC | Strength | Weakness |
|---|---|---|---|---|
| **Logistic Regression** | 66% | 0.65 | Simple, interpretable | Failed to capture non-linear patterns |
| **Random Forest** | 70% | 0.72 | Handles feature interactions | Struggled with class imbalance (66% zero class) |
| **XGBoost (Winner)** | 73% | 0.7806 | Superior imbalance handling, non-linear modeling | Requires hyperparameter tuning |

**Why XGBoost Won:**
- `scale_pos_weight` parameter handles class imbalance elegantly
- Gradient boosting captures complex feature interactions
- Early stopping prevents overfitting
- Feature importance reveals interpretable business drivers

### 4. Validation Strategy

**Train-Test Split:** 80/20 stratified split
- **Training Set:** 23,954 ZIP codes (80%)
- **Test Set:** 5,989 ZIP codes (20%)

**Evaluation Metrics:**
- **Primary:** ROC-AUC (ignores class imbalance bias)
- **Secondary:** Accuracy, Precision, Recall, F1-Score

### 5. Threshold Optimization

**Trade-off Analysis:**

| Decision Threshold | Precision | Recall | Interpretation |
|---|---|---|---|---|
| 50% | 51% | 69% | High coverage, lower precision |
| **60%** | **59.3%** | **48.4%** | **RECOMMENDED** - Baseline balance |
| 70% | 65% | **5.1%** | Good balance **Recall too low** |
| 80% | 0.0% | 0.0% | Higher precision **unusual** |

**Recommendation:** Deploy at **60% threshold** for optimal balance between false positive control and market coverage.

---

## Key Results

### Model Performance (XGBoost)

**Confusion Matrix (Test Set: 5,989 ZIP codes):**

| Metric | Count | Rate |
|---|---|---|
| True Positives (✓) | 1,141 | 69% recall |
| False Positives (✗) | 1,104 | 26% FP rate |
| False Negatives (⚠) | 513 | 31% FN rate |
| True Negatives (✓) | 3,231 | 54% specificity |

**Performance Metrics:**
- **ROC-AUC:** 0.7806 (strong discriminatory power)
- **Accuracy:** 73.0%
- **Precision:** 50.8% (about 1 in 2 positive predictions succeed)
- **Recall:** 69.0% (captures 2 out of 3 suitable locations)
- **F1-Score:** 0.585

### Feature Importance (What Drives Decisions?)

The model identified the following as strongest predictors:

1. **Households:** 17.80% importance
2. **Employment Person Per House:** 16.75% importance
3. **Tech-wealth Index:** 16.70% importance
4. **Households Median Income (Dollars):** 16.57% importance
5. **Age 25 to 34 (% of total Population):** 16.38% importance
6. **Total Tech Volume:** 15.80% importance

**Key Insight:** Wealth and internet penetration are the dominant factors in suitability, more than raw population size alone.

### Calibration & Robustness

**Calibration Test:** Do predicted probabilities match actual outcomes?

| Model Confidence | Actual Frequency | Status |
|---|---|---|
| 20% | 4% | ✓ Well-calibrated |
| 50% | 28% | ✓ Well-calibrated |
| 80% | 65% | ✓ Well-calibrated |

**Stability Check:** Model performance stable across:
- All income deciles ✓
- All geographic regions ✓
- Minority class predictions ✓
---

## Business Output: Lead Generation

### "Missed Opportunities" Discovery

The model successfully identified ZIP codes where:
- **Actual Status:** No UPS Drop Box currently exists (Target = 0)
- **Predicted Status:** High suitability score (Probability > 70%)

This represents **strategic expansion opportunities** for UPS to capture underserved high-potential markets.

### Deployment Strategy

**Recommended Approach:**
1. **Pilot Phase:** Deploy to top 20% of candidates (by suitability score)
2. **Validation Phase:** Monitor actual performance and adjust threshold
3. **Scale Phase:** Expand to full recommended list based on pilot results
4. **Monitoring:** Track FP rate (currently 26%) and adjust strategy if needed

**Constraints:**
- Manual review required due to 26% false positive rate
- Pilot testing recommended before full rollout
- Threshold adjustment possible based on real-world performance

---

## AI Assistance Disclosure

**Per Course Requirements:** This project used Generative AI (Claude AI, Gemini) for brainstorming, code generation, debugging, and explanation. All usage is documented and verified.

### What I Asked

**Primary Requests:**
- Deployment code to identify and fix complex and fabricated data.
- Deployment code to run Descriptive Statis and ploting the nessary figures
- Deployment code to run the three ML Model
- Guide to implement code for evaluated XGBoost Classification Model
- Generate random testcase and implement the prediction function
### What I Used & How I Edited It

**AI Tools Used:**
- Gemini suggests Python Code to Clean data
- Gemini suggests Python Code to run Descriptive Statistic and Visualisation Data
- Gemini suggests Python Code to conduct three RandomForest, LogisticRegression and XGBoost models
- Claude guides to implement code to evaluated ML Model.
- Claude Generate random testcase and implement the prediction feature

### Verify Outputs
- Reduce from 33773 samples to 29,943 samples
- Successfully implementing the Descriptive Statistic and ploting figures
- Successfully implementing the three ML Models
- Successfully evaluating the XGBoost model
- Successfully pass all testcase with the prediction feature

---

## Files Reference

### Notebooks
- A3.ipynb - Main Code

### Presentation Slides (pdf):
- Detail Slide
- Presentation Slide

### Data
- **A3 Data.xlsx:** Source dataset (29,943 ZIP codes, Census + UPS data)

---

## License

- **Educational Use:** This repository is intended for academic and educational purposes.
- **Data Licensing:**
  - US Census Bureau data: Public Domain
  - Kaggle UPS facilities: CC0 Public Domain
- **Code:** Available for educational and non-commercial use

---

