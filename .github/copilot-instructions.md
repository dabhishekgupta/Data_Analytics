# Twitter Sentiment Analysis Data Analytics Project

This repository contains a machine learning project for Twitter sentiment analysis using Natural Language Processing (NLP) with Python, scikit-learn, and various text preprocessing libraries. The project includes both Python scripts and Jupyter notebooks for sentiment classification using Random Forest classifiers.

**CRITICAL: Always follow these instructions first and only fallback to additional search and context gathering if the information here is incomplete or found to be in error.**

## Working Effectively

### Bootstrap and Setup Environment
**NEVER CANCEL: Initial setup takes 5-10 minutes total. Set timeout to 15+ minutes.**

```bash
# Install core dependencies
pip3 install pandas matplotlib seaborn scikit-learn wordcloud jupyter

# Install NLP dependencies
pip3 install spacy textblob googletrans preprocess_kgptalkie

# Download spaCy English model (REQUIRED)
python3 -m spacy download en_core_web_sm
```

**CRITICAL DEPENDENCY ISSUE**: The original code uses `ps.get_basic_features(df)` which does not exist. Always use individual feature extraction functions:
```python
import preprocess_kgptalkie as ps
df['word_count'] = df['text'].apply(lambda x: ps.word_count(x))
df['char_count'] = df['text'].apply(lambda x: ps.char_count(x))
df['avg_word_len'] = df['text'].apply(lambda x: ps.avg_word_len(x))
df['stop_words_count'] = df['text'].apply(lambda x: ps.stop_words_count(x))
```

### Run the Python Script
**NEVER CANCEL: Script execution takes 10-15 seconds. Set timeout to 60+ seconds.**

```bash
# Run the main sentiment analysis script
python3 "Twitter_Sentiment _Analysis_NLP_Project.py"

# Run the corrected test version (recommended)
python3 test_script.py
```

### Run Jupyter Notebooks
**NEVER CANCEL: Notebook execution takes 1-2 minutes. Set timeout to 5+ minutes.**

```bash
# Start Jupyter server (interactive mode)
jupyter notebook "NLP Project  - Twitter Sentiment Analysis.ipynb"

# Convert and execute notebook (command line)
jupyter nbconvert --to notebook --execute "NLP Project  - Twitter Sentiment Analysis.ipynb" --output executed_notebook.ipynb
```

**NOTE**: The original notebook has the same `ps.get_basic_features()` issue as the script. Fix it before execution.

## Validation

**MANDATORY VALIDATION STEPS** - Always run these after making changes:

1. **Environment Validation**:
   ```bash
   python3 -c "import pandas, matplotlib, seaborn, sklearn, wordcloud, jupyter, preprocess_kgptalkie, spacy, textblob, googletrans; print('All dependencies OK')"
   python3 -c "import spacy; nlp = spacy.load('en_core_web_sm'); print('spaCy model OK')"
   ```

2. **Script Execution Test**:
   ```bash
   python3 test_script.py
   # Should complete in ~10 seconds and show "Script completed successfully!"
   # Should create twitter_sentiment.pkl file
   # Should report accuracy around 0.85-0.90
   ```

3. **Data Loading Validation**:
   ```bash
   python3 -c "import pandas as pd; df = pd.read_csv('https://raw.githubusercontent.com/laxmimerit/All-CSV-ML-Data-Files-Download/master/twitter_sentiment.csv', header=None, index_col=[0]); print(f'Dataset loaded: {df.shape}')"
   ```

4. **Model Validation**:
   ```bash
   python3 -c "import pickle; model = pickle.load(open('twitter_sentiment.pkl', 'rb')); print('Model loaded successfully'); print(f'Pipeline steps: {model.steps}')"
   ```

**CRITICAL VALIDATION SCENARIO**: Always test the complete workflow:
- Load Twitter sentiment dataset (75k+ samples)
- Preprocess and extract features
- Train Random Forest classifier
- Achieve accuracy > 85%
- Save model as pickle file
- Verify model can be loaded and used for predictions

## Common Issues and Fixes

### 1. Import Errors
- **Problem**: `ModuleNotFoundError` for spacy, textblob, etc.
- **Fix**: Run the complete dependency installation commands above
- **Time**: 5-10 minutes for all dependencies

### 2. SpaCy Model Missing
- **Problem**: `OSError: [E050] Can't find model 'en_core_web_sm'`
- **Fix**: `python3 -m spacy download en_core_web_sm`
- **Time**: 1-2 minutes

### 3. Function Not Found Error
- **Problem**: `AttributeError: module 'preprocess_kgptalkie' has no attribute 'get_basic_features'`
- **Fix**: Use individual functions as shown in the setup section above
- **Root Cause**: Original code uses non-existent function name

### 4. DataFrame Method Error
- **Problem**: `AttributeError: 'DataFrame' object has no attribute 'split'`
- **Fix**: Apply preprocess_kgptalkie functions to individual text entries, not entire DataFrame

## Repository Structure

### Key Files
```
.
├── README.md                                          # Basic project description
├── Twitter_Sentiment _Analysis_NLP_Project.py        # Main Python script (has bugs)
├── NLP Project  - Twitter Sentiment Analysis.ipynb   # Jupyter notebook (has bugs)
├── test_script.py                                     # Fixed working version
├── twitter_sentiment.pkl                             # Trained model output
└── .github/
    └── copilot-instructions.md                       # This file
```

### Data Source
- **External Dataset**: https://raw.githubusercontent.com/laxmimerit/All-CSV-ML-Data-Files-Download/master/twitter_sentiment.csv
- **Size**: ~75k tweets with sentiment labels
- **Classes**: Positive, Negative, Neutral, Irrelevant
- **No local data files required**

## Timing Expectations

**CRITICAL TIMING INFORMATION:**
- **Dependency Installation**: 5-10 minutes total (NEVER CANCEL)
- **Script Execution**: 10-15 seconds for full workflow
- **Model Training**: ~2-5 seconds (Random Forest with 10 estimators)
- **Data Loading**: ~2-3 seconds (downloads from internet)
- **Notebook Execution**: 1-2 minutes total

**Set appropriate timeouts:**
- Setup commands: 15+ minutes
- Script execution: 60+ seconds  
- Notebook execution: 5+ minutes

## Development Workflow

### Making Changes
1. **Always test the working version first**: `python3 test_script.py`
2. **Make minimal changes** to the problematic files
3. **Test immediately** after each change
4. **Validate model performance** remains above 85% accuracy

### Adding Features
1. **Text Preprocessing**: Use preprocess_kgptalkie individual functions
2. **Feature Engineering**: Add new columns using pandas apply()
3. **Model Changes**: Use scikit-learn Pipeline structure
4. **Always validate** with the complete workflow

### Working with Notebooks
- **Original Notebook Issue**: Contains `ps.get_basic_features(df)` bug
- **Fix Before Running**: Replace with individual function calls
- **Convert to Python**: `jupyter nbconvert --to python "NLP Project  - Twitter Sentiment Analysis.ipynb"`
- **Execute Non-Interactively**: `jupyter nbconvert --execute --to notebook notebook.ipynb`

### No Build System or CI/CD

**IMPORTANT**: This repository contains simple Python scripts and notebooks with no formal build system, continuous integration, or automated testing infrastructure.

- **No Makefile, setup.py, or build scripts**
- **No GitHub Actions workflows** 
- **No formal test suite** - validate manually using the validation steps above
- **No linting configuration** - but code should follow standard Python practices
- **No deployment process** - models are saved as pickle files locally

### Project Limitations
- **Internet Dependency**: Requires internet access to download Twitter dataset
- **No Unit Tests**: Validation is manual through script execution
- **No Documentation Generation**: README.md is minimal
- **No Version Management**: No semantic versioning or release process

## Common Tasks Output Reference

**Repository Root**:
```
$ ls -la
drwxr-xr-x 3 runner docker    4096 .
drwxr-xr-x 3 runner docker    4096 ..
drwxr-xr-x 7 runner docker    4096 .git
-rw-r--r-- 1 runner docker 2190339 'NLP Project  - Twitter Sentiment Analysis.ipynb'
-rw-r--r-- 1 runner docker      48  README.md
-rw-r--r-- 1 runner docker    3853 'Twitter_Sentiment _Analysis_NLP_Project.py'
```

**Available preprocess_kgptalkie functions**:
```python
# Text cleaning functions
ps.remove_urls(text)
ps.remove_html_tags(text)  
ps.remove_special_chars(text)
ps.remove_rt(text)  # Remove retweets

# Feature extraction functions  
ps.word_count(text)
ps.char_count(text)
ps.avg_word_len(text)
ps.stop_words_count(text)
```

**Model Pipeline Structure**:
```python
Pipeline([
    ('tfidf', TfidfVectorizer(max_features=5000, stop_words='english')),
    ('clf', RandomForestClassifier(n_estimators=10, n_jobs=-1, random_state=42))
])
```