#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
  BATH-SEP-BNMF: BAYESIAN TOPIC MODELING VIA NMF (IMPLEMENTACAO PROPRIA)
================================================================================
  Complete NMF implementation for topic analysis of scientific abstracts
  Independent of a previous LDA analysis
  
  With a ROUND, PUBLICATION-READY word cloud (following the tutorial example)
================================================================================
"""

import os
import re
import sys
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from PIL import Image

warnings.filterwarnings('ignore')

# ============================================================================
# 0. PACKAGE CHECK WITH SAFE IMPORT
# ============================================================================
print("\n" + "="*70)
print("  CHECKING REQUIRED PACKAGES")
print("="*70)

missing_packages = []

try:
    import numpy as np
    print("   ✓ numpy")
except ImportError:
    missing_packages.append("numpy")
    print("   ✗ numpy (missing)")

try:
    import pandas as pd
    print("   ✓ pandas")
except ImportError:
    missing_packages.append("pandas")
    print("   ✗ pandas (missing)")

try:
    from scipy.sparse import csr_matrix, issparse
    import scipy.sparse as sp
    print("   ✓ scipy")
except ImportError:
    missing_packages.append("scipy")
    print("   ✗ scipy (missing)")

try:
    from sklearn.feature_extraction.text import CountVectorizer
    print("   ✓ sklearn")
except ImportError:
    missing_packages.append("scikit-learn")
    print("   ✗ scikit-learn (missing)")

try:
    import matplotlib.pyplot as plt
    print("   ✓ matplotlib")
except ImportError:
    missing_packages.append("matplotlib")
    print("   ✗ matplotlib (missing)")

try:
    import seaborn as sns
    print("   ✓ seaborn")
except ImportError:
    missing_packages.append("seaborn")
    print("   ✗ seaborn (missing)")

try:
    from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
    print("   ✓ wordcloud")
except ImportError:
    missing_packages.append("wordcloud")
    print("   ✗ wordcloud (missing)")

try:
    import nltk
    from nltk.corpus import stopwords as nltk_stopwords
    from nltk.stem import PorterStemmer
    print("   ✓ nltk")
except ImportError:
    missing_packages.append("nltk")
    print("   ✗ nltk (missing)")

try:
    import openpyxl
    print("   ✓ openpyxl")
except ImportError:
    missing_packages.append("openpyxl")
    print("   ✗ openpyxl (missing)")

try:
    from tqdm import tqdm
    print("   ✓ tqdm")
except ImportError:
    missing_packages.append("tqdm")
    print("   ✗ tqdm (missing)")

if missing_packages:
    print(f"\n❌ Missing packages: {', '.join(missing_packages)}")
    print(f"   Run: pip install {' '.join(missing_packages)}")
    sys.exit(1)

print("="*70)

# Download NLTK stop words if needed
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords', quiet=True)

# ============================================================================
# 1. SETTINGS
# ============================================================================
class Config:
    INPUT_DIR = Path(".")  # Current folder
    OUTPUT_DIR = Path("./BATH_SEP_NMF_RESULTS")
    WORDCLOUD_DIR = OUTPUT_DIR / "wordclouds"
    
    # Processing parameters
    MIN_ABSTRACT_LEN = 100
    MIN_WORD_FREQ = 2
    MAX_WORD_FREQ_PCT = 0.8
    MAX_FEATURES = 5000
    
    # BNMF parameters
    MIN_K = 3
    MAX_K = 12
    NMF_MAX_ITER = 300
    NMF_TOL = 1e-4

    # Bayesian layer: Gamma priors and posterior uncertainty
    BAYESIAN_SHAPE_W = 1.10
    BAYESIAN_RATE_W = 0.10
    BAYESIAN_SHAPE_H = 1.10
    BAYESIAN_RATE_H = 0.10
    BAYESIAN_POSTERIOR_SAMPLES = 250
    BAYESIAN_POSTERIOR_CONCENTRATION = 120.0
    ARD_ACTIVITY_THRESHOLD = 0.03
    
    # Language (Portuguese/English)
    LANGUAGE = 'english'

# Create folders
Config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
Config.WORDCLOUD_DIR.mkdir(parents=True, exist_ok=True)

print(f"\n[SETTINGS] Folders configured:")
print(f"   - Results: {Config.OUTPUT_DIR}")
print(f"   - Word clouds: {Config.WORDCLOUD_DIR}")

# ============================================================================
# 2. STOP WORDS (English + Portuguese + methodological terms)
# ============================================================================
def load_stop_words():
    """Loads the list of stop words to remove"""
    
    # Start with the default wordcloud STOPWORDS
    stop_words = set(STOPWORDS)
    
    # Default NLTK stop words
    try:
        stop_words.update(nltk_stopwords.words('english'))
    except:
        pass
    
    # Portuguese stop words (data: kept in Portuguese)
    portuguese_stops = {
        'a', 'e', 'o', 'que', 'de', 'da', 'do', 'em', 'um', 'para', 'com', 'nao', 'as', 'os',
        'se', 'mas', 'como', 'por', 'mais', 'sem', 'uma', 'dos', 'das', 'nos', 'nas', 'pelo',
        'pela', 'pelos', 'pelas', 'aos', 'ao', 'aquela', 'aquelas', 'aquele', 'aqueles',
        'aquilo', 'cada', 'coisa', 'contra', 'debaixo', 'dela', 'delas', 'dele', 'deles',
        'depois', 'dessa', 'dessas', 'desse', 'desses', 'desta', 'destas', 'deste', 'destes',
        'dispoe', 'dispoem', 'diversas', 'diversos', 'diz', 'dizem', 'do', 'dos', 'e', 'ela',
        'elas', 'ele', 'eles', 'em', 'entao', 'era', 'eram', 'essa', 'essas', 'esse', 'esses',
        'esta', 'estas', 'este', 'estes', 'estou', 'e', 'a', 'o', 'os', 'as', 'um', 'uma',
        'uns', 'umas', 'de', 'do', 'da', 'dos', 'das'
    }
    stop_words.update(portuguese_stops)
    
    # Methodological terms to remove
    methodological = {
        'study', 'research', 'paper', 'article', 'aim', 'objective', 'purpose',
        'method', 'result', 'conclusion', 'background', 'findings', 'literature',
        'review', 'discussion', 'introduction', 'analysis', 'data', 'analyses',
        'studies', 'results', 'conclusions', 'methods', 'methodology', 'using',
        'used', 'use', 'using', 'show', 'shows', 'shown', 'demonstrate', 
        'demonstrates', 'demonstrated', 'indicate', 'indicates', 'indicated',
        'suggest', 'suggests', 'suggested', 'reveal', 'reveals', 'revealed',
        'however', 'therefore', 'thus', 'hence', 'consequently', 'accordingly',
        'additionally', 'furthermore', 'moreover', 'nevertheless', 'nonetheless',
        'whereas', 'whereby', 'wherein', 'whereof', 'whereto', 'wherewith',
        'then', 'than', 'that', 'these', 'those', 'this', 'there', 'their', 
        'them', 'themselves', 'thence', 'thereafter', 'thereby', 'therefore',
        'therein', 'thereof', 'thereon', 'thereto', 'thereupon', 'therewith',
        'they', 'though', 'through', 'throughout', 'thru', 'thus', 'to', 
        'together', 'too', 'toward', 'towards', 'under', 'until', 'up', 'upon',
        'us', 'very', 'via', 'was', 'we', 'well', 'were', 'what', 'when', 
        'where', 'whereafter', 'whereas', 'whereby', 'wherein', 'whereupon',
        'wherever', 'whether', 'which', 'while', 'whom', 'whose', 'why', 
        'will', 'with', 'within', 'without', 'would', 'yes', 'yet', 'you',
        'your', 'yours', 'yourself', 'yourselves'
    }
    stop_words.update(methodological)
    
    # Words to keep (do NOT remove)
    preserve_words = {
        'agroforestry', 'agroforest', 'agroforests', 'smallholder', 'smallholders', 
        'soil', 'tree', 'trees', 'crop', 'crops', 'coffee', 'cocoa', 'maize', 'banana',
        'livestock', 'farming', 'farmer', 'farmers', 'agriculture', 'agricultural',
        'sustainable', 'sustainability', 'biodiversity', 'conservation',
        'deforestation', 'degradation', 'restoration', 'carbon', 'climate',
        'livelihood', 'livelihoods', 'nutrient', 'yield', 'productivity', 
        'ecosystem', 'ecosystems', 'forest', 'forests', 'management', 
        'practice', 'practices', 'system', 'systems', 'water', 'adaptation',
        'resilience', 'food', 'security', 'production', 'agroecosystem'
    }
    
    # Remove the kept words from the stop-word list
    stop_words = stop_words - preserve_words
    
    return list(stop_words), list(preserve_words)

stop_words, preserve_words = load_stop_words()
print(f"\n[STOP WORDS] Settings:")
print(f"   - Words to remove: {len(stop_words)}")
print(f"   - Words to keep: {len(preserve_words)}")

# ============================================================================
# 3. FUNCTIONS TO READ BIB FILES
# ============================================================================
def parse_bib_content(content, source_name):
    """Parses the content of a .bib file"""
    records = []
    
    # Split by @ entries
    entries = re.split(r'\n\s*@', content)
    
    for entry in entries:
        if not entry.strip():
            continue
        
        # Extract type and ID
        type_match = re.match(r'(\w+)\s*\{([^,]+),?', entry)
        if not type_match:
            continue
        
        entry_type = type_match.group(1).upper()
        entry_id = type_match.group(2).strip()
        
        # Extract fields
        record = {
            'ENTRYTYPE': entry_type,
            'ID': entry_id,
            'source_file': source_name
        }
        
        # Patterns for common fields
        patterns = {
            'author': r'[Aa]uthor\s*=\s*\{([^}]+)\}',
            'title': r'[Tt]itle\s*=\s*\{([^}]+)\}',
            'year': r'[Yy]ear\s*=\s*\{?(\d{4})\}?',
            'journal': r'[Jj]ournal\s*=\s*\{([^}]+)\}',
            'abstract': r'[Aa]bstract\s*=\s*\{([^}]+(?:\{[^}]*\}[^}]*)*)\}',
            'doi': r'[Dd][Oo][Ii]\s*=\s*\{([^}]+)\}',
            'keywords': r'[Kk]eywords\s*=\s*\{([^}]+)\}',
            'publisher': r'[Pp]ublisher\s*=\s*\{([^}]+)\}',
            'url': r'[Uu]rl\s*=\s*\{([^}]+)\}'
        }
        
        for field, pattern in patterns.items():
            match = re.search(pattern, entry, re.DOTALL | re.IGNORECASE)
            if match:
                value = match.group(1)
                # Clean the abstract (remove repeated spaces and line breaks)
                if field == 'abstract':
                    value = re.sub(r'\s+', ' ', value).strip()
                record[field] = value
        
        if 'abstract' in record and len(record['abstract']) > Config.MIN_ABSTRACT_LEN:
            records.append(record)
    
    return records

def load_bib_files(input_dir):
    """Loads every .bib file of the folder"""
    bib_files = list(Path(input_dir).glob("*.bib"))
    
    if not bib_files:
        raise ValueError(f"No .bib file found in {input_dir}")
    
    print(f"\n📁 .bib files found: {[f.name for f in bib_files]}")
    
    all_records = []
    for bib_file in bib_files:
        print(f"   Loading: {bib_file.name}...")
        try:
            with open(bib_file, 'r', encoding='utf-8') as f:
                content = f.read()
            records = parse_bib_content(content, bib_file.name)
            print(f"      ✓ {len(records)} records loaded")
            all_records.extend(records)
        except Exception as e:
            print(f"      ⚠️ Error loading {bib_file.name}: {e}")
    
    if not all_records:
        raise ValueError("No valid record found in the .bib files")
    
    df = pd.DataFrame(all_records)
    
    # Remove duplicates by title
    if 'title' in df.columns:
        df = df.drop_duplicates(subset=['title'], keep='first')
    
    print(f"\n✅ Unique records: {len(df)} documents")
    
    # Show the available columns
    print(f"📋 Available columns: {list(df.columns)}")
    
    return df

# ============================================================================
# 4. TEXT PREPROCESSING
# ============================================================================
def preprocess_text(text, stop_words, stemmer=None):
    """Preprocesses a text: cleaning, tokenization, stop-word removal, stemming"""
    if pd.isna(text) or not isinstance(text, str):
        return ""
    
    # Lower case
    text = text.lower()
    
    # Remove numbers
    text = re.sub(r'\d+', ' ', text)
    
    # Remove punctuation
    text = re.sub(r'[^\w\s]', ' ', text)
    
    # Remove accents (simplified)
    text = re.sub(r'[áàâãä]', 'a', text)
    text = re.sub(r'[éèêë]', 'e', text)
    text = re.sub(r'[íìîï]', 'i', text)
    text = re.sub(r'[óòôõö]', 'o', text)
    text = re.sub(r'[úùûü]', 'u', text)
    text = re.sub(r'ç', 'c', text)
    
    # Tokenizar
    words = text.split()
    
    # Filter stop words and short words
    words = [w for w in words if w not in stop_words and len(w) > 2]
    
    # Stemming (optional)
    if stemmer:
        words = [stemmer.stem(w) for w in words]
    
    return ' '.join(words)

def preprocess_documents(df, stop_words):
    """Preprocesses every document of the dataframe"""
    print(f"\n[PREPROCESSING] Processing {len(df)} documents...")
    
    # Make sure there is an abstract column
    if 'abstract' not in df.columns:
        if 'AB' in df.columns:
            df['abstract'] = df['AB']
        elif 'ABSTRACT' in df.columns:
            df['abstract'] = df['ABSTRACT']
        else:
            raise ValueError("Column 'abstract' not found")
    
    # Filter documents without an abstract
    df = df[df['abstract'].notna()].copy()
    df = df[df['abstract'].str.len() > Config.MIN_ABSTRACT_LEN].copy()
    print(f"   Documents with a valid abstract: {len(df)}")
    
    # Stemmer
    stemmer = PorterStemmer()
    
    # Preprocess abstracts
    abstracts = df['abstract'].tolist()
    processed = []
    
    for text in tqdm(abstracts, desc="   Processing abstracts"):
        processed_text = preprocess_text(text, stop_words, stemmer)
        processed.append(processed_text)
    
    df['text_clean'] = processed
    
    # Filter documents left empty after cleaning
    df = df[df['text_clean'].str.len() > 50].copy()
    print(f"   Documents after cleaning: {len(df)}")
    
    return df

# ============================================================================
# 5. DTM AND FREQUENCY LIST
# ============================================================================
def create_dtm(df, max_features=5000):
    """Creates a sparse document-term matrix"""
    print(f"\n[DTM] Creating the document-term matrix...")
    
    vectorizer = CountVectorizer(
        max_features=max_features,
        min_df=2,
        max_df=0.8,
        token_pattern=r'(?u)\b\w+\b'
    )
    
    dtm = vectorizer.fit_transform(df['text_clean'].tolist())
    
    # Get the terms
    terms = vectorizer.get_feature_names_out()
    
    print(f"   DTM: {dtm.shape[0]} documents x {dtm.shape[1]} terms")
    print(f"   Esparsidade: {100 * (1 - dtm.nnz / (dtm.shape[0] * dtm.shape[1])):.1f}%")
    
    return dtm, terms

def create_word_frequency(df, stop_words, max_terms=1000):
    """Creates the word-frequency list"""
    print(f"\n[FREQUENCY LIST] Processing {len(df)} documents...")
    
    all_words = []
    stemmer = PorterStemmer()
    
    for text in tqdm(df['abstract'].tolist(), desc="   Collecting words"):
        if pd.isna(text):
            continue
        words = preprocess_text(text, stop_words, stemmer).split()
        all_words.extend(words)
    
    word_freq = pd.DataFrame(Counter(all_words).most_common(max_terms), 
                              columns=['word', 'frequency'])
    
    print(f"   Unique terms: {len(word_freq)}")
    
    return word_freq

def export_words_for_review(word_freq, output_path):
    """Exports the word list for manual review"""
    df_review = word_freq.copy()
    df_review['remove'] = ''  # Mark words to remove
    df_review['keep'] = ''    # Mark words to keep
    df_review.to_excel(output_path, index=False)
    print(f"   ✅ Review list saved: {output_path}")
    print(f"   📝 Review this file and update the stop words as needed")

# ============================================================================
# 6. ROUND, PUBLICATION-READY WORD CLOUD (TUTORIAL STYLE)
# ============================================================================
def create_publication_wordcloud(word_freq, output_path, title, 
                                  width=1600, height=800, 
                                  background_color='white', 
                                  colormap='viridis',
                                  max_words=200):
    """
    Creates a word cloud in the style of the Airbnb tutorial.
    Standard rectangular format, visually clear and publication-ready.
    """
    if word_freq is None or len(word_freq) == 0:
        print("   ⚠️ No words to draw a word cloud")
        return
    
    # Create the frequency dictionary
    freq_dict = dict(zip(word_freq['word'].head(max_words), 
                         word_freq['frequency'].head(max_words)))
    
    # Set additional stop words
    stopwords = set(STOPWORDS)
    stopwords.update(['da', 'meu', 'em', 'você', 'de', 'ao', 'os', 'para', 'com', 'uma', 'como'])
    
    # Draw the word cloud in the tutorial style
    wordcloud = WordCloud(
        stopwords=stopwords,
        background_color=background_color,
        width=width,
        height=height,
        max_words=max_words,
        colormap=colormap,
        random_state=42,
        contour_width=1,
        contour_color='steelblue',
        relative_scaling=0.5,
        prefer_horizontal=0.7
    )
    
    wordcloud.generate_from_frequencies(freq_dict)
    
    # Show and save the image (tutorial style)
    fig, ax = plt.subplots(figsize=(width/100, height/100))
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.set_axis_off()
    ax.set_title(title, fontsize=16, pad=20, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor=background_color)
    plt.close()
    
    # Also save the file directly with the to_file method
    wordcloud.to_file(str(output_path).replace('.png', '_direct.png'))
    
    print(f"   ✅ Publication word cloud saved: {output_path} ({width}x{height})")
    
    return wordcloud

def create_round_wordcloud(word_freq, output_path, title, dpi=300, size=2400):
    """
    Creates a ROUND word cloud for special uses (cover, poster, etc.)
    """
    if word_freq is None or len(word_freq) == 0:
        print("   ⚠️ No words to draw a word cloud")
        return
    
    # Create the frequency dictionary
    freq_dict = dict(zip(word_freq['word'].head(200), word_freq['frequency'].head(200)))
    
    # Set stop words
    stopwords = set(STOPWORDS)
    stopwords.update(['da', 'meu', 'em', 'você', 'de', 'ao', 'os', 'para', 'com'])
    
    # Draw the round word cloud
    wordcloud = WordCloud(
        stopwords=stopwords,
        background_color='white',
        width=size,
        height=size,
        max_words=200,
        colormap='viridis',
        random_state=42,
        contour_width=2,
        contour_color='#2c3e50',
        relative_scaling=0.5,
        prefer_horizontal=0.7
    )
    
    wordcloud.generate_from_frequencies(freq_dict)
    
    # Plot in round format
    fig, ax = plt.subplots(figsize=(size/100, size/100), facecolor='white')
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.axis('off')
    ax.set_aspect('equal')
    ax.set_title(title, fontsize=size//20, pad=size//40, fontweight='bold')
    
    plt.tight_layout(pad=0)
    plt.savefig(output_path, dpi=dpi, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"   ✅ Round word cloud saved: {output_path} ({size}x{size}, {dpi} DPI)")
    
    return wordcloud

# ============================================================================
# 7. MANUAL NMF IMPLEMENTATION
# ============================================================================
class NMFModel:
    """Manual NMF implementation for topic modelling"""
    
    def __init__(self, n_components, max_iter=200, tol=1e-4, random_state=42,
                 shape_w=None, rate_w=None, shape_h=None, rate_h=None,
                 posterior_samples=None, posterior_concentration=None):
        self.n_components = n_components
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state
        self.shape_w = Config.BAYESIAN_SHAPE_W if shape_w is None else shape_w
        self.rate_w = Config.BAYESIAN_RATE_W if rate_w is None else rate_w
        self.shape_h = Config.BAYESIAN_SHAPE_H if shape_h is None else shape_h
        self.rate_h = Config.BAYESIAN_RATE_H if rate_h is None else rate_h
        self.posterior_samples = Config.BAYESIAN_POSTERIOR_SAMPLES if posterior_samples is None else posterior_samples
        self.posterior_concentration = (
            Config.BAYESIAN_POSTERIOR_CONCENTRATION
            if posterior_concentration is None else posterior_concentration
        )
        self.W = None  # Documents x Topics
        self.H = None  # Topics x terms
        self.reconstruction_error_ = []
        self.log_posterior_ = []
        self.posterior_summary_ = None
    
    def _initialize(self, X):
        """Differentiated initialization of W and H"""
        np.random.seed(self.random_state)
        
        n_samples, n_features = X.shape
        
        # SVD-based initialization for better quality
        try:
            from scipy.sparse.linalg import svds
            # For sparse matrices
            if hasattr(X, 'toarray'):
                X_dense = X.toarray()
            else:
                X_dense = X
            
            # Compute the SVD
            k = min(self.n_components, min(X_dense.shape) - 1)
            if k >= 2:
                u, s, vt = svds(X_dense, k=k)
                
                # Initialize W and H with positive components
                self.W = np.abs(u[:, :self.n_components])
                self.H = np.abs(vt[:self.n_components, :])
            else:
                raise ValueError("SVD not applicable")
            
            # Add noise to break symmetries
            self.W += np.random.rand(n_samples, self.n_components) * 0.01
            self.H += np.random.rand(self.n_components, n_features) * 0.01
            
        except:
            # Fallback: random initialization
            self.W = np.random.rand(n_samples, self.n_components)
            self.H = np.random.rand(self.n_components, n_features)
        
        # Ensure non-negativity and normalize
        self.W = np.maximum(self.W, 0)
        self.H = np.maximum(self.H, 0)
        
        # Normalize H (each topic sums to 1)
        self.H = self.H / (self.H.sum(axis=1, keepdims=True) + 1e-10)
    
    def fit(self, X):
        """Runs the NMF algorithm with multiplicative rules"""
        
        # Convert to a dense matrix if needed
        if hasattr(X, 'toarray'):
            X = X.toarray()
        
        X = np.maximum(X, 0)
        n_samples, n_features = X.shape
        
        # Initialization
        self._initialize(X)
        
        eps = 1e-10
        shape_w_minus_one = max(self.shape_w - 1.0, 0.0)
        shape_h_minus_one = max(self.shape_h - 1.0, 0.0)
        
        for iteration in range(self.max_iter):
            W_old = self.W.copy()
            H_old = self.H.copy()
            
            # Atualizar W: W = W * (X H^T) / (W H H^T)
            X_hat = self.W @ self.H + eps
            numerator_W = (X / X_hat) @ self.H.T + shape_w_minus_one / (self.W + eps)
            denominator_W = np.sum(self.H, axis=1, keepdims=True).T + self.rate_w + eps
            self.W = self.W * numerator_W / denominator_W
            
            # Atualizar H: H = H * (W^T X) / (W^T W H)
            X_hat = self.W @ self.H + eps
            numerator_H = self.W.T @ (X / X_hat) + shape_h_minus_one / (self.H + eps)
            denominator_H = np.sum(self.W, axis=0, keepdims=True).T + self.rate_h + eps
            self.H = self.H * numerator_H / denominator_H
            
            # Normalizar H
            self.H = self.H / (self.H.sum(axis=1, keepdims=True) + eps)
            
            # Compute the reconstruction error
            X_hat = self.W @ self.H
            error = np.sum((X - X_hat) ** 2) / (n_samples * n_features)
            self.reconstruction_error_.append(error)
            log_likelihood = np.sum(X * np.log(X_hat + eps) - X_hat)
            log_prior_w = np.sum((self.shape_w - 1.0) * np.log(self.W + eps) - self.rate_w * self.W)
            log_prior_h = np.sum((self.shape_h - 1.0) * np.log(self.H + eps) - self.rate_h * self.H)
            self.log_posterior_.append(float(log_likelihood + log_prior_w + log_prior_h))
            
            # Check convergence
            W_change = np.max(np.abs(self.W - W_old))
            H_change = np.max(np.abs(self.H - H_old))
            
            if iteration % 50 == 0:
                print(f"      Iter {iteration}: error={error:.4f}, logpost={self.log_posterior_[-1]:.2f}, dW={W_change:.6f}, dH={H_change:.6f}")
            
            if W_change < self.tol and H_change < self.tol:
                print(f"      ✅ Converged in {iteration} iterations")
                break
        
        # Normalize W (topic distribution per document)
        self.W = self.W / (self.W.sum(axis=1, keepdims=True) + eps)
        
        return self

    def estimate_posterior_uncertainty(self, random_state=None):
        """Approximate posterior uncertainty around W and H with Gamma sampling."""
        if self.W is None or self.H is None:
            raise ValueError("Fit the BNMF model before estimating posterior uncertainty.")

        rng = np.random.default_rng(self.random_state if random_state is None else random_state)
        eps = 1e-10
        n_samples = int(max(1, self.posterior_samples))
        concentration = float(max(self.posterior_concentration, 1.0))

        topic_activity_samples = []
        doc_max_prob_samples = []

        h_mean_accum = np.zeros_like(self.H)
        h_sq_accum = np.zeros_like(self.H)
        w_mean_accum = np.zeros_like(self.W)
        w_sq_accum = np.zeros_like(self.W)

        for _ in range(n_samples):
            w_shape = np.maximum(self.W * concentration, eps)
            h_shape = np.maximum(self.H * concentration, eps)
            w_sample = rng.gamma(shape=w_shape, scale=1.0 / concentration)
            h_sample = rng.gamma(shape=h_shape, scale=1.0 / concentration)
            w_sample = w_sample / (w_sample.sum(axis=1, keepdims=True) + eps)
            h_sample = h_sample / (h_sample.sum(axis=1, keepdims=True) + eps)

            topic_activity_samples.append(w_sample.mean(axis=0))
            doc_max_prob_samples.append(w_sample.max(axis=1))
            h_mean_accum += h_sample
            h_sq_accum += h_sample ** 2
            w_mean_accum += w_sample
            w_sq_accum += w_sample ** 2

        topic_activity_samples = np.asarray(topic_activity_samples)
        doc_max_prob_samples = np.asarray(doc_max_prob_samples)
        h_mean = h_mean_accum / n_samples
        h_std = np.sqrt(np.maximum((h_sq_accum / n_samples) - h_mean ** 2, 0.0))
        w_mean = w_mean_accum / n_samples
        w_std = np.sqrt(np.maximum((w_sq_accum / n_samples) - w_mean ** 2, 0.0))

        self.posterior_summary_ = {
            "topic_activity_mean": topic_activity_samples.mean(axis=0),
            "topic_activity_std": topic_activity_samples.std(axis=0),
            "topic_activity_ci_low": np.quantile(topic_activity_samples, 0.025, axis=0),
            "topic_activity_ci_high": np.quantile(topic_activity_samples, 0.975, axis=0),
            "inactive_topic_probability": (topic_activity_samples < Config.ARD_ACTIVITY_THRESHOLD).mean(axis=0),
            "doc_max_prob_mean": doc_max_prob_samples.mean(axis=0),
            "doc_max_prob_std": doc_max_prob_samples.std(axis=0),
            "doc_max_prob_ci_low": np.quantile(doc_max_prob_samples, 0.025, axis=0),
            "doc_max_prob_ci_high": np.quantile(doc_max_prob_samples, 0.975, axis=0),
            "H_mean": h_mean,
            "H_std": h_std,
            "W_mean": w_mean,
            "W_std": w_std,
        }
        return self.posterior_summary_
    
    def get_top_words(self, terms, n_words=15):
        """Returns the top words per topic"""
        top_words = []
        for topic_idx in range(self.n_components):
            topic_dist = self.H[topic_idx, :]
            top_indices = np.argsort(topic_dist)[-n_words:][::-1]
            top_words.append([terms[i] for i in top_indices])
        return top_words
    
    def get_topic_assignments(self):
        """Assigns each document to the topic with the highest probability"""
        return np.argmax(self.W, axis=1) + 1  # 1-indexed for compatibility

# ============================================================================
# 8. SELECTION OF K (NUMBER OF TOPICS)
# ============================================================================
def compute_nmf_metrics(X, k, max_iter=100):
    """Computes quality metrics for a given k"""
    try:
        nmf = NMFModel(n_components=k, max_iter=max_iter, tol=1e-3)
        nmf.fit(X)
        
        # Metrics
        n_samples, n_features = X.shape
        if hasattr(X, 'toarray'):
            X_dense = X.toarray()
        else:
            X_dense = X
        
        X_hat = nmf.W @ nmf.H
        
        # 1. Reconstruction error
        reconstruction_error = np.sum((X_dense - X_hat) ** 2) / (n_samples * n_features)
        
        # 2. Sparsity of the topic distribution
        topic_dist = nmf.W.mean(axis=0)
        topic_entropy = -np.sum(topic_dist * np.log(topic_dist + 1e-10))
        active_topics = int(np.sum(topic_dist >= Config.ARD_ACTIVITY_THRESHOLD))
        ard_penalty = 1.0 - ((k - active_topics) / max(k, 1))
        log_posterior = nmf.log_posterior_[-1] if nmf.log_posterior_ else np.nan
        
        # 3. Topic exclusivity (based on the top words)
        top_words_per_topic = nmf.get_top_words([f"w{i}" for i in range(n_features)], n_words=10)
        all_top = set()
        for words in top_words_per_topic:
            all_top.update(words)
        exclusivity = 0
        for word in all_top:
            count = sum(1 for words in top_words_per_topic if word in words)
            if count == 1:
                exclusivity += 1
        exclusivity_ratio = exclusivity / len(all_top) if all_top else 0
        
        return {
            'k': k,
            'reconstruction_error': reconstruction_error,
            'topic_entropy': topic_entropy,
            'exclusivity': exclusivity_ratio,
            'active_topics_ard': active_topics,
            'ard_activity_score': ard_penalty,
            'log_posterior': log_posterior,
            'converged': True
        }
    except Exception as e:
        print(f"   NMF error for k={k}: {str(e)[:50]}")
        return {'k': k, 'converged': False}

def select_optimal_k(dtm, min_k=3, max_k=12):
    """Selects the optimal number of topics with several metrics"""
    print(f"\n[SELECTING K] Testing K from {min_k} to {max_k}...")
    
    results = []
    for k in range(min_k, max_k + 1):
        print(f"   Testando K = {k}...", end=" ")
        metrics = compute_nmf_metrics(dtm, k)
        if metrics['converged']:
            results.append(metrics)
            print(f"OK error={metrics['reconstruction_error']:.4f}; active_ARD={metrics['active_topics_ard']}/{k}")
        else:
            print("Falhou")
    
    if not results:
        raise ValueError("Metrics could not be computed for any K")
    
    df_metrics = pd.DataFrame(results)
    
    # Normalize the metrics
    df_metrics['reconstruction_norm'] = 1 - (df_metrics['reconstruction_error'] - df_metrics['reconstruction_error'].min()) / (df_metrics['reconstruction_error'].max() - df_metrics['reconstruction_error'].min() + 1e-10)
    df_metrics['entropy_norm'] = df_metrics['topic_entropy'] / df_metrics['topic_entropy'].max()
    df_metrics['exclusivity_norm'] = df_metrics['exclusivity']
    df_metrics['ard_norm'] = df_metrics['ard_activity_score']
    
    # Composite score with the Bayesian-ARD layer
    df_metrics['composite_score'] = (
        0.35 * df_metrics['reconstruction_norm'] +
        0.25 * df_metrics['entropy_norm'] +
        0.25 * df_metrics['exclusivity_norm'] +
        0.15 * df_metrics['ard_norm']
    )
    
    optimal_k = df_metrics.loc[df_metrics['composite_score'].idxmax(), 'k']
    
    print(f"\n🎯 OPTIMAL K SELECTED: {optimal_k}")
    
    return int(optimal_k), df_metrics

# ============================================================================
# 9. PLOTS
# ============================================================================
def plot_k_selection(metrics_df, output_path):
    """Plots the K-selection charts"""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    k_values = metrics_df['k'].values
    
    # 1. Reconstruction error
    axes[0, 0].plot(k_values, metrics_df['reconstruction_error'], 'bo-', linewidth=2)
    axes[0, 0].set_xlabel('Number of Topics (K)', fontsize=11)
    axes[0, 0].set_ylabel('Reconstruction Error', fontsize=11)
    axes[0, 0].set_title('Reconstruction Error (lower is better)', fontsize=12)
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. Entropia
    axes[0, 1].plot(k_values, metrics_df['topic_entropy'], 'ro-', linewidth=2)
    axes[0, 1].set_xlabel('Number of Topics (K)', fontsize=11)
    axes[0, 1].set_ylabel('Distribution Entropy', fontsize=11)
    axes[0, 1].set_title('Entropy (higher is more balanced)', fontsize=12)
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. Exclusivity
    axes[1, 0].plot(k_values, metrics_df['exclusivity'], 'go-', linewidth=2)
    axes[1, 0].set_xlabel('Number of Topics (K)', fontsize=11)
    axes[1, 0].set_ylabel('Exclusivity', fontsize=11)
    axes[1, 0].set_title('Word Exclusivity (higher is better)', fontsize=12)
    axes[1, 0].grid(True, alpha=0.3)
    
    # 4. Score composto
    axes[1, 1].plot(k_values, metrics_df['composite_score'], 'purple', marker='s', linewidth=2, markersize=8)
    best_k = metrics_df.loc[metrics_df['composite_score'].idxmax(), 'k']
    axes[1, 1].axvline(x=best_k, color='red', linestyle='--', label=f'Optimal K = {best_k}')
    axes[1, 1].set_xlabel('Number of Topics (K)', fontsize=11)
    axes[1, 1].set_ylabel('Composite Score', fontsize=11)
    axes[1, 1].set_title('Composite Score (higher is better)', fontsize=12)
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.suptitle('Optimal Number of Topics Selection', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"   ✅ K-selection chart saved")

def plot_topic_distribution(topic_counts, output_path, title="Topic Distribution"):
    """Plots the topic distribution"""
    fig, ax = plt.subplots(figsize=(12, 6))
    
    topics = sorted(topic_counts.keys())
    counts = [topic_counts[t] for t in topics]
    percentages = [c/sum(counts)*100 for c in counts]
    
    bars = ax.bar(topics, counts, color=plt.cm.viridis(np.linspace(0.2, 0.8, len(topics))))
    ax.set_xlabel('Topic', fontsize=12)
    ax.set_ylabel('Number of Documents', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    
    # Add percentage labels
    for bar, count, pct in zip(bars, counts, percentages):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f'{count}\n({pct:.1f}%)', ha='center', va='bottom', fontsize=9)
    
    ax.set_xticks(topics)
    ax.set_xticklabels([f'Topic {t}' for t in topics])
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"   ✅ Distribution chart saved")

def plot_topic_correlation(W, output_path):
    """Plots the topic correlation matrix"""
    corr_matrix = np.corrcoef(W.T)
    
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='RdBu_r', 
                center=0, vmin=-1, vmax=1, square=True,
                xticklabels=[f'T{i+1}' for i in range(W.shape[1])],
                yticklabels=[f'T{i+1}' for i in range(W.shape[1])],
                ax=ax)
    ax.set_title('Correlation Between Topics', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"   ✅ Correlation matrix saved")

def plot_importance_scores(importance_df, output_path):
    """Plots the topic importance scores (CORRECTED for inf values)"""
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Keep valid values (remove inf and nan)
    valid_data = importance_df[importance_df['total_score'] != float('inf')].copy()
    valid_data = valid_data[valid_data['total_score'].notna()].copy()
    
    if len(valid_data) == 0:
        print("   ⚠️ No valid score to plot")
        return
    
    topics = valid_data['topic'].values
    scores = valid_data['total_score'].values
    
    # Replace inf with a reasonable maximum
    scores = np.nan_to_num(scores, nan=0.0, posinf=100.0, neginf=0.0)
    
    colors = plt.cm.RdYlGn(np.linspace(0.2, 0.8, len(topics)))[::-1]
    bars = ax.bar(range(len(topics)), scores, color=colors)
    
    ax.set_xlabel('Topic', fontsize=12)
    ax.set_ylabel('Importance Score', fontsize=12)
    ax.set_title('Topic Prioritization', fontsize=14, fontweight='bold')
    ax.set_xticks(range(len(topics)))
    ax.set_xticklabels([f'Topic {t}' for t in topics])
    
    # Safe upper limit
    max_score = np.max(scores)
    if np.isfinite(max_score):
        ax.set_ylim(0, min(max_score * 1.15, 110))
    else:
        ax.set_ylim(0, 100)
    
    ax.axhline(y=80, color='darkgreen', linestyle='--', alpha=0.7, label='Critical Priority (80+)')
    ax.axhline(y=60, color='orange', linestyle='--', alpha=0.7, label='High Priority (60-80)')
    ax.axhline(y=40, color='gold', linestyle='--', alpha=0.7, label='Medium Priority (40-60)')
    ax.legend(loc='upper right')
    ax.grid(axis='y', alpha=0.3)
    
    # Add labels
    for bar, score in zip(bars, scores):
        if np.isfinite(score):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                    f'{score:.1f}', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"   ✅ Importance chart saved")

# ============================================================================
# 10. TOPIC PRIORITIZATION
# ============================================================================
def compute_topic_importance(nmf_model, terms, topic_counts):
    """Computes the importance score of each topic"""
    
    n_topics = nmf_model.n_components
    n_docs = nmf_model.W.shape[0]
    
    importance_scores = []
    
    # Maximum documents for normalization
    max_docs = max(topic_counts.values()) if topic_counts else 1
    
    for t in range(n_topics):
        # 1. Frequency score (0-40)
        doc_count = topic_counts.get(t+1, 0)
        freq_score = (doc_count / max_docs) * 40 if max_docs > 0 else 0
        
        # 2. Exclusivity score (0-30) - avoid division by zero
        top_words_idx = np.argsort(nmf_model.H[t, :])[-10:]
        exclusivity = 0
        for idx in top_words_idx:
            appears_in = np.sum(nmf_model.H[:, idx] > 0.01)
            if appears_in > 0:
                exclusivity += 1.0 / appears_in
        exclusivity_score = (exclusivity / 10) * 30 if exclusivity > 0 else 0
        
        # 3. Coherence score (0-20)
        topic_weights = nmf_model.H[t, top_words_idx]
        max_weight = np.max(topic_weights) if len(topic_weights) > 0 and np.max(topic_weights) > 0 else 1
        coherence = (np.mean(topic_weights) / max_weight) * 20 if max_weight > 0 else 0
        
        # 4. Distinctiveness score (0-10)
        shared_count = 0
        for idx in top_words_idx:
            for other_t in range(n_topics):
                if other_t != t and nmf_model.H[other_t, idx] > 0.01:
                    shared_count += 1
                    break
        distinction_score = max(0, 10 - shared_count)
        
        total_score = freq_score + exclusivity_score + coherence + distinction_score
        
        # Make sure the score is finite
        if not np.isfinite(total_score):
            total_score = freq_score + coherence + distinction_score
        
        importance_scores.append({
            'topic': t + 1,
            'documents': doc_count,
            'percentage': (doc_count / n_docs * 100) if n_docs > 0 else 0,
            'freq_score': round(freq_score, 2),
            'exclusivity_score': round(exclusivity_score, 2) if np.isfinite(exclusivity_score) else 0,
            'coherence_score': round(coherence, 2) if np.isfinite(coherence) else 0,
            'distinction_score': round(distinction_score, 2) if np.isfinite(distinction_score) else 0,
            'total_score': round(total_score, 2) if np.isfinite(total_score) else freq_score + coherence + distinction_score
        })
    
    df_importance = pd.DataFrame(importance_scores)
    
    # Classify priority
    if len(df_importance) >= 3:
        valid_scores = df_importance[df_importance['total_score'] != float('inf')]['total_score']
        if len(valid_scores) > 0:
            q80 = valid_scores.quantile(0.8)
            q60 = valid_scores.quantile(0.6)
            q40 = valid_scores.quantile(0.4)
            
            df_importance['priority'] = df_importance['total_score'].apply(
                lambda x: 'CRITICAL' if x >= q80 else ('HIGH' if x >= q60 else ('MEDIUM' if x >= q40 else 'LOW'))
            )
        else:
            df_importance['priority'] = 'MEDIUM'
    else:
        df_importance['priority'] = 'HIGH'
    
    # Sort by score
    df_importance = df_importance.sort_values('total_score', ascending=False).reset_index(drop=True)
    df_importance['rank'] = df_importance.index + 1
    
    return df_importance

def export_bayesian_posterior(nmf_model, terms, df, output_path):
    """Exports Bayesian posterior uncertainty diagnostics for BNMF."""
    posterior = nmf_model.estimate_posterior_uncertainty()

    topic_rows = []
    for t in range(nmf_model.n_components):
        topic_rows.append({
            'topic': t + 1,
            'activity_mean': posterior['topic_activity_mean'][t],
            'activity_std': posterior['topic_activity_std'][t],
            'activity_ci_2_5': posterior['topic_activity_ci_low'][t],
            'activity_ci_97_5': posterior['topic_activity_ci_high'][t],
            'inactive_probability_ard': posterior['inactive_topic_probability'][t],
            'ard_status': 'inactive_candidate' if posterior['inactive_topic_probability'][t] >= 0.5 else 'active',
        })
    df_topic_uncertainty = pd.DataFrame(topic_rows)

    word_rows = []
    h_mean = posterior['H_mean']
    h_std = posterior['H_std']
    for t in range(nmf_model.n_components):
        top_idx = np.argsort(h_mean[t, :])[-15:][::-1]
        for rank, idx in enumerate(top_idx, 1):
            mean_weight = h_mean[t, idx]
            sd_weight = h_std[t, idx]
            word_rows.append({
                'topic': t + 1,
                'rank': rank,
                'word': terms[idx],
                'posterior_mean_weight': mean_weight,
                'posterior_std_weight': sd_weight,
                'ci_2_5': max(0.0, mean_weight - 1.96 * sd_weight),
                'ci_97_5': mean_weight + 1.96 * sd_weight,
            })
    df_word_uncertainty = pd.DataFrame(word_rows)

    df_doc_uncertainty = pd.DataFrame({
        'title': df['title'].values if 'title' in df.columns else np.arange(len(df)),
        'topic_bnmf': df['topic_nmf'].values,
        'topic_prob_map': df['topic_prob'].values,
        'topic_prob_posterior_mean': posterior['doc_max_prob_mean'],
        'topic_prob_posterior_std': posterior['doc_max_prob_std'],
        'topic_prob_ci_2_5': posterior['doc_max_prob_ci_low'],
        'topic_prob_ci_97_5': posterior['doc_max_prob_ci_high'],
    })

    with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
        df_topic_uncertainty.to_excel(writer, sheet_name='Topic_Uncertainty', index=False)
        df_word_uncertainty.to_excel(writer, sheet_name='Word_Uncertainty', index=False)
        df_doc_uncertainty.to_excel(writer, sheet_name='Document_Uncertainty', index=False)

    print(f"   OK Bayesian diagnostics saved: {output_path}")
    return df_topic_uncertainty, df_word_uncertainty, df_doc_uncertainty

# ============================================================================
# 11. MAIN FUNCTION
# ============================================================================
def main():
    print("\n" + "="*70)
    print("  BATH-SEP-BNMF: Bayesian Topic Modeling via NMF")
    print("="*70)
    
    # 1. Load data
    print("\n[LOADING] Loading bibliographic databases...")
    try:
        df = load_bib_files(Config.INPUT_DIR)
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        return None
    
    # 2. Preprocessing
    df = preprocess_documents(df, stop_words)
    
    if len(df) == 0:
        print("\n❌ ERROR: No document with a valid abstract after filtering.")
        return None
    
    # 3. Frequency list
    word_freq = create_word_frequency(df, stop_words, max_terms=1000)
    word_freq.to_excel(Config.OUTPUT_DIR / "00_word_frequency.xlsx", index=False)
    
    # 3.5 Export words for manual review
    export_words_for_review(word_freq, Config.OUTPUT_DIR / "00_words_for_review.xlsx")
    
    # 4. Word cloud (PUBLICATION VERSION - tutorial style)
    print("\n[WORD CLOUD] Drawing the publication word cloud (tutorial style)...")
    
    # Version 1: standard rectangular format (recommended for articles)
    create_publication_wordcloud(
        word_freq, 
        Config.WORDCLOUD_DIR / "01_wordcloud_publication.png",
        title="Word Cloud - Bibliographic Analysis",
        width=1600, height=800,
        background_color='white',
        colormap='viridis',
        max_words=200
    )
    
    # Version 2: round format (for covers, posters, etc.)
    create_round_wordcloud(
        word_freq,
        Config.WORDCLOUD_DIR / "01_wordcloud_round.png",
        title="Word Cloud - Bibliographic Analysis",
        dpi=300, size=2400
    )
    
    # 5. Create the DTM
    dtm, terms = create_dtm(df, max_features=Config.MAX_FEATURES)
    
    # 6. Select the optimal K
    max_k_allowed = min(Config.MAX_K, dtm.shape[0] // 5)
    if max_k_allowed < Config.MIN_K:
        max_k_allowed = Config.MIN_K + 1
    optimal_k, k_metrics = select_optimal_k(dtm, Config.MIN_K, max_k_allowed)
    
    # 7. Save the K-selection metrics
    k_metrics.to_excel(Config.OUTPUT_DIR / "02_k_selection_metrics.xlsx", index=False)
    plot_k_selection(k_metrics, Config.OUTPUT_DIR / "02_k_selection_plot.png")
    
    # 8. Run NMF with the optimal K
    print(f"\n[RUNNING BNMF] Training the model with K={optimal_k}...")
    nmf_model = NMFModel(n_components=optimal_k, max_iter=Config.NMF_MAX_ITER, tol=Config.NMF_TOL)
    nmf_model.fit(dtm)
    
    # 9. Extract results
    topic_assignments = nmf_model.get_topic_assignments()
    topic_probs = nmf_model.W
    
    # Add topics to the dataframe
    df['topic_nmf'] = topic_assignments
    df['topic_prob'] = np.max(topic_probs, axis=1)
    
    # 10. Topic distribution
    topic_counts = df['topic_nmf'].value_counts().to_dict()
    plot_topic_distribution(topic_counts, Config.OUTPUT_DIR / "03_topic_distribution.png")
    
    # 11. Topic correlation matrix
    plot_topic_correlation(topic_probs, Config.OUTPUT_DIR / "04_topic_correlation.png")
    
    # 12. Words per topic
    top_words_per_topic = nmf_model.get_top_words(terms, n_words=15)
    
    print(f"\n[WORDS PER TOPIC]")
    print("="*60)
    topic_words_list = []
    for t, words in enumerate(top_words_per_topic, 1):
        print(f"Topic {t}: {', '.join(words[:10])}")
        if len(words) > 10:
            print(f"          {', '.join(words[10:15])}")
        for rank, word in enumerate(words, 1):
            topic_words_list.append({'topic': t, 'rank': rank, 'word': word})
    
    df_topic_words = pd.DataFrame(topic_words_list)
    df_topic_words.to_excel(Config.OUTPUT_DIR / "05_topic_words.xlsx", index=False)
    
    # 13. Topic prioritization
    df_importance = compute_topic_importance(nmf_model, terms, topic_counts)
    
    print(f"\n[TOPIC PRIORITIZATION]")
    print("="*60)
    for _, row in df_importance.iterrows():
        print(f"Rank {int(row['rank'])} | Topic {int(row['topic'])} | Score: {row['total_score']:.1f} | Priority: {row['priority']}")
        print(f"      Documents: {int(row['documents'])} ({row['percentage']:.1f}%)")
    
    df_importance.to_excel(Config.OUTPUT_DIR / "06_topic_importance.xlsx", index=False)
    plot_importance_scores(df_importance, Config.OUTPUT_DIR / "06_topic_importance_plot.png")

    # 13.5 Bayesian posterior uncertainty and ARD diagnostics
    print(f"\n[BAYESIAN UNCERTAINTY] Estimating the BNMF posterior...")
    df_topic_uncertainty, df_word_uncertainty, df_doc_uncertainty = export_bayesian_posterior(
        nmf_model,
        terms,
        df,
        Config.OUTPUT_DIR / "06_bayesian_posterior_uncertainty.xlsx"
    )
    
    # 14. Dataset for machine learning
    ml_dataset = df[['title', 'year', 'journal', 'abstract', 'topic_nmf', 'topic_prob']].copy()
    ml_dataset['topic_prob_posterior_mean'] = df_doc_uncertainty['topic_prob_posterior_mean'].values
    ml_dataset['topic_prob_posterior_std'] = df_doc_uncertainty['topic_prob_posterior_std'].values
    ml_dataset['topic_prob_ci_2_5'] = df_doc_uncertainty['topic_prob_ci_2_5'].values
    ml_dataset['topic_prob_ci_97_5'] = df_doc_uncertainty['topic_prob_ci_97_5'].values
    # Add probabilities per topic
    for t in range(optimal_k):
        ml_dataset[f'topic_{t+1}_prob'] = topic_probs[:, t]
    
    ml_dataset.to_excel(Config.OUTPUT_DIR / "07_ml_ready_dataset.xlsx", index=False)
    ml_dataset.to_csv(Config.OUTPUT_DIR / "07_ml_ready_dataset.csv", index=False, encoding='utf-8')
    
    # 15. Save the complete results
    avg_corr = np.mean(np.corrcoef(topic_probs.T)[np.triu_indices(optimal_k, k=1)]) if optimal_k > 1 else 0
    results_summary = {
        'metric': ['method', 'total_documents', 'total_terms', 'optimal_k', 'active_topics_ard',
                   'gamma_prior_w_shape_rate', 'gamma_prior_h_shape_rate',
                   'posterior_samples', 'convergence_iterations',
                   'avg_reconstruction_error', 'final_log_posterior', 'avg_topic_correlation'],
        'value': ['BATH-SEP-BNMF', len(df), len(terms), optimal_k,
                  int((df_topic_uncertainty['ard_status'] == 'active').sum()),
                  f"{Config.BAYESIAN_SHAPE_W}/{Config.BAYESIAN_RATE_W}",
                  f"{Config.BAYESIAN_SHAPE_H}/{Config.BAYESIAN_RATE_H}",
                  Config.BAYESIAN_POSTERIOR_SAMPLES,
                  len(nmf_model.reconstruction_error_),
                  np.mean(nmf_model.reconstruction_error_[-10:]) if len(nmf_model.reconstruction_error_) >= 10 else np.mean(nmf_model.reconstruction_error_),
                  nmf_model.log_posterior_[-1] if nmf_model.log_posterior_ else np.nan,
                  avg_corr]
    }
    df_summary = pd.DataFrame(results_summary)
    df_summary.to_excel(Config.OUTPUT_DIR / "08_results_summary.xlsx", index=False)
    
    # 16. Save word clouds per topic
    print(f"\n[WORD CLOUDS PER TOPIC] Drawing individual word clouds...")
    
    # Create the dictionary of documents per topic
    docs_by_topic = {}
    for t in range(1, optimal_k + 1):
        docs_by_topic[t] = df[df['topic_nmf'] == t]['abstract'].tolist()
    
    stemmer = PorterStemmer()
    for topic_id, docs in docs_by_topic.items():
        if not docs:
            continue
        
        # Collect the words of this topic
        all_text = ' '.join(docs)
        processed = preprocess_text(all_text, stop_words, stemmer)
        
        # Count frequencies
        words = processed.split()
        if words:
            freq_dict = dict(Counter(words).most_common(100))
            
            if freq_dict:
                # Use the publication style for topic word clouds
                wc = WordCloud(
                    stopwords=set(STOPWORDS),
                    background_color='white',
                    width=1200, height=800,
                    max_words=100,
                    colormap='viridis',
                    random_state=42
                )
                wc.generate_from_frequencies(freq_dict)
                
                plt.figure(figsize=(14, 10))
                plt.imshow(wc, interpolation='bilinear')
                plt.axis('off')
                plt.title(f'Topic {topic_id} - Top Words', fontsize=16, pad=20)
                plt.tight_layout()
                plt.savefig(Config.WORDCLOUD_DIR / f'topic_{topic_id}_wordcloud.png', dpi=150, bbox_inches='tight')
                plt.close()
    
    print(f"\n   ✅ {len([d for d in docs_by_topic.values() if d])} topic word clouds saved")
    
    # ========================================================================
    # FINAL SUMMARY
    # ========================================================================
    print("\n" + "="*70)
    print("  EXECUTIVE SUMMARY BATH-SEP-BNMF")
    print("="*70)
    
    print(f"\n📌 ANALYSIS SETTINGS:")
    print(f"   • Documents analysed: {len(df)}")
    print(f"   • Unique terms in the DTM: {len(terms)}")
    print(f"   • Stop words utilizadas: {len(stop_words)}")
    print(f"   • Optimal K selected: {optimal_k}")
    
    print(f"\n📌 TOP 3 TOPICS MAIS RELEVANTES:")
    for i in range(min(3, len(df_importance))):
        row = df_importance.iloc[i]
        top_words = top_words_per_topic[int(row['topic'])-1][:7]
        print(f"\n   🥇 RANK {int(row['rank'])} - TOPIC {int(row['topic'])} (Score: {row['total_score']:.1f}) - Priority {row['priority']}")
        print(f"      • {int(row['documents'])} documents ({row['percentage']:.1f}% of the total)")
        print(f"      • Keywords: {', '.join(top_words)}")
        print(f"      • Scores: Freq={row['freq_score']} | Excl={row['exclusivity_score']} | Coh={row['coherence_score']} | Dist={row['distinction_score']}")
    
    print(f"\n📁 FILES WRITTEN:")
    print(f"   📝 FOR MANUAL REVIEW:")
    print(f"   • {Config.OUTPUT_DIR}/00_words_for_review.xlsx - review and mark words")
    print(f"   • {Config.OUTPUT_DIR}/00_word_frequency.xlsx - complete frequency list")
    print(f"\n   🖼️ WORD CLOUDS (PUBLICATION-READY):")
    print(f"   • {Config.WORDCLOUD_DIR}/01_wordcloud_publication.png - standard word cloud (1600x800)")
    print(f"   • {Config.WORDCLOUD_DIR}/01_wordcloud_round.png - round word cloud (2400x2400, 300 DPI)")
    print(f"   • {Config.WORDCLOUD_DIR}/topic_*_wordcloud.png - word clouds per topic")
    print(f"\n   BNMF ANALYSIS:")
    print(f"   • {Config.OUTPUT_DIR}/02_k_selection_metrics.xlsx")
    print(f"   • {Config.OUTPUT_DIR}/02_k_selection_plot.png")
    print(f"   • {Config.OUTPUT_DIR}/03_topic_distribution.png")
    print(f"   • {Config.OUTPUT_DIR}/04_topic_correlation.png")
    print(f"   • {Config.OUTPUT_DIR}/05_topic_words.xlsx")
    print(f"   • {Config.OUTPUT_DIR}/06_topic_importance.xlsx")
    print(f"   • {Config.OUTPUT_DIR}/06_topic_importance_plot.png")
    print(f"   • {Config.OUTPUT_DIR}/07_ml_ready_dataset.csv/.xlsx")
    print(f"   • {Config.OUTPUT_DIR}/08_results_summary.xlsx")
    
    print("\n" + "="*70)
    print("  🏁 ANALYSIS COMPLETED SUCCESSFULLY!")
    print("="*70)
    
    print("\n📌 NEXT STEPS:")
    print("   1. Open the file '00_words_for_review.xlsx'")
    print("   2. Review the words and mark in the 'remove' column those to be removed")
    print("   3. Mark in the 'keep' column the words to be kept")
    print("   4. Atualize as listas stop_words e preserve_words no script")
    print("   5. Run again to refine the results")
    print("\n📌 ABOUT THE WORD CLOUDS:")
    print("   - Rectangular format (1600x800): recommended for articles and reports")
    print("   - Round format (2400x2400, 300 DPI): for covers, posters and presentations")
    print("="*70 + "\n")
    
    return {
        'df': df,
        'nmf_model': nmf_model,
        'optimal_k': optimal_k,
        'topic_importance': df_importance,
        'topic_words': top_words_per_topic
    }

# ============================================================================
# EXECUTION
# ============================================================================
if __name__ == "__main__":
    try:
        results = main()
    except Exception as e:
        print(f"\n❌ FATAL ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
