# GROUP-7-MH6301-INFORMATION-RETRIEVAL-ANALYSIS-Group-Project

MH6301 Information Retrieval & Analysis - Group Project
Group 7: Review Data Analysis and Processing

1. Project Overview
-------------------
This project implements the MH6301 group assignment on Yelp review data. It covers:

1) Dataset analysis for one selected metropolitan area: Philadelphia.
2) A Yelp business search engine supporting:
   - keyword search over business names/categories;
   - similar-business search based on review similarity using BM25;
   - similar-business search based on dense sentence embeddings;
   - hybrid BM25 + embedding search;
   - incremental add/update simulation for online review data.
3) FAQ generation based on customer reviews for one selected business.
4) A small NLP/IR application that detects comparison sentences in Yelp reviews.

The submitted SourceCode folder should contain only our own source code. The Yelp dataset, generated indexes, model caches, third-party libraries, and downloaded resources should not be included in the zip submission.


2. Folder Structure
-------------------
Recommended source submission structure:

Src_G07.zip
|-- Readme.txt
`-- SourceCode/
    |-- 3.2. Dataset Analysis.ipynb
    |-- 3.3. Development of a Search Engine.ipynb
    |-- 3.4. FAQ Generation based on Reviews.ipynb
    |-- data_loader.py
    `-- compare_detector.py

Optional runtime folders created locally after execution:

|-- dataset/                         # not submitted; stores Yelp JSON files if using CLI scripts
|-- philly_index/                    # not submitted; Whoosh index generated at runtime
|-- philly_embeddings.pkl            # not submitted; embedding cache generated at runtime
|-- faq_generation_results.csv       # not submitted unless requested as sample output
`-- sample_output/                   # not submitted unless requested as sample output


3. Third-Party Libraries and Download Links
-------------------------------------------
Python 3.10 or above is recommended.

Required Python packages:

- pandas: https://pandas.pydata.org/
- numpy: https://numpy.org/
- nltk: https://www.nltk.org/
- whoosh: https://pypi.org/project/Whoosh/
- sentence-transformers: https://www.sbert.net/
- scikit-learn: https://scikit-learn.org/
- tqdm: https://tqdm.github.io/
- kagglehub: https://pypi.org/project/kagglehub/
- symspellpy: https://pypi.org/project/symspellpy/
- notebook or jupyterlab: https://jupyter.org/

The search engine uses Whoosh as the main IR library. Sentence-Transformers and scikit-learn are used for semantic review similarity. NLTK is used for tokenization, stopword removal, stemming, POS tagging, and sentence splitting.


4. Installation Guide
---------------------
Create and activate a virtual environment:

macOS/Linux:
    python3 -m venv .venv
    source .venv/bin/activate

Windows PowerShell:
    python -m venv .venv
    .venv\Scripts\Activate.ps1

Install dependencies:

    pip install pandas numpy nltk whoosh sentence-transformers scikit-learn tqdm kagglehub symspellpy notebook

Download required NLTK resources:

    python -c "import nltk; nltk.download('punkt'); nltk.download('punkt_tab'); nltk.download('stopwords'); nltk.download('averaged_perceptron_tagger_eng')"


5. Dataset Setup
----------------
The assignment dataset is the Yelp Open Dataset. Only business and review data are required.

Notebook workflow:
    The notebooks use kagglehub to download a CSV-packaged version of Yelp business and review data:

        kagglehub.dataset_download("waterwaterf/yelp-review-business")

    Running the first cells of the notebooks will download and load the data automatically.

CLI workflow for data_loader.py and compare_detector.py:
    Place the official Yelp Open Dataset JSON files in a dataset folder located one level above SourceCode:

        dataset/yelp_academic_dataset_business.json
        dataset/yelp_academic_dataset_review.json

    Example structure:

        ProjectRoot/
        |-- dataset/
        |   |-- yelp_academic_dataset_business.json
        |   `-- yelp_academic_dataset_review.json
        `-- SourceCode/
            |-- data_loader.py
            `-- compare_detector.py

Do not include the dataset in the final source-code zip.


6. How to Run the System
------------------------
6.1 Dataset Analysis

Open and run:

    jupyter notebook "3.2. Dataset Analysis.ipynb"

This notebook:
- loads the Yelp business and review datasets;
- filters businesses and reviews to Philadelphia;
- reports basic sampled-dataset statistics;
- selects Reading Terminal Market as the business for detailed review analysis;
- compares no stemming, Porter stemming, and Snowball stemming;
- discusses review writing style compared with formal news writing.

Expected sample output:

    Number of businesses in Philadelphia: 14569
    Number of reviews for Philadelphia businesses: 967552
    Extracting reviews for: Reading Terminal Market (ID: ytynqOUb3hjKeJfRj5Tshw)
    Total reviews for Reading Terminal Market: 5778

The stemming output shows that without stemming, words such as "place" and "places" are counted separately. Porter and Snowball stemming merge related forms and increase the count of root concepts such as place, food, and market.


6.2 Search Engine

Open and run:

    jupyter notebook "3.3. Development of a Search Engine.ipynb"

The notebook creates:
- philly_index/ as the Whoosh BM25F index;
- philly_embeddings.pkl as the Sentence-Transformer embedding cache.

The search menu supports:

    1. Keyword search (BM25F + fuzzy)
    2. Similar reviews - BM25 keywords
    3. Similar reviews - Embeddings
    4. Similar reviews - Hybrid
    5. Add/update business (online simulation)
    6. Exit

Example keyword-search call inside the notebook:

    keyword_search("Italian restaurant", top_n=5)

Expected sample output format:

    BM25 KEYWORD SEARCH
    Query   : (italian OR pasta OR pizza OR trattoria OR osteria) (restaurant OR eatery OR dining OR bistro OR grill)
    Hits    : 73
    Time    : 13.5 ms

    Rank 1 | Score: 32.5965 | DocID: V0DE3adlCc6m3t-yTxAg4g
    Business : Venice Pizza and Italian Eatery
    City     : Philadelphia
    Snippet  : Venice Pizza and Italian Eatery

Example typo-tolerant query:

    keyword_search("coffe", top_n=5)

Expected behavior:

    Spell check: 'coffe' -> 'coffee'
    Rank 1: Espresso Cafe & Sushi Bar

Example similar-review query:

    sample_id = "MTSW4McQd7CbVtyjqoe9mw"   # St Honore Pastries
    bm25_similar(sample_id, top_n=5)
    embedding_similar(sample_id, top_n=5)
    hybrid_similar(sample_id, top_n=5, alpha=0.4)

Expected behavior:
- BM25 returns businesses with overlapping review vocabulary such as bakery, buns, tea, pastries, and Chinatown.
- Embedding search returns semantically similar bakery/dessert businesses even when exact words differ.
- Hybrid search combines both effects.

Example online update simulation:

    add_or_update_business(
        business_id="TEST_BIZ_001",
        name="Test Ramen House",
        city="Philadelphia",
        categories="Ramen, Japanese, Noodles",
        new_reviews="Amazing tonkotsu broth. Very cosy atmosphere. Great service.",
        stars=4.5,
        review_count=1,
    )
    keyword_search("Test Ramen House", top_n=3)

Expected sample output:

    Adding new business: Test Ramen House
    Index and embeddings updated for business_id=TEST_BIZ_001
    Rank 1 | DocID: TEST_BIZ_001 | Business: Test Ramen House


6.3 FAQ Generation

Open and run:

    jupyter notebook "3.4. FAQ Generation based on Reviews.ipynb"

This notebook:
- filters to Philadelphia;
- selects a business with more than 100 reviews;
- selects John's Roast Pork with 1,609 reviews;
- splits reviews into sentences;
- matches sentences to review aspects using keyword dictionaries;
- generates 5 to 10 FAQ items with supporting evidence sentences;
- saves faq_generation_results.csv.

Expected sample output:

    Selected business for FAQ generation:
    Business ID: LM54ufrINJWoTN5imV8Etw
    Business Name: John's Roast Pork
    Actual Reviews in Philadelphia Dataset: 1609

Generated FAQ aspects and matched sentence counts:

    food quality: 1173
    waiting time: 911
    location and parking: 509
    portion size: 492
    service: 336
    price and value: 316
    menu options: 171
    ambience: 68

Example FAQ output:

    Question: How do customers describe the food quality?
    Answer: Food quality is mentioned in 1173 review sentences. The average rating of the reviews mentioning this aspect is 4.23. Most related comments appear in positive reviews.
    Evidence: Now this is a cheese steak with FLAVOR.


6.4 Comparison-Sentence Detector Application

The CLI application is implemented in compare_detector.py and uses data_loader.py.

From the SourceCode folder, run a quick smoke test:

    python data_loader.py stats
    python data_loader.py sample-reviews --n 5

Run the comparison detector:

    python compare_detector.py --top 20

Fast development mode:

    python compare_detector.py --scan-cap 50000 --top 20

Restrict to one business:

    python compare_detector.py --business-id LM54ufrINJWoTN5imV8Etw --top 20

Save output to file:

    python compare_detector.py --top 50 --output ../sample_output/sample_top50.txt

Parameters:

    --top N             Number of top-scoring comparison sentences to return.
    --business-id ID    Optional Yelp business_id filter.
    --scan-cap N        Stop after N sentences; useful for quick testing.
    --review-cap N      Stop after N reviews; useful for quick testing.
    --output PATH       Save output to a text file.
    --quiet             Suppress progress logs.

Expected sample output format:

    # Section 3.5 Application - Comparison-Sentence Detector
    # Dataset: Yelp Open Dataset, Philadelphia subset.
    # Reviews scanned: ...
    # Sentences scanned: ...
    # Sentences detected as comparison: ...
    # Returned: top-20 by cue-diversity score

    [1] score=3 stars=5 business_id=... cues=pos_jjr,multiword,than_with_comparative target="..."
        "This place is better than the other bakery nearby."

The detector flags sentences using comparative POS tags such as JJR/RBR, multi-word lexical cues such as "compared to" and "instead of", and the "than" construction when accompanied by comparative language.


7. Input Format
---------------
Business data should include at least:

    business_id, business_name or name, city, state, latitude, longitude,
    business_avg_stars or stars, business_review_count or review_count

Review data should include at least:

    review_id, user_id, business_id, stars, text, date

For official Yelp JSON files, data_loader.py expects line-delimited JSON.
For the notebooks, the KaggleHub CSV version is read using pandas.read_csv.


8. Explanation of Main Output Files
-----------------------------------
philly_index/
    Whoosh index for Philadelphia business documents. Each document represents one business.

philly_embeddings.pkl
    Pickle cache containing the Sentence-Transformer embedding matrix and corresponding business IDs.

faq_generation_results.csv
    Generated FAQ table containing aspect, question, answer, matched sentence count, and evidence sentences.

sample_output/sample_top50.txt
    Optional saved output from compare_detector.py containing top comparison sentences and run statistics.


9. Notes on Reproducibility and Scalability
-------------------------------------------
- Set random_state=42 in FAQ generation to reproduce the selected business.
- The Whoosh index and embedding cache are generated locally and should not be submitted.
- The current implementation can simulate incremental updates by calling add_or_update_business().
- For high-volume online review streams, the design should be extended with batch commits, a queue such as Kafka, and a background indexing worker.
- For large-scale FAQ generation, aspect matching can be run business-by-business and parallelized. Quality can be evaluated by human relevance judgments, evidence support, redundancy, and answer faithfulness.

