
"""
GAME PLAN
Our outputs need to be formatted as follows:
    Create a JSONL(can be done via pandas) with an object per line
    Each Object should contain the following fields:
        - id: id provided from the dataset
        - source_language: always english (en)
        - target_language: from ['ar', 'zh', 'fr', 'de', 'it', 'ja', 'ko', 'es', 'th', 'tr']
        - text: the source from the dataset
        - predicition: our predicted text

In order to do this effectively we will most likely just want to:
    1. load in the test data via a pandas call
    2. make predections on some batch of those items (in order)
    3. place our predictions into a new column
    4. output the data to a new jsonl file 
"""