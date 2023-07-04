import pandas as pd
import spacy
from langdetect import detect
import langdetect
import string

def sanitize(data):
    return ''.join(filter(lambda x: x in string.printable, data))

# Load spacy models
nlp_en = spacy.load("en_core_web_lg")
nlp_nl = spacy.load("nl_core_news_lg")

# Load the CSV file
df = pd.read_csv('Wageningen-1995-2023-papers-export.csv')

# Sanitize all string columns in the DataFrame
for col in df.columns:
    if df[col].dtype == object:  # if the column is of string type
        df[col] = df[col].apply(lambda x: sanitize(x) if isinstance(x, str) else x)
        
# Define necessary columns
necessary_columns = ['Lens ID', 'Title', 'Funding']

# Function to extract organizations
def extract_orgs(funding_info):
    orgs = []
    if isinstance(funding_info, str) and funding_info.strip():
        try:
            lang = detect(funding_info)
        except langdetect.lang_detect_exception.LangDetectException:
            lang = None

        if lang == 'nl':
            nlp = nlp_nl
        else:
            nlp = nlp_en

        doc = nlp(funding_info)
        extracted_orgs = [ent.text for ent in doc.ents if ent.label_ == 'ORG']
        for org in extracted_orgs:
            org_name = org.split(' (')[0] if '(' in org else org  # Extract organization name before parentheses
            orgs.append(org_name.strip())
    return orgs if orgs else None

# Extract organizations and assign them to the 'Funding_NER' column
df['Funding_NER'] = df['Funding'].apply(extract_orgs)

# Dataframe where 'Funding_NER' is null
df_null = df[df['Funding_NER'].isnull()]

df_null.to_excel("papers_with_no_funders.xlsx", index=False)

# Remaining rows where 'Funding_NER' is not null
exploded = df[df['Funding_NER'].notnull()].set_index("Lens ID")["Funding_NER"].explode().reset_index().drop_duplicates().rename(columns={"Funding_NER" : "Funding_split"})

# All original columns from the original df are included in merged
merged = pd.merge(left=df, right=exploded, on="Lens ID")

merged.to_excel("papers_with_multiple_funders_exploded.xlsx", index=False)

# Dataframe with all rows regardless of the 'Funding_NER' status
exploded_all = df.set_index("Lens ID")["Funding_NER"].explode().reset_index().drop_duplicates().rename(columns={"Funding_NER" : "Funding_split"})

merged_all = pd.merge(left=df, right=exploded_all, on="Lens ID")

merged_all.to_excel("all_papers_exploded.xlsx", index=False)
