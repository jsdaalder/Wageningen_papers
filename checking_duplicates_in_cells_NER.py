import pandas as pd
import spacy
import re
from langdetect import detect
import langdetect

# Load spacy models
nlp_en = spacy.load("en_core_web_md")
nlp_nl = spacy.load("nl_core_news_md")

# Load the CSV file
df = pd.read_csv('Wageningen-1995-2023-papers-export.csv')

# Define necessary columns
necessary_columns = ['Lens ID', 'Title', 'Funding']

# Function to extract organizations
def extract_orgs(funding_info):
    orgs = []
    if isinstance(funding_info, str) and funding_info.strip():
        # split by commas, semicolons or pipes
        split_info = re.split(',|;|\|', funding_info)

        for info in split_info:
            try:
                lang = detect(info)
            except langdetect.lang_detect_exception.LangDetectException:
                lang = None

            if lang == 'nl':
                nlp = nlp_nl
            else:
                nlp = nlp_en

            doc = nlp(info)
            extracted_orgs = [ent.text for ent in doc.ents if ent.label_ == 'ORG']
            for org in extracted_orgs:
                org_name = org.split(' (')[0] if '(' in org else org  # Extract organization name before parentheses
                orgs.append(org_name.strip())
    return orgs if orgs else None

# Extract organizations and assign them to the 'Funding_NER' column
df['Funding_NER'] = df['Funding'].apply(extract_orgs)

print(df.sample(20))

# Explode the 'Funding_NER' column
# df = df.explode('Funding_NER')

exploded = df[["Lens ID","Funding_NER"]].set_index("Lens ID")["Funding_NER"].explode().reset_index().drop_duplicates()

merged = pd.merge(left=df, right=exploded, on="Lens ID")

# Save the updated DataFrame to a new CSV file
merged.to_csv('Wageningen-1995-2023-papers-export_NER.csv', index=False)

print('Succesfully exported to Wageningen-1995-2023-papers-export_NER.csv')

# Create a new DataFrame with only the necessary columns and save it to a new CSV file
result_df = merged[['Lens ID', 'Title', 'Funding', 'Funding_NER']]
result_df.to_csv('Wageningen-papers-export_filtered_NER.csv', index=False)

print('Succesfully exported to Wageningen-papers-export_filtered_NER.csv')

