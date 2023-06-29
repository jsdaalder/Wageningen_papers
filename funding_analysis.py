import pandas as pd
from collections import Counter
import openai
import os

# Set up the OpenAI API
openai.api_key = '[insert API-key here]'

# Read the csv file
df = pd.read_csv('wageningen_papers_extended_version.csv')

# Create an empty Counter object to hold the counts
counts = Counter()

# Iterate over the 'Funding' column
for funding in df['Funding']:
    # Skip rows where 'Funding' is NaN
    if pd.isna(funding):
        continue
    # Split the string on semicolon and update the counts
    for org in funding.split(';'):
        counts[org.strip()] += 1

# Create a DataFrame to hold the funders and their counts
df_counts = pd.DataFrame.from_records(list(counts.items()), columns=['Funder', 'Number of Papers Funded'])

# Add a column for the category
df_counts['Category'] = ''

# Check if a file with the same name already exists, and if it does, append a number to the filename
base_filename = 'funders_and_counts_extended'
extension = '.csv'
i = 1
filename = base_filename + extension

while os.path.exists(filename):
    filename = f"{base_filename}({i}){extension}"
    i += 1

df_counts.to_csv(filename, index=False)
