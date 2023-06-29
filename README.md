# Wageningen_papers

These are three parts of code for an investigation in the links between the agro-industry and Wageningen University & Research (WUR).

The first step was to download all [~60.000 papers created by WUR in the last 25 years through Lens.org]([url](https://www.lens.org/lens/search/scholar/table?q=author.affiliation.name:(%22Wageningen%20University%22)&p=0&n=50&s=date_published&d=%2B&f=false&e=false&l=en&authorField=author&dateFilterField=publishedYear&orderBy=%2Bdate_published&presentation=false&preview=true&stemmed=true&useAuthorId=false&publishedYear.from=1998&publishedYear.to=2023)https://www.lens.org/lens/search/scholar/table?q=author.affiliation.name:(%22Wageningen%20University%22)&p=0&n=50&s=date_published&d=%2B&f=false&e=false&l=en&authorField=author&dateFilterField=publishedYear&orderBy=%2Bdate_published&presentation=false&preview=true&stemmed=true&useAuthorId=false&publishedYear.from=1998&publishedYear.to=2023) 

The data (a 133.1 MB csv) is very dirty however. To clean it up, we first run **checking_duplicates_in_cells_NER.py**. It uses Named-entity recognition (with the spacy library), for both Dutch and English. There's still an unfixed bug with papers that have multiple funders (a new row for every funder should be created, but for some reason a new column is created). 

Then comes the next step: there's a lot of duplicates. Take for example these organisations:
1. Dutch ministry of LNV
2. The Dutch ministry of Agriculture
3. Ministerie van Landbouw
4. Ministerie van Landbouw, Natuur & Voedselkwaliteit
5. etc.

To combine all these, we run **matching_funders_v2.py**. Main ingredient: the Dedupe library, which uses a whiff of machine learning and some manual labeling.  

The last step is to do the analysis. First question: is the WUR funded mostly by governments, ngo’s or companies? For this we run the code in count_funders.py. It uses the OpenAI API to guess what kind of organisation we are dealing with. 

Some other questions, for which I haven't written any code:
1. Did the share of business-funded research increase over the yers?
2. What kind of  research do big agro-companies fund?
3. Which scientists received the most funding from agro-companies?
