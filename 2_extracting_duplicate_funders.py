import pandas as pd
import dedupe
from unidecode import unidecode
import os
import pickle
import time
import sys
import csv

print(sys.executable)

# Load your data
print('Loading data...')
start = time.time()
data_df = pd.read_excel('all_papers_exploded.xlsx')

# We only need 'Funding_split' column, and count the occurrences
data = data_df['Funding_split'].value_counts().reset_index().values.tolist()
end = time.time()
print(f'Data loaded. Time taken: {end-start} seconds')

# Prepare the data for dedupe
data_dict = {idx: {"name": unidecode(str(name)), "count": int(count)} for idx, (name, count) in enumerate(data)}

# Define the fields dedupe will pay attention to
fields = [
    {'field' : 'name', 'type': 'String'},
    {'field' : 'count', 'type': 'Price', 'has missing' : True}
]

# Create a new deduper object and pass our data model to it
deduper = dedupe.Dedupe(fields, num_cores=1, in_memory=False)

# File name where the training data will be saved
training_file = 'training_data.pkl'

# Training the data
if os.path.exists(training_file):
    print('Training file found.')
    use_existing = input('Do you want to use previously prepared training data? (y/n)\n')
    if use_existing.lower() == 'y':
        with open(training_file, 'rb') as tf:
            deduper = pickle.load(tf)
        print('Training data loaded from file.')
    else:
        start_prepare = time.time()
        print('Preparing training data...')
        deduper.prepare_training(data_dict)
        end_prepare = time.time()
        print('Training data prepared. Time taken:', end_prepare - start_prepare, 'seconds.')
else:
    start_prepare = time.time()
    print('Preparing training data...')
    deduper.prepare_training(data_dict)
    end_prepare = time.time()
    print('Training data prepared. Time taken:', end_prepare - start_prepare, 'seconds.')

# Manually labeling the data
print('Please manually label some data...')
import random
suggestions = deduper.uncertain_pairs()

counts = {'y': 0, 'n': 0, 'skip': 0}
break_labeling = False

while suggestions and not break_labeling:
    for pair in suggestions:
        print(f'Record 1: {pair[0]}')
        print(f'Record 2: {pair[1]}')

        valid_input = False
        while not valid_input:
            is_match = input("Are these records a match? (y/n/skip/stop/finished) ")
            if is_match.lower() == 'y':
                counts['y'] += 1
                deduper.mark_pairs({'match': [pair], 'distinct': []})
                valid_input = True
            elif is_match.lower() == 'n':
                counts['n'] += 1
                deduper.mark_pairs({'match': [], 'distinct': [pair]})
                valid_input = True
            elif is_match.lower() == 'skip':
                counts['skip'] += 1
                valid_input = True
            elif is_match.lower() == 'stop':
                sys.exit()  # This will stop the script execution
            elif is_match.lower() == 'finished':
                break_labeling = True
                valid_input = True
            else:
                print("Invalid input. Please enter either 'y', 'n', 'skip', 'stop' or 'finished'.")

        if break_labeling:
            break

        suggestions = list(deduper.uncertain_pairs())
        print(f"Remaining uncertain pairs: {len(suggestions)}")
        print(f"Yes: {counts['y']}")
        print(f"No: {counts['n']}")
        print(f"Skip: {counts['skip']}")

end_prepare = time.time() # End the timer
print(f'Preparing training data completed in {end_prepare-start_prepare} seconds') # Print the time difference

# Create file for trained data and another for the trained model
start_prepare = time.time()
print('Creating file for trained data and another for the trained model')
with open(training_file, 'wb') as tf:
    pickle.dump(deduper, tf)

end_prepare = time.time()
print(f'Created file for trained data and another for the trained model in {end_prepare-start_prepare} seconds')

# Training the model
print('Training model...')
deduper.train()

# Save our weights and predicates to disk
with open('dedupe_model', 'wb') as f:
    deduper.write_settings(f)

# We are done with training data
deduper.cleanup_training()
print('Training data has been cleaned up')

# Find the threshold that will maximize a weighted average of our precision and recall
# print('Finding threshold')
# threshold = deduper.good_threshold(data_dict, recall_weight=1)
# print(f'Threshold found: {threshold}')

try:
    print('Clustering data')
    clustered_dupes = deduper.partition(data_dict, 0.1)
    
    # Create a dictionary where keys are original record indices and values are cluster IDs
    cluster_dict = {}
    for cluster_id, (records, scores) in enumerate(clustered_dupes):
        for record in records:
            cluster_dict[record] = cluster_id

    # Initialize a dictionary to keep track of combined data for each cluster
    combined_data = {}
    for idx, data in data_dict.items():
        cluster_id = cluster_dict[idx]
        if cluster_id in combined_data:
            # If the cluster ID already exists in combined_data, add the count to the existing count
            combined_data[cluster_id]['count'] += data['count']
        else:
            # Otherwise, create a new entry for this cluster
            combined_data[cluster_id] = {'name': data['name'], 'count': data['count']}

    # Write the combined data to a new CSV file
    print('Writing results to file...')
    with open('funders_and_counts_extended_deduped.csv', 'w') as f:
        writer = csv.DictWriter(f, fieldnames=['Funder', 'Number of Papers Funded'])
        writer.writeheader()
        for data in combined_data.values():
            writer.writerow({'Funder': data['name'], 'Number of Papers Funded': data['count']})

    print('Results have been written to file')
    
except Exception as e:
    print("Caught an exception during clustering and writing to CSV:")
    print(e)
