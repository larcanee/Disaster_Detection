import pandas as pd

# Read the CSV file into a DataFrame
df = pd.read_csv('categories.csv')

# Replace "other_relevant_information" with 1 in a specific column (replace 'column_name' with the actual column name)
df['target'] = df['target'].replace('rescue_volunteering_or_donation_effort', 1)
df['target'] = df['target'].replace('other_relevant_information', 2)
df['target'] = df['target'].replace('infrastructure_and_utility_damage', 3)
df['target'] = df['target'].replace('sympathy_and_support', 4)
df['target'] = df['target'].replace('injured_or_dead_people', 5)
df['target'] = df['target'].replace('caution_and_advice', 6)
df['target'] = df['target'].replace('displaced_people_and_evacuations', 7)
df['target'] = df['target'].replace('not_humanitarian', 8)
df['target'] = df['target'].replace('requests_or_urgent_needs', 9)
df['target'] = df['target'].replace('missing_or_found_people', 10)

# Save the updated DataFrame to a new CSV file or overwrite the existing one
df.to_csv('categorynums.csv', index=False)
