import pandas as pd
import yaml

# Load the CSV file
file_path = 'missing_classifications.csv'
data = pd.read_csv(file_path)

# Convert dataframe to a list of tuples, matching the desired output structure
data_tuples = [{tuple(x): "Other transportation support activities"} for x in data.to_numpy()]

# Prepare the data in the specified YAML format with the custom tag
yaml_data = yaml.dump(data_tuples, default_flow_style=False, allow_unicode=True, sort_keys=False)

# Save the YAML data to a file
yaml_file_path = 'converted_yaml.yaml'
with open(yaml_file_path, 'w') as file:
    file.write(yaml_data)