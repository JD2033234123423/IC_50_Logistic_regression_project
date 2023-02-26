#!/usr/bin/env python3


# Import libraries
import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from tqdm import tqdm
from sys import argv
from chembl_webresource_client.new_client import new_client

# Showing all dataframe rows
pd.set_option('display.max_rows', None)

# Quick instructions on the program
input("""\t\nWelcome, this program is used to estimate the IC50 of a compound based on it's smiles notation.\nThis program works by building a logistic regression model from the CHEMBL database.
      \nTo run this program, you need to search the CHEMBL database for the same biological target as the smiles query.\nThen, build the model by customising the IC50 value which will split the dataset for the model into active or inactive.
      \nFor whatever value you enter, any value above, but not including that, will be considered as inactive.
      \nOnly select a target of 'SINGLE PROTEIN' to build the model.
      \nTo continue, press ENTER: """)
# Search the CHEMBL database for biological targets
def search_query():
    
    target = new_client.target
    user_search = input("\nName the compound or biological target you want to search CHEMBL for: ")
    if user_search == '':
        return search_query()
    elif len(user_search) < 3:
        print("\nSearch too short")
        return search_query()
    try: 
        target_query = tqdm(target.search(user_search), desc="\nSearching", bar_format="{desc}:{r_bar}")
        targets = pd.DataFrame.from_dict(target_query)
        print(targets)
    except (ValueError, AttributeError) as e:
        print(f"Input Error: {str(e)}, try again")
        return search_query()

    try:
        user_choice = int(input("\nSelect the organism row ID you want to target:\nHit ENTER to search again: "))
        selected_target = targets.target_chembl_id[user_choice]
        print(selected_target)


    except (ValueError, KeyError, AttributeError):
        return search_query()
    
    activity = new_client.activity
    res = tqdm(activity.filter(target_chembl_id=selected_target).filter(standard_type="IC50"), desc="\nFiltering data", bar_format="{desc}:{r_bar}")
    df = pd.DataFrame.from_dict(res)
    
    print(df)
    
    should_continue = input("\nDo you want to continue with this selection?(y/n): ")
    
    if should_continue == 'n':
        return search_query()
    
    elif should_continue == 'y':
        pass
        
        if len(df) < 2:
            print("\nYou need at minimum two samples to train the dataset")
            return search_query()
    else:
        print('\nInvalid input')
        return search_query()
    
    df2 = df[df.standard_value.notna()]
    df2 = df2[df.canonical_smiles.notna()]
        
    df2_nr = df2.drop_duplicates(['canonical_smiles'])
    
    selection = ['canonical_smiles','standard_value']
    df3 = df2_nr[selection]
       
    data = df3
    return data

data = search_query()
data['standard_value'] = pd.to_numeric(data['standard_value'], errors='coerce')
data = data.drop(data[data.standard_value < 0].index, axis=0).reset_index(drop=True)
print(data)
print(f"\nDataset entries: {len(data)}")
IC_50_mean = data['standard_value'].mean()
IC_50_max = data['standard_value'].max()
IC_50_min = data['standard_value'].min()
IC_50_median = data['standard_value'].median()

print(f'\nDataset IC50 Mean: {IC_50_mean} nM\nDataset IC50 Min: {IC_50_min} nM\nDataset IC50 Max: {IC_50_max} nM\nDataset IC50 Median: {IC_50_median} nM')

# Defining the class for the QSAR assessment
class Compound:
    # Passing in the class name and the smiles dataset
    def __init__(self, smiles):
        self.smiles = smiles
        # Obtaining the molecule from the rdkit.Chem library
        self.mol = Chem.MolFromSmiles(smiles)
        # Generating a fingerprint from the molecule
        self.fps = AllChem.GetMorganFingerprintAsBitVect(self.mol, 3, nBits=2048)

    # Returning the fingerprint for data training    
    def get_features(self):
        return np.concatenate(([self.fps]))

    # This class is used for the prediction method for estimating IC50 from smiles string            
    def predict_activity(self, model):
        X_new = self.get_features().reshape(1, -1)
        ic50_pred = model.predict(X_new)
        label = ic50_pred[0]
        activity_status = {0: f'Greater than {cutoff_value}nM', 1: f'Less than or equal to {cutoff_value}nM'}[label]
        return activity_status


# An empty array to append the iterated smiles data in 
compounds = []
print("\n")
for smiles in tqdm(data['canonical_smiles'], desc="Making fingerprints", bar_format="{desc}:{r_bar}"):
    compound = Compound(smiles)
    compounds.append(compound)

# Assigning a new column to the dataframe
data['fps'] = [compound.fps for compound in compounds]

# Create an array for each individual fingerprint
print("\n")
X = np.array([compound.get_features() for compound in tqdm(compounds, desc="Getting features", bar_format="{desc}:{r_bar}")])


def training_model(data, X):
    # Adjust the IC50 dependent on dataset
    while True:
        try:
            cutoff_value = int(input("\nSet the value the IC50 inactive threshold in nM: "))
            if cutoff_value < 1:
                print("\nInvalid input, must be greater than 1")
                return training_model(data, X)
            y = data['standard_value'].apply(lambda x: 1 if x <= cutoff_value else 0)
            break
        except ValueError:
            print("\nPlease enter a valid integer for the IC50 inactive threshold in nM.")
            
    # Split the data into training and test sets, using a standard 80 train,20 test split
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    # Displays the model building steps
    clf = LogisticRegression(max_iter=1_000_000, verbose=1)
    clf.fit(x_train, y_train)
    y_pred = clf.predict(x_test)
    
    
    print("\nModel metrics: ")
    # Reports the overall fit of the model to the data
    print('Fit:', clf.score(x_test, y_test))
    # Compares the number of true positives to the positives predicted 
    print("Precision:", precision_score(y_test, y_pred))
    # Compares the number of positives predicted to the true number of positives
    print("Recall:", recall_score(y_test, y_pred))
    # F1 score combines precision and recall, 2* (precision*recall)/(precision + recall)
    print("F1 score:", f1_score(y_test, y_pred))
    
    # Loop to ask if the model scores are good enough or should be re trained
    while True:
        decision = input("\nWould you like to retrain the model? (y/n): ")
        if decision == 'n':
            break
        elif decision == 'y':
            while True:
                try:
                    return training_model(data, X)
                except ValueError:
                    print("\nPlease enter a valid integer for the IC50 inactive threshold in nM.")
            
    return x_test, y_test, y_pred, x_train, y_train, clf, cutoff_value

x_test, y_test, y_pred, x_train, y_train, clf, cutoff_value = training_model(data, X)

# User enters SMILES as a string and is returned with an estimated IC50 
def string_input():
    while True:
        to_smiles = input("\nWould you like to estimate IC50 for a compound? (y/n): ")
        if to_smiles == 'y':
            testing_String = input("\nWhat is the string of the query?: ")
            try: 
                compound = Compound(testing_String)
                print(f"\nPredicted activity status for:\n{compound.smiles}\nIC50 activity estimation: {compound.predict_activity(clf)}\n")
            except (ValueError, TypeError) as e:
                print(f"\nError: {str(e)}. Please check the input and try again.")
                return string_input()
        elif to_smiles == 'n':
            print("\nGoodbye\n")
            break
        else:
            print("\nInvalid answer")
            return string_input()
string_input()
