import json
import csv

def convert_smiles_json_to_csv(json_path='smiles.json', csv_path='smiles.csv'):
    """
    Read a JSON file containing ligand_id to SMILES mapping,
    and convert it to a CSV file with the specified format.

    Args:
        json_path (str): Path to the input JSON file.
        csv_path (str): Path to the output CSV file.
    """
    import os
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            smiles_dict = json.load(f)

        with open(csv_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            # Write header
            writer.writerow(['smiles', 'ligand_id'])
            # Write data
            for ligand_id, smiles in smiles_dict.items():
                writer.writerow([smiles, ligand_id])

        print(f"Successfully converted '{json_path}' to '{csv_path}'.")

    except FileNotFoundError:
        print(f"Error: File '{json_path}' not found. Please make sure the file exists at the correct path.")
    except json.JSONDecodeError:
        print(f"Error: File '{json_path}' is not a valid JSON file.")
    except Exception as e:
        print(f"An unknown error occurred while processing the file: {e}")


def create_data_csv_from_ids(id_json_path='id.json', smiles_json_path='smiles.json', output_csv_path='data.csv'):
    """
    Read a JSON list containing {prot_id}_{ligand_id} strings,
    and use a SMILES mapping JSON file to create a CSV file with Smiles, prot_id, and ligand_id.

    Args:
        id_json_path (str): Path to the input JSON file containing ID list.
        smiles_json_path (str): Path to the JSON file containing ligand_id to SMILES mapping.
        output_csv_path (str): Path to the output CSV file.
    """
    import os
    os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)
    try:
        # Load ligand_id to smiles mapping
        with open(smiles_json_path, 'r', encoding='utf-8') as f:
            smiles_map = json.load(f)

        # Load list of prot_id and ligand_id combinations
        with open(id_json_path, 'r', encoding='utf-8') as f:
            id_list = json.load(f)

        with open(output_csv_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            # Write header
            writer.writerow(['Smiles', 'prot_id', 'ligand_id'])

            # Process and write data
            missing_ligands = []
            for item in id_list:
                try:
                    prot_id, ligand_id = item.rsplit('_', 1)
                    smiles = smiles_map.get(ligand_id)

                    if smiles:
                        writer.writerow([smiles, prot_id, ligand_id])
                    else:
                        # Record ligand_id not found in smiles.json
                        missing_ligands.append(ligand_id)

                except ValueError:
                    print(f"Warning: Skipping incorrectly formatted entry '{item}'.")
            
            if missing_ligands:
                print(f"\nWarning: The following {len(missing_ligands)} ligand_id(s) were not found in '{smiles_json_path}':")
                # Only print first 10 missing IDs to avoid overly long output
                print(", ".join(missing_ligands[:10]) + ('...' if len(missing_ligands) > 10 else ''))


        print(f"\nSuccessfully generated '{output_csv_path}'.")


    except FileNotFoundError as e:
        print(f"Error: File '{e.filename}' not found. Please make sure the file exists at the correct path.")
    except json.JSONDecodeError as e:
        print(f"Error: Invalid file format. Please check if '{e.doc}' is a valid JSON file.")
    except Exception as e:
        print(f"An unknown error occurred while processing the file: {e}")


if __name__ == '__main__':
    # --- Task 1 ---
    # Convert smiles.json to smiles.csv
    SMILES_JSON_PATH = './data/mf-pcba/smiles.json'
    SMILES_CSV_PATH = './FABind_plus/mf-pcba/smiles.csv'
    ID_JSON_PATH = './data/mf-pcba/id.json'
    DATA_CSV_PATH = './FABind_plus/mf-pcba/data.csv'
    print("--- Starting Task 1: Convert smiles.json to smiles.csv ---")
    convert_smiles_json_to_csv(SMILES_JSON_PATH, SMILES_CSV_PATH)
    print("-" * 50)

    # --- Task 2 ---
    # Generate data.csv using id.json and smiles.json
    # Note: Before running this task, ensure smiles.json contains all ligand_ids needed in id.json
    print("\n--- Starting Task 2: Generate data.csv ---")
    create_data_csv_from_ids(ID_JSON_PATH, SMILES_JSON_PATH, DATA_CSV_PATH)