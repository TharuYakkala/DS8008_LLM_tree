from pathlib import Path
import os
from tqdm import tqdm

def generate_dt_funcs(datasets_path, output_path, llm, num_dt=5, use_subset=None):
  data_sets = os.listdir(datasets_path)

  # sometimes you may only want to run it on a subset of all datasets
  if use_subset is not None:
    data_sets = [dataset for dataset in data_sets if dataset in use_subset]

  dt_main_path = Path(output_path)
  dt_main_path.mkdir(exist_ok=True)

  DTs_per_dataset = 5
  SYSTEM_PROMPT = f"""
    You are a domain expert with years of experience in building the best-performing decision trees.
    You have an astounding ability to identify the best features for the task at hand, and you know how to combine them to get the best predictions.
    Impressively, your profound world knowledge allows you to do that without looking at any real-world data.

    # Do the following Steps
    ### STEP 1
    I want you to induce a decision tree classifier based on features and a prediction target.
    I first give an example of the decision tree. Given Features and a new prediction target, I then want you to build a decision tree using the most important features.

    ### STEP 2
    Format the decision tree as a Python function that returns a single prediction as well as a list representing the truth values of the inner nodes.
    The entries of this list should be 1 if the condition of the corresponding inner node is satisfied, and 0 otherwise.
    Use only the feature names that I provide, generate the decision tree without training it on actual data, and return the Python function.
    """

  for dataset in tqdm(data_sets, desc="Generating DTs for each dataset"):
    # Read dataset description
    with open(os.path.join(datasets_path, dataset, 'description.txt')) as f:
      description_part = f.read()

    # Read specific prompt for dataset
    with open(os.path.join(datasets_path, dataset, 'prompt.txt')) as f:
      final_part = f.read()



    current_prompt = f"""
    ### SYSTEM
    <system>
    {SYSTEM_PROMPT}
    </system>

    ### DESCRIPTION OF DATASET
    <description>
    {description_part}
    </description>

    ### USER
    <user>
    {final_part}
    </user>

    ### OUTPUT INSTRUCTION
    Only return the decision tree function. Do not add any reasoning or other materials.
    """

    for i in range(DTs_per_dataset):
      try:
        response = llm.generate_text(model_name='google/gemini-2.5-flash', prompt=current_prompt)
      except Exception as e:
        response = str(e)
        print(f"\n {response}")
      dataset_path = dt_main_path / dataset
      dataset_path.mkdir(exist_ok=True)
      response_path = dataset_path / f"dt_function{i}.txt"
      with open(response_path, 'w') as f:
        f.write(response)