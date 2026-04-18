# LLM Trees - Decision Trees through LLMs
![python](https://img.shields.io/badge/python-3.12.13%2B-blue)

## SUMMARY
Replication of Oh LLM, I’m Asking Thee, Please Give Me a Decision Tree”: Zero-Shot Decision Tree Induction and Embedding with Large Language Models (KDD conference) paper for DS8008 Class.
- We prompt Gemini 2.5 Flash with informationa about a dataset, such as features, and ask it to generate decision tree logic internally.
- It also then takes that internal logic and creates a function that can be used as a classifier.
- The logic of these functions are then used to evaluate the performance of of these decision trees vs. randomly guessing.
- Each dataset is used to prompt the LLM 5 times, to generate 5 different decision trees.

## 📂 Folder Structure
```
DS8008_LLM_TREE
├──data
│   ├── data_sets
│   └── Gemini_outputs
├──src
│   ├── Evaluators
│   │   ├──Embeddings
│   │   │   ├── config.py
│   │   │   ├── emb_generator_bankruptcy.py
│   │   │   ├── emb_generator_boxing1.py
│   │   │   ├── emb_generator_boxing2.py
│   │   │   ├── emb_generator_colid.py
│   │   │   ├── emb_generator_credit.py
│   │   │   └── embedding_eval.py
│   │   ├──model_extractors
│   │   │   ├── bankruptcy_models.py
│   │   │   ├── boxing1_models.py
│   │   │   ├── boxing2_models.py
│   │   │   ├── colic_models.py
│   │   │   └── credit_models.py
│   │   ├── bankruptcy.py
│   │   ├── boxing1.py
│   │   ├── boxing2.py
│   │   ├── colic.py
│   │   ├──creditcore.py
│   │   ├── emb_eval_looper.py
│   │   ├── evaluate_bankruptcy.py
│   │   ├── evaluate_boxing1.py
│   │   ├── evaluate_boxin2.py
│   │   ├── evaluate_colic.py
│   │   └── evaluate_creditcore.py
│   ├── Figures
│   │   ├── func_example_full.png
│   │   ├── llm_flow.png
│   │   ├── sample_output.png
│   │   ├── sample_output2.png
│   │   └── sample_return_format.png
│   ├── dt_extractor_colab.ipynb
│   ├── gemini_prompter.py
    └── plotter.py
├──.gitignore
├── LICENSE
├── LLM_Trees_Abstract.pdf
├──LLM_Trees_project_report.ipynb
├── README.md
└── requirements.txt
```

## Required Environment
- python >= 3.12.13
``` python
pip install -r requirements.txt
```

## How to extract LLM trees
The method we used mainly works on colab using the google.colab.ai api which doesn't require an api key.

- Upload [dt_extractor_colab](src/dt_extractor_colab.ipynb) and use the data_sets folder in order to extract the functions.

For Example: if data_sets is in the working directory, you can set your out output_path and it will save the LLM outputs there.
``` python
generate_dt_funcs(
    datasets_path="./data_sets",
    output_path="./Gemini_2-5-Flash",
    model=ai,
    num_dt=5
)
```

## Evaluate LLM Trees
This is done using the [project report notebook](LLM_Trees_project_report.ipynb) which you can easily follow along to test the functions that we extracted.

**[NOTE]** To test your own functions that you extract, you will need to edit them yourself, and create an evaluation function for them. You can refer to the scripts in `src/Evaluators` for examples.

