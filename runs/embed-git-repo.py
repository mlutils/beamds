import os
import git
import torch
import pandas as pd

from beam.llm import text_splitter
from beam import tqdm, BeamData, resource


def clone_and_extract_textual_files(repo_url, local_dir, branch='master', textual_extensions=None):
    if textual_extensions is None:
        textual_extensions = {'.txt', '.md', '.py', '.json', '.xml', '.html', '.js'}  # Add more as needed

    # Clone the repository
    repo = git.Repo.clone_from(repo_url, local_dir, branch=branch)

    file_contents = {}

    # Walk through the repository directory
    for subdir, dirs, files in os.walk(local_dir):
        for file in files:
            filepath = os.path.join(subdir, file)
            extension = os.path.splitext(file)[1]
            if extension in textual_extensions:
                relative_path = os.path.relpath(filepath, start=local_dir)
                # Read and store the content of textual files
                with open(filepath, 'r', encoding='utf-8') as file_handle:
                    file_contents[relative_path] = file_handle.read()

    return file_contents


def main():

    # Usage example
    repo_url = 'https://github.com/mlutils/beamds.git'  # Replace with your repo URL
    local_dir = '/tmp/repo'  # Replace with your desired local path
    output_dir = '/tmp/emb'  # Replace with your desired output path
    branch = 'dev'  # Specify the branch you want to clone
    emb_model = 'emb-openai:///text-embedding-3-large'

    local_dir = resource(local_dir)
    local_dir.clean()
    local_dir.mkdir()

    text_files_content = clone_and_extract_textual_files(repo_url, local_dir.str, branch=branch)

    emb = resource(emb_model)

    git_embs = []
    for k, v in tqdm(text_files_content.items()):
        if len(v):
            try:
                git_embs.append({'path': k, 'embedding': emb.embed_query(v), 'text': v, 'part': 0, 'parts': 1})
            except:
                vs = text_splitter(v, chunk_size=4000)
                for i, vsi in enumerate(vs):
                    git_embs.append(
                        {'path': k, 'embedding': emb.embed_query(vsi), 'text': vsi, 'part': i, 'parts': len(vs)})

    d = torch.stack([v['embedding'] for v in git_embs])
    df = pd.DataFrame(git_embs, columns=[k for k in git_embs[0].keys() if k not in ['embedding']])
    bd = BeamData({'values': d, 'metadata': df})

    bd.store(output_dir)

    # print(text_files_content)  # Print the dictionary containing file paths and their contents
