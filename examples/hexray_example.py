from examples.example_utils import add_beam_to_path
add_beam_to_path()
from src.beam.hex import HexBeam


if __name__ == '__main__':

    path = '/home/dsi/elads/data/hexray/code_analysis.json'
    llm = 'openai:///gpt-4'

    hb = HexBeam.from_analysis_path(path, llm)
    states = hb.research_question("What is the purpose of this program?", budget=20)

    print(states)

