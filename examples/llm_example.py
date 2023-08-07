from example_utils import add_beam_to_path
add_beam_to_path()

from src.beam.llm import beam_llm

if __name__ == '__main__':

    llm = beam_llm("tgi://192.168.10.45:40081")

    small_chat_example = ['hi my name is elad', 'what your name?', 'how is the weather in london?',
                          'do you remember my name?']

    for t in small_chat_example:
        print(f"User: {t}")
        print(f"LLM: {llm.chat(t).text}")

    # print(llm.chat('hi my name is elad').text)
    # llm = beam_llm("tgi://192.168.10.45:40081")
    # print(llm.chat('do you remember my name?').text)