from src.beam import resource


if __name__ == '__main__':
    llm = resource('openai:///gpt-4o')
    # res = llm.chat('Hello, my name is elad?')
    # print(res.text)
    # res = llm.chat('Can you remember my name?',)
    # print(res.text)
    # res = llm.chat('Can you remember my name?', system='return the answer as a json format')
    # print(res.json)
    # res = llm.chat('Can you remember my name?', reset_chat=True)
    # print(res.text)
    path = resource(f"{__file__}").parent.parent.joinpath('resources', 'beam_icon.png')
    res = llm.chat('What is in the picture?', image=path.str,  reset_chat=True)
    print(res.text)


