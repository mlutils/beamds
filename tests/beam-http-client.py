from src.beam import resource


def main():
    alg = resource('beam-https://localhost:31150')
    print(alg.run('hi'))


if __name__ == '__main__':
    main()