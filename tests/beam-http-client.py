from beam import resource


def main():
    alg = resource('http://localhost:44044')
    print(alg.run('hi'))


if __name__ == '__main__':
    main()