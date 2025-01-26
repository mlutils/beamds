from beam import resource


if __name__ == '__main__':

    es = resource('elastic://10.0.7.228:31063')
    print(list(es))

    ind = es.joinpath('historical_prices')

    # r = (ind & "ticker: avgo")['open, close']

    # r = (ind & "ticker: avgo")
    # r.set_fields(['open', 'close'])

    q = ind & "ticker: av*"
    g = q.dropna(subset=['open', 'close']).groupby(['ticker'])
    gg = g.agg({'open': 'sum', 'close': 'max'})

    print(gg.values)

    # t = ind['ticker']
    #
    # q = (t == 'avgo')
    # print(q.ids)

    print('Done!')