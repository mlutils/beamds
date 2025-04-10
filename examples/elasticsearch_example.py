from beam import resource
from beam.docs.queries import TimeFilter
from datetime import datetime

if __name__ == '__main__':

    es = resource('elastic://10.0.7.228:31063')
    try:
        es.ping()
    except Exception as e:
        es.ping()

    ind = es.joinpath('historical_prices')
    # df = (ind & "ticker: msft").sample(2)

    # q = (ind & "ticker: msft").random_generator(field='open')
    # df = q.as_df(size=4, score=True)
    #
    # print(df)

    q = ind & 'ticker: ms*' & TimeFilter(field='date', start=datetime(2023, 1, 1), end=datetime(2024, 1, 1))
    q.groupby(['ticker', 'interval']).agg(
        {'close': 'mean', 'open': 'mean', 'high': 'mean', 'low': 'mean'}).as_df().head()

    # r = (ind & "ticker: avgo")['open, close']

    # r = (ind & "ticker: avgo")
    # r.set_fields(['open', 'close'])

    # q = ind & "ticker: av*"
    # g = q.dropna(subset=['open', 'close']).groupby(['ticker', 'version'])
    # gg = g.agg({'open': 'sum', 'close': 'max', 'date': ['count', 'nunique']})
    #
    # print(gg.values)

    # t = ind['ticker']
    #
    # q = (t == 'avgo')
    # print(q.ids)

    print('Done!')