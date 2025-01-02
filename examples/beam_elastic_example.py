from beam import resource

es = resource('elastic://localhost:9200')

stocks = es.joinpath('stocks')
print(stocks)