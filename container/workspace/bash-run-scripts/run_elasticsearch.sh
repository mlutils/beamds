# alternatively run with docker-compose: https://github.com/shazforiot/Elasticsearch-logstash-Kibana-Docker-Compose/blob/main/docker-compose.yml

ELASTICSEARCH_PORT=$1
KIBANA_PORT=$2

# configure the port in the elasticsearch.yml file (/etc/elasticsearch/elasticsearch.yml)
# replace with sed http.port: 9200 with http.port: $$ELASTICSEARCH_PORT

sed -i "s/http.port: 9200/http.port: $ELASTICSEARCH_PORT/g" /etc/elasticsearch/elasticsearch.yml

# Kibana configuration settings

# The URL of the Elasticsearch instance to use for all your queries.
echo "elasticsearch.hosts: [\"http://localhost:$ELASTICSEARCH_PORT\"]" >> /etc/kibana/kibana.yml
echo "server.port: $KIBANA_PORT" >> /etc/kibana/kibana.yml

mkdir /workspace/.kibana
export PUPPETEER_HOME=/workspace/.kibana
chown -R kibana:kibana /workspace/.kibana/
chmod -R 755 /workspace/.kibana

supervisorctl reread
supervisorctl update

supervisorctl restart elasticsearch
supervisorctl restart kibana
