mkdir -p /mnt/data/elasticsearch
chown 1000:1000 /mnt/data/elasticsearch
docker-compose up -d



docker network create elk_network

docker run -d \
  --name elasticsearch \
  --network elk_network \
  -p 9200:9200 -p 9300:9300 \
  -v /localdata/elasticsearch:/usr/share/elasticsearch/data \
  -e discovery.type=single-node \
  -e xpack.security.enabled=false \
  -e xpack.security.http.ssl.enabled=false \
  -e xpack.security.transport.ssl.enabled=false \
  docker.elastic.co/elasticsearch/elasticsearch:8.16.1


docker run -d \
  --name kibana \
  --network elk_network \
  -p 5601:5601 \
  -e ELASTICSEARCH_HOSTS=http://elasticsearch:9200 \
  docker.elastic.co/kibana/kibana:8.16.1

docker run -d \
  --name logstash \
  --network elk_network \
  -p 5000:5000 -p 9600:9600 \
  -v "$(pwd)/logstash/pipeline:/usr/share/logstash/pipeline" \
  -e xpack.monitoring.enabled=false \
  docker.elastic.co/logstash/logstash:8.16.1
