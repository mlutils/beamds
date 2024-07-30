apt-get update
# apt-get install openjdk-11-jdk
wget -qO - https://artifacts.elastic.co/GPG-KEY-elasticsearch | sudo apt-key add -
apt-get install apt-transport-https
echo "deb https://artifacts.elastic.co/packages/8.x/apt stable main" | sudo tee -a /etc/apt/sources.list.d/elastic-8.x.list
apt-get update
apt-get install elasticsearch

echo "[program:elasticsearch]
command=/usr/share/elasticsearch/bin/elasticsearch
autostart=true
autorestart=true
stderr_logfile=/var/log/elasticsearch.err.log
stdout_logfile=/var/log/elasticsearch.out.log
user=elasticsearch" >> /etc/supervisor/conf.d/elasticsearch.conf

supervisorctl reread
supervisorctl update

# to start elasticsearch
# supervisorctl start elasticsearch
