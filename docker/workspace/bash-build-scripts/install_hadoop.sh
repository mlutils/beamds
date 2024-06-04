apt-get install -y software-properties-common wget openjdk-8-jdk

HADOOP_LATEST_VERSION=$(curl -s https://downloads.apache.org/hadoop/common/ | grep -o 'hadoop-[0-9]*\.[0-9]*\.[0-9]*' | sort -V | tail -1)
wget https://downloads.apache.org/hadoop/common/$HADOOP_LATEST_VERSION/$HADOOP_LATEST_VERSION.tar.gz
tar -xzf $HADOOP_LATEST_VERSION.tar.gz
mv $HADOOP_LATEST_VERSION /usr/local/hadoop
rm $HADOOP_LATEST_VERSION.tar.gz