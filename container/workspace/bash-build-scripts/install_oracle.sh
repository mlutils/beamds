mkdir -p /opt/oracle \
    && cd /opt/oracle \
    && wget https://download.oracle.com/otn_software/linux/instantclient/instantclient-basiclite-linuxx64.zip \
    && unzip instantclient-basiclite-linuxx64.zip \
    && rm instantclient-basiclite-linuxx64.zip \
    && ln -s $ORACLE_HOME/*/libclntsh.so* $ORACLE_HOME \
    && ln -s $ORACLE_HOME/*/libocci.so* $ORACLE_HOME

pip install cx_Oracle

rm /opt/oracle/instantclient

oracle_dir=$(ls -l /opt/oracle | awk '/^d/ {print $9; exit}')
ln -s $oracle_dir $ORACLE_HOME