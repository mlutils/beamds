mkdir -p /opt/oracle \
    && cd /opt/oracle \
    && wget https://download.oracle.com/otn_software/linux/instantclient/instantclient-basiclite-linuxx64.zip \
    && unzip instantclient-basiclite-linuxx64.zip \
    && rm instantclient-basiclite-linuxx64.zip \
    && ln -s $ORACLE_HOME/*/libclntsh.so* $ORACLE_HOME \
    && ln -s $ORACLE_HOME/*/libocci.so* $ORACLE_HOME

pip install cx_Oracle