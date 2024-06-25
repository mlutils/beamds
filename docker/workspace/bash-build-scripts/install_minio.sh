# follow https://chatgpt.com/share/4ac3a8f6-81af-42b9-915d-7ac0e552aff8

wget https://dl.min.io/server/minio/release/linux-amd64/minio
chmod +x minio
mv minio /usr/local/bin/

useradd -r minio-user -s /sbin/nologin
mkdir /usr/local/share/minio
chown minio-user:minio-user /usr/local/share/minio
mkdir /etc/minio
chown minio-user:minio-user /etc/minio

cp /workspace/configuration/minio.service /etc/systemd/system/minio.service

