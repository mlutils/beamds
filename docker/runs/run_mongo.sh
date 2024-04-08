#!/bin/bash

# MongoDB new port number
NEW_PORT=$1

# MongoDB configuration file
CONFIG_FILE="/etc/mongod.conf"

# Check if the script is run as root
if [[ $EUID -ne 0 ]]; then
   echo "This script must be run as root"
   exit 1
fi

# Function to create a new MongoDB configuration file
create_new_config() {
    echo "Creating a new MongoDB configuration file..."
    echo "net:" > $CONFIG_FILE
    echo "  port: $NEW_PORT" >> $CONFIG_FILE
    echo "  bindIp: 127.0.0.1" >> $CONFIG_FILE
    echo "storage:" >> $CONFIG_FILE
    echo "  dbPath: /var/lib/mongodb" >> $CONFIG_FILE
    echo "  journal:" >> $CONFIG_FILE
    echo "    enabled: true" >> $CONFIG_FILE
    echo "systemLog:" >> $CONFIG_FILE
    echo "  destination: file" >> $CONFIG_FILE
    echo "  logAppend: true" >> $CONFIG_FILE
    echo "  path: /var/log/mongodb/mongod.log" >> $CONFIG_FILE
    echo "processManagement:" >> $CONFIG_FILE
    echo "  timeZoneInfo: /usr/share/zoneinfo" >> $CONFIG_FILE
    echo "New MongoDB configuration file created."
}

# Check if the configuration file exists
if [ -f "$CONFIG_FILE" ]; then
    # Update the MongoDB configuration file to use the new port
    sed -i "/^net:/a \ \ port: $NEW_PORT" $CONFIG_FILE
else
    create_new_config
fi

# Restart MongoDB to apply the changes
#systemctl restart mongod
service mongod start

# Confirm completion
echo "MongoDB port changed to $NEW_PORT and service restarted."