##How to setup this project
#create and enter the virtualenv
virtualenv gcp
source env/bin/activate

#install required software
chmod +x initalsoftware.sh
./initalsoftware.sh

#get cert
chmod +x generate_keys.sh
./generate_keys.sh

#How to run this project
virtualenv gcp
source env/bin/activate
    