# mount the nfs
for i in {1..14}
do
    echo "mounting node$i"
    ssh node$i "mount 192.168.10.254:/home /home"
done