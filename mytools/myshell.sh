# mount the nfs
for i in {1..14}
do
    echo "mounting node$i"
    ssh node$i "mount 192.168.10.254:/home /home"
done

# ------ Makefile_sample --------
gsl = gcc -g -o exe
run:
    $(gsl) main.c 
    ./exe
2:
    gcc -O2 main.c -o exe
    ./exe
g:
    gdb exe
p:
    gnuplot pic.sh
    gspng
    crop_png pic.png
    rm pic.png && mv newpic.png pic.png
    eog pic.png