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


# ------ add_number.sh --------
#!/usr/bin/bash

addNumber()
{
    convert -pointsize 120 \
            -fill black \
            -draw 'text 154,177 "10"'\
            $1.png number_$1.png
}
cmd="convert -pointsize 120 -fill black"
$cmd -draw 'text 154,177 "1"'  1.png number_1.png
# ------ avi2mp4 --------
#!/usr/bin/bash
avi2mp4()
{
    ffmpeg -i $1.avi\
           -vcodec libx264\
           -crf 25\
           -acodec libfaac\
           -t 60 $1.mp4
}
avi2webm()
{
    ffmpeg -i $1.avi\
           -codec:a libvorbis\
           -f webm $1.webm
}
flv2mp4()
{
    ffmpeg -i $1.avi\
           -c:a libx264\
           -crf 25 $1.mp4
}
mp4()
{
    ffmpeg -i $1\
           -vcodec libx264\
           new$1.mp4
}
mp4 $1