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

# ------ bak --------
#!/usr/bin/bash

name=`pwd|awk -F / '{print $NF}'`
name=${name}.tgz
rm -fr *\.o
tar zcvf ../$name *
# ------ base --------
#!/bin/sh
rename A a *
rename B b *
rename C c *
rename D d *
rename E e *
rename F f *
rename G g *
rename H h *
rename I i *
rename J j *
rename K k *
rename L l *
rename M m *
rename N n *
rename O o *
rename P p *
rename Q q *
rename R r *
rename S s *
rename T t *
rename U u *
rename V v *
rename W w *
rename X x *
rename Y y *
rename Z z *
# ------ bcchash.sh --------
bitcoin-cli getblock `bitcoin-cli getblockhash $1`
# ------ bcctran.sh --------
bitcoin-cli decoderawtransaction `bitcoin-cli getrawtransaction $1`


# ------ check --------
#!/usr/bin/bash

ping -c 3 202.38.82.29
ping -c 3 222.195.94.17
ping -c 3 222.195.94.7
# ------ clear_picture.sh --------
#!/usr/bin/bash

mkdir new
for i in `cat new.sh`
do
    mv IMG_${i}.jpg new
done
rm -fr *jpg
mv new/* . && rmdir new
rm -fr new.sh
# ------ combinePic.sh --------
#!/usr/bin/bash

file1=new1.png
file2=new2.png
convert $1.png $2.png  +append $file1
convert $3.png $4.png  +append $file2
convert $file1 $file2  -append combine_$5.png
# ------ combinePic9.sh --------
#!/usr/bin/bash

combine()
{

file1=new1.png
file2=new2.png
convert $1 $2 $3  +append $file1
convert $4 $5 $6  +append $file2
convert $7 $8 $9  +append $file3
convert $file1 $file2 $file3 -append combine_${10}.png
}