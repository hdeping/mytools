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

# ------ crop_png --------
#!/usr/bin/bash

cut_sw()
{
    convert $1 -gravity southwest -crop 1500X1500+0+0 new$1
}
cut()
{
    convert $1 -gravity center -crop 900X900+0+0 new$1
}
for name in $*
do
    cut $name
done
# ------ discuz --------
#!/bin/sh

rsync -avz hdp@210.45.125.225:/home/http/ /home/hdeping/02_disk/discuz/ 
ssh hdp@210.45.125.225  'mysqldump -u root -pnclxin ultrax > ultrax.sql'
scp hdp@210.45.125.225:ultrax.sql /home/hdeping/02_disk/

# ------ down.sh --------
#!/bin/bash
#  program 
#    to produce a series of numbers
# by xiaohengdao
#    2015-03-18 18:37:44    

web=""
htm=".html"
wget -b $web${htm}
n1=1
n2=20
for((i = ${n1}; i <= $n2; i = i + 1))
do
    wget -b  ${web}${htm}
done

# deal with the html files
grep jpg *html|sed 's/jpg/jpg\n/g'|sed 's/http/\nhttp/g'|grep jpg|grep http|sort -u > new.txt
wget -b -i new.txt


# ------ doxy --------
#!/usr/bin/bash

doxy()
{
    cp ~/shell/Doxyfile .
    sed "s/0000/$1/g" -i Doxyfile
    doxygen
}
doxy $1
# ------ doxy.sh --------
#!/usr/bin/bash

doxy()
{
for name in `find . -maxdepth 1 -type d|grep ..`
do
    cd $name
    cp ~/shell/Doxyfile .
    doxygen
    cd latex
    make
    cp refman.pdf ../../${name}.pdf
    cd ../..
done
}
cp_png()
{
    j=0
    for i in `find . -name "inherit*" -a -name "*png"`
    do
        ((j=j+1))
        if [ $j -lt 10 ];then
            name="00${j}.png"
        elif [ $j -lt 100 ];then
            name="0${j}.png"
        else
            name="${j}.png"
        fi
        cp $i .
    done
}
main()
{
    doxy
    cp_png
    mkdir doc
    mv *pdf doc
    mv *png doc
    cd doc
    mkdir pdf png
    mv *.pdf pdf
    mv *.png png
}
main
# ------ doxy_new.sh --------
#!/usr/bin/bash

doxy()
{
for name in `find . -maxdepth 1 -type d|cut -d / -f 2|grep ..`
do
    echo "$name"
    cd $name
    cd latex
    cp ~/shell/Doxyfile .
    sed "s/0000/$name/g" -i Doxyfile
    doxygen
    cd latex
    make
    cp refman.pdf ../../${name}.pdf
    cd  ../..
done
}
doxy

# ------ extern_produce --------
#!/usr/bin/bash
sub()
{
    name=$1
    sed -i '/}/'d $name
    sed -i '/{/'d $name
    sed -i '/;/'d $name
    sed -i '/if /'d $name
    sed -i '/include/'d $name
    sed -i '/^    /'d $name
    sed -i '/^$/'d $name
    sed -i 's/$/;/g' $name
    sed -i '/\/\//'d $name
    sed -i '/#define/'d $name
    awk -v column=1 -v value="extern " '
        BEGIN {
            FS = OFS = "";
        }
        {
            for ( i = NF + 1; i > column; i-- ) {
                $i = $(i-1);
            }
            $i = value;
            print $0;
        }
    ' $name > now
    mv now $name
}
sub $1

# ffmpeg related
# concat the videos
ffmpeg -f concat -i list1.lst -c copy video.mp4
# distill the audio from a video
ffmpeg -i ${name}.mp4 -ab 32k ${name}.mp3

#  remove all the files with specific sizes
name=-10k
find . -name "*jpg" -a -size $name -exec rm {} \;

# ------ font.sh --------
#!/usr/bin/gnuplot
set terminal eps
#set terminal pdfcairo
#set font "Times-Roman,22"
set output "pic.eps"
set xlabel "t" font "Times-Roman,18"
set ylabel "{/Symbol r}" font "Times-Roman,18"
set xtics nomirror
set ytics nomirror
set tics font "Times-Roman,16"
set xrange[0:50]
set yrange[0:1]
set size square
set key font "Times-Roman,18"
set title "{/Symbol b}=0.5,{/Symbol g}=0.1" font "Times-Roman,18"
set key at 48,0.6
set pointsize 0.2
filename="data0.50-0.10.txt" 
plot filename using 1:2 w l lw 4 title "S" ,\
    filename using 1:3 w l lw 4  title "I" ,\
    filename using 1:4 w l lw 4  title "R" 







