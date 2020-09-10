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
# ------ fstab.sh --------
for name in c0202 c0203 c0204 c0205 c0206 c0207 c0208 c0209 c0211 c0212
do
    scp fstab $name:/etc/fstab
    ssh $name mount -a
done
# ------ get --------
#!/usr/bin/bash

sed 's/http/\nhttp/g'|\
sed 's/jpg/jpg\n/g'|\
sed 's/png/png\n/g'|\
grep http|\
sort -u

# ------ getGif --------
 #!/bin/sh

palette="palette.png"

filters="fps=25,scale=900:-1:flags=lanczos"

ffmpeg -v warning -i $1 \
       -vf "$filters,palettegen" \
       -y $palette
ffmpeg -v warning -i $1 -i $palette \
       -lavfi "$filters [x]; [x][1:v] paletteuse" -y $2

# ------ get_ref_tex --------
#!/usr/bin/bash

add_frame()
{
    echo "  %%%%%%%%%%%%%%%%%%%%%%%%%%" >> article.tex
    echo "  \\begin{frame}" >> article.tex
    echo "      \\frametitle{Data Pictures}" >> article.tex
    echo "      \\begin{figure}[h]" >> article.tex
    echo "          \\includegraphics[height=6cm]{fig${1}.png}" >> article.tex
    echo "      \\end{figure}" >> article.tex
    echo "  \\end{frame}" >> article.tex
}
addframes()
{
    n1=7
    n2=$1
    if [ $n2 -ge 7 ];then
        for(( i = $n1;i <= $n2; i = i + 1 ))
        do
            add_frame $i
        done
        sed 's/\r//g' -i article.tex
        sed '74{h;d};$G' -i article.tex
    fi
}
mv ~/pictures/fig*png figures
rename fig0 fig figures/*png
i=`ls figures/*png|wc -l`
# add frames to the tex file
addframes $i
cd figures
pngWidth
# ------ getnum --------
#!/usr/bin/bash

name=$1
i=0
for num in `cat $name`
do
    ((i = i + num))
done
echo "total num is $i"
# ------ getpackage --------
#!/usr/bin/bash

dnf search $1|\
    cut -d : -f 1|\
    sed '/    /'d|\
    sed '/src/'d > pack.sh
vim pack.sh
# ------ getpdf --------
#!/usr/bin/bash

pdftk `cat mv.sh` output total.pdf

# ------ gspng --------
#!/bin/bash

for name in `ls *ps`
do
gs -r300 -dEPSCrop -dTextAlphaBits=4 -sDEVICE=png16m -sOutputFile=${name}.png -dBATCH -dNOPAUSE $name 
done
rename ps.png png *png
rename epng png *png
# ------ inc --------
#!/usr/bin/bash
name=$1
find /usr/include -name "*${name}*"
# ------ index --------
#!/usr/bin/bash

# rename a file with serial number
# by xiaoeheng
# 2015-12-18 15:54:36    

i=$1
name=$2

mv $name ${i}_$name
# ------ indexAll --------
#!/usr/bin/bash

i=0
#for name in `ls *pdf`
for name in `ls *pdf`
do
    ((i = i + 1))
    if [ $i -lt 10 ];then
        file=0${i}_$name
    else
        file=${i}_$name
    fi
    mv $name $file
done

# ------ install.sh --------
./configure
make
make install
# ------ makeAll.sh --------
#!/usr/bin/bash

for i in 0 1 3 5 7
do
    for j in 0 2 4 6 8 10
    do
        if [ $j -lt 10 ];then
            dirName=0${i}_0$j
        else
            dirName=0${i}_$j
        fi
        echo "entering $dirName"
        #cd $dirName
        #make clean
        #make
        echo "leaving $dirName"
        #cd ..
    done

done

# ------ md.sh --------
for name in `ls *md`
do
    markdown_py -o html4 $name > ${name}.html
    wkhtmltopdf ${name}.html ${name}.pdf
done
rm *html
rename md.pdf pdf
mkdir pdfs
mv *pdf pdfs
# ------ mkvideo --------
#!/usr/bin/bash

format=`ls 0*|cut -d . -f 2|sort -u`
ffmpeg -i ${3}%0${1}d.${format} \
       -vb ${2}M -vcodec mpeg4  new.avi

# ------ nameAllMedias.sh --------
#!/usr/bin/bash

for j in pdf txt srt mp4
do
    for i in {1..20}
    do
        rename ' ' '' *$j
        rename , ''   *$j
        rename '(' '' *$j
        rename ')' '' *$j
    done
done
# ------ nameall --------
#!/usr/bin/bash

for i in {1..20}
do
    rename ' ' '' *pdf
    rename , '' *pdf
    rename '.' '' *pdf
    rename '"' '' *pdf
    rename '(' '' *pdf
    rename ')' '' *pdf
    rename "'" '' *pdf
    rename "[" '' *pdf
    rename "]" '' *pdf
    rename "{" '' *pdf
    rename "}" '' *pdf
    rename "&" '_' *pdf
done
rename pdf '.pdf' *pdf

# ------ namepdf --------
#!/usr/bin/bash

for i in {1..20}
do
    rename ' ' '' *pdf
    rename , '' *pdf
    rename '.' '' *pdf
    rename '(' '' *pdf
    rename ')' '' *pdf
    rename "'" '' *pdf
done
rename 'pdf' '.pdf' *pdf

# ------ ncl --------
#!/usr/bin/expect 
set timeout 60 
spawn ssh ncl@222.195.94.7
      interact {         
            timeout 300 {send "\x20"} 
      }
# ------ new.sh --------
#!/usr/bin/bash


n1=1
n2=100
for ((i = $n1; i <= $n2; i = i + 1))
do
    echo "go for a new one $i"
done

source cd ~/fortran
# ------ newC --------
#!/usr/bin/bash

name=$1
cp $name /etc/init.d
for i in {1..5}
do
    j=`ls /etc/rc${i}.d|wc -l`
    ((j = j + 100))
    echo "j = is $j"
    cp $name /etc/rc${i}.d/S${j}$name
done

# ------ newicon.sh --------
#!/usr/bin/bash

path=/home/hdeping/Desktop
cp $path/eog.desktop $path/$1.desktop
sed -i "s/eog/$1/g" -i  $path/$1.desktop
# ------ newtex.sh --------
#!/bin/bash
# to replace some special
# characters with new one
#  in tex files 
# by xiaohengdao
# 2015-06-23 14:07:08    

ft=tex
except=
for name in `ls|grep $ft|grep -v $except`
do
sed '1,$s/#/\\#/g' -i $name
sed '1,$s/&/\\\&/g' -i $name
#sed '1,$s/&/\\\&/g' -i $name
done
# ------ pan --------
#!/usr/bin/bash

name=$1
pandoc ${name}.md -o ${name}.pdf
# ------ pic.sh --------
#!/usr/bin/gnuplot
set terminal eps
#set terminal pdfcairo
#set font "Times-Roman,22"
set output "pic.eps"
set xlabel "x" font "Times-Roman,18"
set ylabel "y" font "Times-Roman,18"
set xtics nomirror
set ytics nomirror
#set xtics 0.2
set tics font "Times-Roman,16"
#set xrange[0:50]
#set yrange[0:1]
set size square
set key font "Times-Roman,18"
set title "Data" font "Times-Roman,18"
#set key at 48,0.6
set pointsize 0.2
filename="data.txt" 
plot filename using 1:2 w l lw 4   title "S" 


# ------ pic_histogram.sh --------
#!/usr/bin/gnuplot
set terminal eps
#set terminal pdfcairo
#set font "Times-Roman,22"
set output "pic.eps"
set xlabel "x" font "Times-Roman,18"
set ylabel "y" font "Times-Roman,18"
set xtics nomirror
set ytics nomirror
set tics font "Times-Roman,16"
#set xrange[0:50]
#set yrange[0:1]
set size square
set key font "Times-Roman,18"
set title "{/Symbol b}=0.5,{/Symbol g}=0.1" font "Times-Roman,18"
#set key at 48,0.6
set style data histogram
set style fill solid 0.4 border
filename="data.txt" 
plot filename 
# ------ pic_palette.sh --------
#!/usr/bin/gnuplot
set terminal eps
set output "pic.eps"
set xlabel "x"
set ylabel "y"
set key box
set title "data"
set palette model RGB defined (0 "red", 1 "blue", 2 "green")
plot "data.txt" using 1:2:3 notitle w p pt 3 palette
# ------ pm3d.sh --------
#!/usr/bin/gnuplot
set terminal pdf enhanced
set output "pm3ddata.pdf"
set xlabel "/Symbol{r}"
set ylabel "r"
set xrange[0.12:1.04]
set yrange[1:7] %set zrange[0:7]
set title "data"
#set dgrid3d 100,100
#set contour
#set cntrparam levels incremental 1,0.2,7
#unset surface
unset ztics
set view 0,0
splot "dataall.txt"  w  pm3d 
# ------ pngWidth --------
#!/usr/bin/bash

deal_with_png()
{
file *png|\
    awk -F , '{print $1 $2}'|\
    sed 's/PNG image data//g'|\
    sed 's/fig//g'|\
    sed 's/.png://g'|\
    sed 's/^\([0-9] \)/0\1/g'|\
    sort -u|\
    sed 's/\( [0-9][0-9][0-9] \)/ \1/g'|\
    awk '{print $1 " " $2 " " $4}' > data.txt

}
deal_with_jpg()
{
file *jpg|\
    awk -F , '{print $1 $8}'|\
    sed 's/JPEG image data//g'|\
    sed 's/.jpg://g'|\
    sed 's/x/ /g'|\
    sort -u|\
    awk '{print $1 " " $2 " " $3}' > data.txt

}
deal_with_jpg
get_name()
{
    if [ $1 -lt 10 ];then
        echo "00${1}.jpg"
    elif [ $1 -lt 100 ];then
        echo "0${1}.jpg"
    else
        echo "${1}.jpg"
    fi
}
for i in `get_pngWidth data.txt`
do
    name=`get_name $i`
    echo "name is $name"
    sed "s/\[height=6cm\]\({$name}\)/\[width=11cm\]\1/g" -i mv.tex 
done
# ------ pvg --------
#!/usr/bin/bash

for i in $*
do
    echo "convert $i.pdf to $i.svg"
    pdf2svg $i.pdf $i.svg
done

# ------ raw.sh --------
#!/usr/bin/bash

for i in `ls *CR2`
do
    ufraw-batch --out-type=jpg --out-path=. $i
done

# ------ removeAll --------
#!/usr/bin/bash

find . -name "*$1" -exec rm -fr {} \;
# ------ rename.sh --------
#!/bin/bash
i=0
for name in `ls|grep -v mv`
do
   let "i ++"
   if [ $i -lt 10 ];then
        mv $name 0${i}.jpg
    else
        mv $name ${i}.jpg
    fi
done 
i1="new.sh"
echo $i1
# ------ res.sh --------
#!/usr/bin/bash

convert -resize $1% $2 new_$2
# ------ resize.sh --------
#!/usr/bin/bash

for i in `ls *jpg`
do
    echo "$i"

    res.sh $1 $i
done

# ------ rsync.sh --------
#!/usr/bin/bash

ip="hdp@210.45.125.225:node_modules"
node=shell/lib/node_modules
rsync -avz $ip/ /home/hdeping/$node
#rsync -avz /home/hdeping/$node/ 202.38.82.23:$node
# ------ sample.sh --------
#!/usr/bin/bash


# ------ scpfile --------
#!/bin/bash

name=$1
scp $name hdp@202.38.82.29:
# ------ secondLayerRename.sh --------
#!/usr/bin/bash

for i in {1..12}
do
    if [ $i -lt 10 ];then
        name=0${i}*
    else
        name=${i}*
    fi
    cp nameall $name
    cd $name
    k=`ls|grep 0|wc -l`
    n1=1
    n2=$k
    for(( ii = $n1;ii <= $n2; ii = ii + 1 ))
    do
        file=0${ii}*
        cp nameall $file
        cd $file
        ./nameall
        cd ..
    done
    cd ..
done

# ------ sed.sh --------
#!/bin/bash
#  program
#   to extract specified 
#   lines in multiple files
#  by xiaohengdao
#  2015-03-26 22:41:45    
tp=jpg
html=htm
grep ${jpg}  *${html}|sed "s/${tp}/${tp}\n/g" -i|sed "s/http/\nhttp/g" -i|sort -u|grep jpg|grep http > new.txt
for name in `cat new.txt`
do
    wget -b $name
done

# ------ sedc --------
#!/bin/sh

for name in `ls *c`
do
    sed -i "s/for(int i/int i;\nfor (i/g" $name
    sed -i "s/for(int j/int j;\nfor (j/g" $name
    sed -i "s/for(int k/int k;\nfor (k/g" $name
done
# ------ serial_number --------
#!/usr/bin/bash

j=0 # count number
# get the total number of pdf files
num=`ls *pdf|wc -l` 
for i in `ls *pdf`
do
   let "j++"
   if [ $num -lt 10 ];then
       name=${j}.pdf
   elif [ $num -lt 100 ];then
       if [ $j -lt 10 ];then
           name=0${j}.pdf
       else
           name=${j}.pdf
       fi
   elif [ $num -lt 1000 ];then
       if [ $j -lt 10 ];then
           name=00${j}.pdf
       elif [ $j -lt 100 ];then
           name=0${j}.pdf
       else
           name=${j}.pdf
       fi
   elif [ $num -lt 10000 ];then
       if [ $j -lt 10 ];then
           name=000${j}.pdf
       elif [ $j -lt 100 ];then
           name=00${j}.pdf
       elif [ $j -lt 1000 ];then
           name=0${j}.pdf
       else
           name=${j}.pdf
       fi
   fi
   echo "Old name: $i, New name: $name"
   mv $i $name
done
# ------ sort2.sh --------
#!/usr/bin/bash

home=/home/hdeping
dir=$1
for i in `ls $home/*jpg`
do
    file=$(file $i|cut -d '`' -f 2|sed "s/'//g")
    echo "mv $file to $dir"
    mv $file ../$dir
done
echo "DONE!"
rm $home/*jpg
# ------ sortPicture.sh --------
#!/usr/bin/bash

home=/home/hdeping
dir=$1
mkdir $dir
for i in `ls $home/*jpg`
do
    file=$(file $i|cut -d '`' -f 2|sed "s/'//g")
    echo "mv $file to $dir"
    mv $file $dir
done
echo "DONE!"
rm $home/*jpg
mv $dir ..
# ------ speedAudio.sh --------
#!/usr/bin/bash


ffmpeg -i $1 -filter:a "atempo=2.0" -vn 2x_$1
# ------ st --------
#!/usr/bin/bash

var=`date |sed 's/ĺš´//g'|sed 's/ć//g'|sed 's/ćĽ//g'|sed 's/ //g'|sed 's/://g'`

dir=${var:0:8}
dir2=${var:11:6}

echo "cp $1 to ${dir}_${dir2}_$1"

cp $1 ${dir}_${dir2}_$1

# ------ svg --------
#!/usr/bin/bash

for i in $*
do

    echo "convert $i.svg to $i.png"
    convert $i.svg $i.png
done

# ------ svg2pdf --------
#!/bin/bash

inkscape --without-gui --file=$1.svg --export-pdf=$1.pdf
# ------ svgNoBack --------
#!/usr/bin/bash

for i in $*
do

    echo "convert $i.svg to $i.png"
    convert -background none  $i.svg $i.png
done

# ------ sync_ref --------
#!/usr/bin/bash

ref=/home/hdeping/complexNetwork/02_ReferenceNote
ref1=/home/hdeping/complexNetwork/01_myReference
cp_ref()
{
    i=$1
    cd $i
    for name in `ls ${ref1}/${i}/|cut -d '.' -f 1`
    do
        echo $name
        cp -r $ref/sample  $name 
    done
    cd ..
}
dir=`pwd|sed 's/02_ReferenceNote/01_myReference/g'`
for name in `ls $dir|cut -d '.' -f 1`
do
    if [ ! -d $name ];then
        echo $name
        cp -r $ref/sample  $name 
    fi
done

# ------ timeSync.sh --------
#!/usr/bin/bash

ip=210.45.125.10
# start the ntp server
service ntpd start

# synchronization with the ntp server
for i in {1..9}
do
    echo " ssh ${ip}${i} 'ntpdate ${ip}0'"
done

# ------ tran --------
#!/bin/sh
rename F90 f90 *F90
char()
{
    local name=$1
    sed -i 's/A/a/g' $name
    sed -i 's/B/b/g' $name
    sed -i 's/C/c/g' $name
    sed -i 's/D/d/g' $name
    sed -i 's/E/e/g' $name
    sed -i 's/F/f/g' $name
    sed -i 's/G/g/g' $name
    sed -i 's/H/h/g' $name
    sed -i 's/I/i/g' $name
    sed -i 's/J/j/g' $name
    sed -i 's/K/k/g' $name
    sed -i 's/L/l/g' $name
    sed -i 's/M/m/g' $name
    sed -i 's/N/n/g' $name
    sed -i 's/O/o/g' $name
    sed -i 's/P/p/g' $name
    sed -i 's/Q/q/g' $name
    sed -i 's/R/r/g' $name
    sed -i 's/S/s/g' $name
    sed -i 's/T/t/g' $name
    sed -i 's/U/u/g' $name
    sed -i 's/V/v/g' $name
    sed -i 's/W/w/g' $name
    sed -i 's/X/x/g' $name
    sed -i 's/Y/y/g' $name
    sed -i 's/Z/z/g' $name
    sed -i 's/\r//g' $name
}
for name in `ls *f90`
do
    echo "change the tab character"
    sed 's/\t/    /g' -i $name
    echo "change the base"
    char $name
done
# ------ unarall --------
#!/bin/sh

back=*$1
for name in `ls $back`
do
    unar $name 
done

# ------ vm --------
#!/usr/bin/bash

name=${1}.vim
vim ~/.vim/ftplugin/$name