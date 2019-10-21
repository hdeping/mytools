#git checkout 4e40e1ca2ada38ccb019dcfc96cca18531bee68d
#git checkout a602ee3a820451ec22499d99d303ba8b5771c9b7

run(){
    for line in main.c makefile head.h
    do
        cp $1/$line .
    done
}
run2(){
    for dir in 01_v1 02_v2 03_rightMatrix
    do
        run $dir
        git add . 
        git commit -m "add 06_CTRW/$dir"
    done
}
runCorr(){
    prefix=/Users/huangdeping/c/01_codes/05_treeMaker/02_getValue
    for dir in 01_v1 02_v2 03_v3 04_v4 05_v5 06_v6 07_07 08_08 09_v9 10_v10 11_v11 12_v12 13_v13 14_bisector 15_bisector 16_escribe 17_v17 18_v18 19_v19
    do
        #run $prefix/$dir
        cp $prefix/$dir/*py .
        git add . 
        git commit -m "add 05_treeMaker/02_getValue/$dir"
    done
}
runCorr
