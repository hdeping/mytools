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
    path=/home/hdeping/c/43_haishan_erg/00_codeBefore
    for dir in 001_v1 002_v2 003_v3_NoO- 004_newStati 005_atomSeq 006_repeatsSplit 007_twoRepeatsClass 008_smilesGraph 009_smilesAna 010_openbabel 011_smilesSvg 012_smilesCompara 013_jsonTry 014_getJson 015_getJson2 016_getJson3 017_getJson4 018_getJson5 019_getJson6 020_getJson7 021_getCorrentNumber 022_getCorrentNumber 023_getCorrentNumber 024_getCorrentNumber 025_getCorrentNumber 026_getCorrentNumber 027_getCorrentNumber 028_getCorrentNumber 029_getCorrentNumber 030_getJson7 031_getJson8 032_getCorrentNumber 033_getCorrentNumber 034_getCorrentNumber 035_repeatIDMol 036_getJson10 037_repeatIDMol 038_getCorrentNumber 039_getCorrentNumber 040_getCorrentNumber 041_getCorrentNumber 042_atomSeq 043_getCorrentNumber 044_getCorrentNumber 045_atomSeq 046_getCorrentNumber 047_getCorrentNumber 048_getCorrentNumber 049_getCorrentNumber 050_atomSeq 051_getCorrentNumber 052_getCorrentNumber 053_getCorrentNumber 054_getCorrentNumber 055_getCorrentNumber 056_getCorrentNumber 057_getCorrentNumber 058_getCorrentNumber 059_getCorrentNumber 060_getCorrentNumber 061_getFinger 062_getFinger 063_getFinger 064_getFinger 065_getFinger 066_getFinger 067_getFinger 068_getFinger 069_getFinger 070_getFinger 071_getFinger 072_getFinger 073_getJson8 074_getFinger 075_getFinger 076_getFinger 077_getFinger 078_getFinger 079_getFinger 080_getFinger 081_getFinger 082_getParas 083_getParas 084_getParas 085_getParas 086_getParas 087_getParas 088_getParas 089_getParas 090_getTrainingData 092_getParas 093_getParas 094_getParas 095_getParas 096_getParas 097_getParas 098_getParas 099_getParas 101_getParas 102_getTrainningData 103_gettrain_valence 104_gettrain_valence 105_getParas_valence 106_getParas_valence 107_getParas_valence 108_getParas_valence_order
    do
        cp $path/$dir/*py .
        git add . 
        git commit -m "add $dir"
    done
}
runCorr
