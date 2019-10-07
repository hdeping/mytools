# .bashrc

# Source global definitions
if [ -f /etc/bashrc ]; then
	. /etc/bashrc
fi

# Uncomment the following line if you don't like systemctl's auto-paging feature:
# export SYSTEMD_PAGER=

# User specific aliases and functions
#  about fcitx
#PS1='\[\e[35m\]\t\[\e[32m\]\#\[\e[1;31m\]^_^hdplab@\[\[\e[36m\]\h \W]\$\[\e[m\] '
PS1='\[\e[35m\]\T\[\e[32m\] ^_^ \[\e[31m\]hdplab \[\e[36m\]\W]\$\[\e[m\] '
#PS1= "\[\e[35m\]^o^\[\e[0m\]$ \[\e[31m\]\t\[\e[0m\] [\[\e[36m\]\W\[\e[0m\]] \[\e[32m\]\u\[\e[0m\]\[\e[33m\]@\[\e[0m\]\[\e[34m\]\h\[\e[0m\]\n\[\e[35m\].O.\[\e[0m\]\$ " 
# my paths {{{
ifort=/opt/intel/Compiler/11.1/064/bin/
mkl=/opt/intel/Compiler/11.1/064/mkl/
icc=/opt/icc/bin
tex=/usr/share/texlive
mender=/home/hdeping/downloads/mendeleydesktop-1.14-linux-x86_64/bin
#PATH=$mender:$ifort/intel64:$icc/intel64:$tex:$PATH:/usr/x86_64-w64-mingw32/bin/qt5
sbin=/sbin:/usr/sbin
PATH=$mender:$tex:$PATH:$sbin
export PATH
export GTK_IM_MODULE=fcitx  
export QT_IM_MODULE=fcitx  
export XMODIFIERS="@im=fcitx"
#  path for opengl
#source $ifort/ifortvars.sh intel64
#source $icc/iccvars.sh     intel64
#source /opt/intel/Compiler/11.1/064/mkl/tools/environment/mklvars64.sh
#}}}
#some varibales{{{
#}}}
#some common alias command{{{
alias vi="vim"
alias la="ls -a"
alias mx="rlwrap maxima -q"
alias em="emacs -nw -q"
alias gf="gfortran -g"
alias cr="cp -r"
alias sdb4="sudo umount /dev/sdb4"   # for u disk
alias sdb1="sudo umount /dev/sdb1"   #  for the elements
alias vo="alsamixer"
alias tel="rlwrap telnet"
alias mp="mplayer"
alias ev="evince -f"
alias detect="sudo tcpdump -i p3p1 -s 0 -X 'tcp and port 80' -w urlcap"
alias din="sudo dnf install"
alias drm="sudo dnf erase"
alias sr="source"
alias mpl="mplayer  -shuffle -loop 0 -playlist"
alias mpv="mplayer -volume"
alias down="wget -U Mozilla"
alias doc="libreoffice"
alias gw="gwenview"
alias px="ps aux|grep -i"
alias wl="wc -l"
alias scr="scp -r"
alias rmd="rm -fr"
alias ori="TreeMaker"
alias du="du -h"
alias df="df -h"
alias ..="cd .."
alias ...="cd ../.."
alias du="du -h"
alias guan="sudo shutdown -h now"
alias shut="sudo shutdown -h"
alias gp="gnuplot"
alias pdf="pdflatex"
alias ud="sudo mount /dev/sdb4 /home/hdeping/disk/udisk"
alias hd="sudo mount /dev/sdb1 /home/hdeping/disk/hdisk"
alias info="info --vi-keys"
alias pxt="pdftotext"
alias x="xbacklight -set"
alias fig="find|grep -i"
alias gi="gvim"
alias del="sudo umount"
alias tgz="tar zcvf"
alias bz2="tar jcvf"
alias win="VirtualBox"
alias m="make"
alias g="gvim *f90"
alias dup="sudo dnf  --exclude=kernel* -y update"
alias math="cp -r /home/hdeping/tex/sample"
alias f="vi *f90 makefile"
alias c="vi *c makefile"
alias j="vi *java makefile"
alias x="vi *tex makefile"
alias fnew="cp -r /home/hdeping/fortran/sample"
alias cnew="cp -r /home/hdeping/c/sample"
alias jnew="cp -r /home/hdeping/java/sample"
alias pnew="cp -r /home/hdeping/python/sample"
alias gnew="cp -r /home/hdeping/fortran/02_opengl/sample"
alias  mc="make -C"
alias ok="okular"
alias ct="ctags -R"
alias vp="vi figure.mp makefile"
alias dch="dnf search"
alias dls="dnf list"
alias  ns="netstat"
alias html="wkhtmltopdf"
alias td="texdoc"
alias s="vi *sh"
alias wlt="ping wlt.ustc.edu.cn"
alias  l="vim *lisp makefile"
alias  x0="startx -bpp 32 -quiet"
alias  p="vim *py makefile"
alias  jl="vim *jl makefile"
alias  gl="vim ~/downloads/f03gl/source/*f90"
alias  xm="xmodmap ~/.xmodmap"
#alias x1="startx :1 -bpp 32 -quiet"
#alias x2="startx :2 -bpp 32 -quiet"
#alias x3="startx :3 -bpp 32 -quiet"
#alias x4="startx :4 -bpp 32 -quiet"
#alias x5="startx :5 -bpp 32 -quiet"
#}}}
