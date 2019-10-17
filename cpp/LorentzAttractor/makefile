CXX = g++ -g
option = -o exe -lm -lGL -lGLU -lglut 
object += opengl.o
object += Lorentz.o
source = $(object:.o=.cpp)
run:$(object)
	$(CXX) $(object) $(option)
	./exe
2:
	gcc -O2 main.c -o exe
	./exe
g:
	$(CC) $(source)
	gdb exe
p:
	gnuplot pic.sh
	gspng
	crop_png pic.png
	rm pic.png
	eog newpic.png
clean:
	rm exe $(object)
#ffmpeg -i %4d.png -vb 250k -maxrate 300k -minrate 200k -bf 5 -b_strategy 0 lorentz.mp4
#ffmpeg -i %4d.png -vb 250k -pass 2 -passlogfile ffmpeg2pass lorentz.mp4
f:
	ffmpeg -i %4d.png -vb 400k lorentz.mp4
