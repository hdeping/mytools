source += MyCommon.c
source += Occupy.c

object = $(source:.c=.o)

exefile = a.out
run:$(object)
	gcc $(object)  -o $(exefile)
e:
	./$(exefile)
clean:
	rm $(object) $(exefile)

