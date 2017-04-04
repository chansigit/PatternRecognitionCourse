CC = gcc
CFLAGS = -W -Wall -O2 -I. 
LFLAGS = -lpgm

tdtest: tdtest.o td.o ortho.o
	$(CC) $(LFLAGS) -o $@ $^

%.o: %.c %.h
	$(CC) $(CFLAGS) -c $< -o $@

%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

clean: 
	rm -f *.o tdtest *~
