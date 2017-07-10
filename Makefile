SRCS = ./src

push:
	rsync -au --delete $(SRCS) hav16@graphic02.doc.ic.ac.uk:~/object_detection
pull:
	rsync -au hav16@graphic02.doc.ic.ac.uk:~/object_detection/log .
