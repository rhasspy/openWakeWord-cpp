.PHONY: release debug clean

ARCH := $(shell uname -m)
ONNXDIR := lib/$(ARCH)

release:
	mkdir -p build
	g++ -o build/openwakeword -I$(ONNXDIR)/include -L$(ONNXDIR)/lib -O2 -std=c++20 -Wall -Wextra -Wl,-rpath,'$$ORIGIN' src/main.cpp -lpthread -lonnxruntime
	cp -a $(ONNXDIR)/lib/* build/

clean:
	rm -rf build/ dist/
