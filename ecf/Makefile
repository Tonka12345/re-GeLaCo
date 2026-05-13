# GeLaCo / ECF NSGA-II — Makefile
# -----------------------------------------------------------------------------
# Assumes ECF was built and installed via:
# git clone https://github.com/djakobovic/ECF.git
# cd ECF
# mkdir build && cd build
# cmake .. -DCMAKE_CXX_FLAGS="-fpermissive"
# cmake --build . --config Release
# echo 'export ECF_ROOT=~/ecf-install' >> ~/.bashrc
# echo 'export LD_LIBRARY_PATH=$ECF_ROOT/lib' >> ~/.bashrc
# source ~/.bashrc

# cp ./libECF.a $ECF_ROOT/lib/
# cp -r ../ECF $ECF_ROOT/include/
#
# Usage:
#   make           # produce ./gelaco_ecf
#   make clean
# -----------------------------------------------------------------------------

ECF_ROOT ?= $(HOME)/ecf-install

CXX      := g++
CXXFLAGS := -std=c++14 -O2 -Wall -Wextra -Wno-unused-parameter \
            -I$(ECF_ROOT)/include
LDFLAGS  := -L$(ECF_ROOT)/lib
LDLIBS   := -lECF -lpthread

SRCS := main.cpp Evaluate.cpp
OBJS := $(SRCS:.cpp=.o)
BIN  := gelaco_ecf

.PHONY: all clean

all: $(BIN)

$(BIN): $(OBJS)
	$(CXX) $(OBJS) -o $@ $(LDFLAGS) $(LDLIBS)

%.o: %.cpp Evaluate.h
	$(CXX) $(CXXFLAGS) -c $< -o $@

clean:
	rm -f $(OBJS) $(BIN)
