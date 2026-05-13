# GeLaCo / ECF NSGA-II — Makefile
# -----------------------------------------------------------------------------
# Assumes ECF was built and installed via:
#   cd $HOME/ecf && mkdir build && cd build
#   cmake .. -DCMAKE_INSTALL_PREFIX=$HOME/ecf-install
#   make -j8 && make install
# and ECF_ROOT is exported in your environment (~/.bashrc):
#   export ECF_ROOT=$HOME/ecf-install
#   export LD_LIBRARY_PATH=$ECF_ROOT/lib:$LD_LIBRARY_PATH
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
LDLIBS   := -lecf -lpthread

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
