CC = g++

# OPT_LVL = -O3
OPT_LVL = -g

CCFLAGS = -std=c++1z -Wall
CCFLAGS += $(OPT_LVL)

SRCDIR = src
INCDIR = include
OBJDIR = obj
BINDIR = bin


SRCS = $(wildcard $(SRCDIR)/*.cpp)
HDRS = $(wildcard $(INCDIR)/*.h)
OBJS := $(patsubst $(SRCDIR)/%.cpp, $(OBJDIR)/%.o, $(SRCS))
DEPS := $(patsubst $(SRCDIR)/%.cpp, $(OBJDIR)/%.d, $(SRCS))
BINS := $(patsubst $(SRCDIR)/%.cpp, $(BINDIR)/%, $(SRCS))

GTK_INC = $(shell pkg-config --cflags gtk+-3.0)
GTK_LIB = $(shell pkg-config --libs gtk+-3.0)
INC = -I./include -I./external/cxxopts/include $(GTK_INC)
LDFLAGS = -lpthread -lz -lm $(GTK_LIB)

$(OBJDIR)/%.o: $(SRCDIR)/%.cpp
	@mkdir -p $(OBJDIR)
	$(CC) -MMD $(CCFLAGS) -o $@ -c $< $(INC)

# $(BINDIR)/%: $(OBJS)
# $(CC) -o $@ $^ $(INC) $(LDFLAGS)
$(BINDIR)/%: $(SRCDIR)/%.cpp
	@mkdir -p $(BINDIR)
	$(CC) $(CCFLAGS) -o $@ $< $(INC) $(LDFLAGS)

.PHONY: clean external

default: $(BINS)

all: external default

external:
	git submodule update --init --recursive

clean:
	rm -rf $(OBJDIR) $(BINDIR)

-include $(DEPS)
