CC = g++
RM = rm

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

INC = -I./include
LDFLAGS = -lpthread -lz -lm

$(OBJDIR)/%.o: $(SRCDIR)/%.cpp
	@mkdir -p $(OBJDIR)
	$(CC) -MMD $(CCFLAGS) $(INC) -o $@ -c $<

# $(BINDIR)/%: $(OBJS)
$(BINDIR)/%: $(SRCDIR)/%.cpp
	@mkdir -p $(BINDIR)
	$(CC) -o $@ $^ $(INC) $(LDFLAGS)

.PHONY: clean

all: $(BINS)

clean:
	$(RM) -rf $(OBJDIR) $(BINDIR)

-include $(DEPS)
