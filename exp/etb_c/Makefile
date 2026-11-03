CC ?= cc
LDFLAGS ?= -I/opt/homebrew/include -L/opt/homebrew/lib -llzma
CFLAGS ?= -O2 -std=c11 -Wall -Wextra -Werror -fno-strict-aliasing
SRC := src/board.c src/symm.c src/layer.c src/gen.c src/pack.c src/main.c
BIN := etb

all: $(BIN)

$(BIN): $(SRC) src/types.h src/symm.h src/layer.h src/gen.h src/pack.h
	$(CC) $(CFLAGS) -o $(BIN) $(SRC) $(LDFLAGS)

test: $(BIN)
	./tests/run_tests.sh

BOT := bot
$(BOT): src/bot.c src/board.c src/types.h
	$(CC) $(CFLAGS) -o $(BOT) src/bot.c src/board.c $(LDFLAGS)

MC := mc_eval
$(MC): src/mc_eval.c src/gen.c src/pack.c src/symm.c src/board.c src/layer.c \
       src/types.h src/gen.h src/pack.h src/symm.h src/layer.h
	$(CC) $(CFLAGS) -o $(MC) src/mc_eval.c src/gen.c src/pack.c src/symm.c src/board.c src/layer.c $(LDFLAGS)

clean:
	rm -f $(BIN) tests/etb
