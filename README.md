# Puzzle Hunt Utils

## Tools

### `copy_as_html`
Shell script to convert stdin to html and copy to the clipboard.

Dependencies:
- `ansi2html` (from `pip`)
- `xclip` (from `apt`)

## Programs

Run `make` to build all programs into `./bin`.

Python 3 dependencies are in `requirements.txt`. Run `pip3 install --user -r requirements.txt` to install (or use a virtual environment).

### `wordsearch`
```
Find words in a grid. Reads from stdin if GRID_FILE not specified. Prints the
grid highlighted with entries followed by a list of results.
Usage:
  wordsearch [OPTION...] [GRID_FILE]

  -h, --help               help
  -d, --dict DICT_FILE     dictionary file (default: /usr/share/dict/words)
  -l, --min_length LENGTH  min length of words to consider from dictionary
                           (default: 5)

 search options:
      --max_turns N_TURNS  max turns to take, -1 for infinite (default: 0)
      --inserted N_SKIPS   number of letters inserted into words placed in
                           the grid (default: 0)
      --removed N_SKIPS    number of letters removed from words placed in the
                           grid (default: 0)
      --allow_less         allow less than the max letters inserted/removed
      --card_only          search in cardinal directions only

 display options:
      --no_list         do not list results
      --no_grid         do not show the grid
      --show_artifacts  show inserted and removed letters in list
```

Consider piping to `tools/copy_as_html` to copy to the clipboard in a sheet-pastable format.

### `xword`
```
usage: xword [-h] [--image IMAGE] [--clues CLUES] [--entries ENTRIES]
             [--output OUTPUT] [--clip]

Server for automated crossword solving.

This tracks changes on the clipboard to update source data. After an image is
copied, the server will generate the board as html, which can be pasted into
Google Sheets. After an image and a set of clues are copied, the server will
generate its best guess at filling out the board along with board and clue
information.

The server will keep running in the background, taking whatever is put on the
clipboard, solving crosswords, and putting the results back on the clipboard.
In particular, it does not matter whether the image of the board or the text
of the clues is copied first. Once finished, it is recommended to stop the
server so that it will not change new clipboard contents.

On repeated uses, all crossword answer queries are cached. Additionally, the
last image, clues, generated entry answers, and output are all stored to last-
image.png, etc, under the queso project root. The last inputs are reused by
default. To ignore a file, use the options to set the input file to /dev/null.

implementation details:
This program has two parts, board extraction and crossword solving. A board is
extracted from an image by performing autocorrelation with itself and then
cross-correlation with a constructed grid of the detected size. Crossword
solving is done by scraping a series of online crossword clue databases for
potential answers (via proxies so that three are no rate limits). The answer
scores are aggreated and used as priors for Bayesian inference (using Markov
Random Fields). Belief propagation is performed using the sum-product
algorithm where the values of cells in the crossword are the variables and the
probability distributions over the answer candidates are the factors. Unknown
answers are accounted for by a special answer candidate that uses a smoothed
trigram model over past NYT answers.

optional arguments:
  -h, --help            show this help message and exit
  --image IMAGE, -i IMAGE
                        Image file to read and extract a board from.
  --clues CLUES, -c CLUES
                        Text file to read clues from.
  --entries ENTRIES, -e ENTRIES
                        Text file to read entry scores from.
  --output OUTPUT, -o OUTPUT
                        Output file to write html (same data that is copied to
                        the clipboard).
  --clip                Use the clipboard contents on startup instead of
                        loading from files. (Changes in clipboard contents are
                        always used.)
```

The clue answer candidates are fetched in about a minute, and the rest takes about 10 seconds.
