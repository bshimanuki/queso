# Puzzle Hunt Utils

## Tools

### `copy_as_html`
Shell script to convert stdin to html and copy to the clipboard.

Required:
- `ansi2html` (from `pip`)
- `xclip` (from `apt`)

## Programs

Run `make` to build all programs in `./bin`.

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

Consider piping to `./tools/copy_as_html` to copy to the clipboard in a sheet-pastable format.
