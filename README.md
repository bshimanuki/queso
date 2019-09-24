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
Search a grid for words using a dictionary. Set options in the source file.

#### Usage:
```
$ ./bin/wordsearch [dict.txt] < input.txt
```

Consider piping to `./tools/copy_as_html` to copy in a sheet-pastable format.
