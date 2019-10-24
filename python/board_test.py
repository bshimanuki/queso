from collections import defaultdict
import functools
import operator
import os
import sys
from typing import DefaultDict, List, Optional

import imageio
import numpy as np

from board import Board
from board_extract import make_board
from clipboard_qt import set_clipboard


if __name__ == '__main__':
    root = os.path.dirname(os.path.dirname(__file__))
    impath = 'smogon.png'
    clues_file = os.path.join(root, 'smogon-clues.txt')
    entries_file = os.path.join(root, 'smogon-entries.txt')
    # impath = 'Oct2219.png'
    # clues_file = os.path.join(root, 'Oct2219-clues.txt')
    # entries_file = os.path.join(root, 'Oct2219-entries.txt')
    im = imageio.imread(os.path.join(root, 'img_test', impath))
    board = make_board(im)

    weight_for_unknown = 10
    force_recompile = True
    if not force_recompile and os.path.exists(entries_file):
        with open(entries_file) as f:
            board.load_entries(f.read(), weight_for_unknown=weight_for_unknown)
    else:
        with open(clues_file) as f:
            clues = f.read()
        board.use_clues(clues, weight_for_unknown=weight_for_unknown)
        with open(entries_file, 'w') as f:
            f.write(board.dump_entries())

    for i in range(30):
        board.update_cells()
        board.update_entries()
    board.update_cells()

    output = board.format_multiple()
    set_clipboard(html=output)
