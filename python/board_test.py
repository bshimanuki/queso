from collections import defaultdict
import functools
import operator
import os
import sys

import imageio
import numpy as np
from typing import DefaultDict, List, Optional

from board import Board
from board_extract import make_board
from clipboard_qt import set_clipboard


if __name__ == '__main__':
    root = os.path.dirname(os.path.dirname(__file__))
    impath = 'smogon.png'
    im = imageio.imread(os.path.join(root, 'img_test', impath))

    board = make_board(im)
    if False:
        # answers_file = os.path.join(root, 'smogon-answers-partial.txt')
        answers_file = os.path.join(root, 'smogon-answers-minus.txt')
        # answers_file = os.path.join(root, 'smogon-answers.txt')
        all_answers = defaultdict(lambda: [None]) # type: DefaultDict[int, List[Optional[str]]]
        for line in open(answers_file):
            for word in line.strip().split():
                if word.isalpha():
                    all_answers[len(word)].append(word.upper())

        for entries in board.entries:
            for entry in entries:
                answers = all_answers[len(entry)]
                scores = [1] * len(answers)
                scores[0] = 10
                entry.set_answers(answers, scores)

    clues_file = os.path.join(root, 'smogon-clues.txt')
    with open(clues_file) as f:
        clues = f.read()
    board.use_clues(clues, weight_for_unknown=10)

    for i in range(30):
        board.update_cells()
        board.update_entries()
    board.update_cells()

    output = board.format(fill=True, number=False, probabilities=False)
    # output = board.format(fill=True, number=False, probabilities=False, reduce_op=operator.itemgetter(1))
    # output = board.format(fill=False, number=False, probabilities=True)
    # output = board.format(fill=False, number=False, probabilities=True, reduce_op=operator.itemgetter(0))
    # output = board.format(fill=False, number=False, probabilities=True, reduce_op=functools.partial(np.max, axis=0))
    set_clipboard(html=output)
