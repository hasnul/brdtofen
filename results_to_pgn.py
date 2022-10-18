#!/bin/python3

import argparse
import json
# TODO: Either replace this library with MIT licensed lib or move this file to its own GPL-licensed repo
# import chess


def header(book_title, white, black, result, filename, confidence, fen, pgn):
    print(f'[Event "{book_title}"]', file=pgn)
    print(f'[Site "Book - {filename}"]', file=pgn)
    print(f'[Date "2003.01.01"]', file=pgn)
    print(f'[Round "1"]', file=pgn)
    print(f'[White "{white}"]', file=pgn)
    print(f'[Black "{black}"]', file=pgn)
    print(f'[Result "{result}"]', file=pgn)
    print(f'[FEN "{fen}"]', file=pgn)
    print(f'[Confidence "{confidence:.6f}"]', file=pgn)
    print(file=pgn)
    print('*', file=pgn)
    print(file=pgn)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-k", "--book", help="Book author - title")
    parser.add_argument("results", help="Results file")
    parser.add_argument("pgn", help="Name of pgn file to write")
    args = parser.parse_args()

    if args.book:
        book = args.book
    else:
        book = "Unknown"

    white = "N.N"
    black = "N.N"
    game_result = "1-0"

    status_BAD = 0
    skipped = 0
    with open(args.results, mode='r') as result, open(args.pgn, mode='w') as pgn:
        for line in result:
            r = json.loads(line.strip())
            filename = r['file']
            confidence = r['confidence']
            fen = r['fen']
            if r['status'] == "BAD":
                status_BAD += 1
                fen = " ".join([fen, "w - - 0 1"])
                header(book, "BAD", "BAD", game_result, filename, confidence, fen, pgn)
            else:
                try:
                    # TODO: See todo note at the top
                    # b = chess.Board(fen)
                    header(book, white, black, game_result, filename, confidence, fen, pgn)
                except:
                    skipped += 1

    print(f"BAD = {status_BAD}")
    print(f"Skipped = {skipped}")
   
