#include <cassert>
#include <cctype>
#include <climits>
#include <cstring>
#include <algorithm>
#include <array>
#include <fstream>
#include <iostream>
#include <sstream>
#include <tuple>
#include <vector>
using namespace std;

constexpr char SEP[] = " ";
constexpr int MIN_LENGTH = 6;
int max_turns;
int skip_word;
int skip_grid;

void init() {
	// max_turns = INT_MAX;
	max_turns = 0;
	skip_word = 0;
	skip_grid = 0;
	// skip_grid = 1;
}

enum Direction {
	NW = 1,
	N = 2,
	NE = 3,
	W = 4,
	NOOP = 5,
	E = 6,
	SW = 7,
	S = 8,
	SE = 9,
};

static inline char itoc(char c) {return (c | 0x40) + 1;}
static inline char ctoi(char c) {return (c & 0x1f) - 1;}
constexpr char TERMINAL_MAKE_FAINT[] = "\033[2m";
constexpr char TERMINAL_MAKE_RED[] = "\033[7;31m";
constexpr char TERMINAL_MAKE_GREEN[] = "\033[7;32m";
constexpr char TERMINAL_RESET[] = "\033[0m";

void make_next(int y, int x, Direction dir, int *ny, int *nx) {
	*ny = y;
	*nx = x;
	switch (dir) {
	case NW:
		--*ny;
		--*nx;
		break;
	case N:
		--*ny;
		break;
	case NE:
		--*ny;
		++*nx;
		break;
	case W:
		--*nx;
		break;
	case E:
		++*nx;
		break;
	case SW:
		++*ny;
		--*nx;
		break;
	case S:
		++*ny;
		break;
	case SE:
		++*ny;
		++*nx;
		break;
	default:
		break;
	}
}

class Result : public vector<tuple<char, Direction, bool>> { // tuple(char, direction, special)
public:
	pair<int, int> start;

	Result(pair<int, int> start={}) : start(start) {}
	~Result() {}

	string word() const {
		ostringstream oss;
		for (auto &tup : *this) {
			if (get<2>(tup)) {
				oss << TERMINAL_MAKE_RED << get<0>(tup) << TERMINAL_RESET;
			} else {
				oss << get<0>(tup);
			}
		}
		return oss.str();
	}
	string word_filtered() const {
		ostringstream oss;
		for (auto &tup : *this) {
			if (!get<2>(tup)) {
				oss << get<0>(tup);
			}
		}
		return oss.str();
	}
	string direction() const {
		string s;
		transform(begin(), end(), back_inserter(s), [](auto &p){return get<1>(p) | 0x30;});
		return s;
	}
	string repr(bool filtered=false) const {
		ostringstream oss;
		oss << (start.first+1) << SEP << (start.second+1) << SEP << "<" << direction() << ">";
		oss << SEP;
		if (filtered) oss << word_filtered();
		else oss << word();
		return oss.str();
	}
};

class Trie {
public:
	bool end_of_word;
	array<Trie*, 26> children;
	Trie() : end_of_word{}, children{} {}
	~Trie() {
		for (Trie *c : children) delete c;
	}

	void add(const char *s) {
		while (*s && !isalpha(*s)) ++s;
		if (*s) {
			const char idx = ctoi(*s);
			if (!children[idx]) children[idx] = new Trie();
			return children[idx]->add(++s);
		} else {
			end_of_word = true;
		}
	}
};

struct WordSearchOptions {
	int max_turns;
	int skip_word;
	int skip_grid;
};

class WordSearch {
public:
	const Trie &dict;
	const vector<vector<char>> &grid;
	vector<vector<bool>> used;
	Result partial;
	vector<Result> results;

	WordSearch(const Trie &dict, const vector<vector<char>> &grid) : dict(dict), grid(grid) {}
	~WordSearch() {}

	inline size_t rows() const {return grid.size();}
	inline size_t cols() const {return grid.front().size();}

	void go(size_t y, size_t x, Direction dir, const Trie *trie, const WordSearchOptions &options) {
		if (y < 0 || y >= rows() || x < 0 || x >= cols()) return;
		if (used[y][x]) return;
		if (dir != NOOP && options.skip_word) {
			for (char i=0; i<trie->children.size(); ++i) {
				if (trie->children[i]) {
					partial.emplace_back(itoc(i), NOOP, true);
					WordSearchOptions next_options(options);
					--next_options.skip_word;
					go(y, x, dir, trie->children[i], next_options);
					partial.pop_back();
				}
			}
		}
		used[y][x] = true;
		partial.emplace_back(grid[y][x], dir, false);
		char idx = ctoi(grid[y][x]);
		Trie *child = trie->children[idx];
		if (child) {
			if (child->end_of_word) {
				if (options.skip_word == 0 && options.skip_grid == 0) {
					results.push_back(partial);
				}
			}
		}
		for (Direction next : {NW, N, NE, W, E, SW, S, SE}) {
			if (options.max_turns == 0 && dir != NOOP && dir != next) continue;
			WordSearchOptions next_options(options);
			if (dir != NOOP && dir != next ) --next_options.max_turns;
			int ny, nx;
			make_next(y, x, next, &ny, &nx);
			if (options.skip_grid && dir != NOOP) {
				get<2>(partial.back()) = true;
				WordSearchOptions _next_options(next_options);
				--_next_options.skip_grid;
				go(ny, nx, next, trie, _next_options);
				get<2>(partial.back()) = false;
			}
			if (child) {
				go(ny, nx, next, child, next_options);
			}
		}
		used[y][x] = false;
		partial.pop_back();
	}

	void search(const WordSearchOptions &options) {
		used = vector<vector<bool>>(rows(), vector<bool>(cols()));
		results.clear();
		for (size_t y=0; y<rows(); ++y) for (size_t x=0; x<cols(); ++x) {
			partial = Result({y, x});
			go(y, x, NOOP, &dict, options);
		}
		sort(results.begin(), results.end(), [](const Result &a, const Result &b){return a.size() > b.size();});
	}

	void print_results(ostream &os=cout) const {
		os << "y" << SEP << "x" << SEP << "direction" << SEP << "word" << endl;
		for (const Result &r : results) {
			os << r.repr(true) << endl;
		}
	}

	void print_grid(ostream &os=cout) const {
		vector<vector<const char*>> colors (rows(), vector<const char*>(cols()));
		// color used letters green
		for (const Result &r : results) {
			int y = r.start.first;
			int x = r.start.second;
			for (auto &tup : r) {
				make_next(y, x, get<1>(tup), &y, &x);
				colors[y][x] = TERMINAL_MAKE_GREEN;
			}
		}
		// color special letters red
		for (const Result &r : results) {
			int y = r.start.first;
			int x = r.start.second;
			for (auto &tup : r) {
				make_next(y, x, get<1>(tup), &y, &x);
				if (get<2>(tup)) {
					colors[y][x] = TERMINAL_MAKE_RED;
				}
			}
		}

		// print
		os << SEP << "x" << SEP;
		for(int x=0; x<cols(); ++x) os << (x+1) << SEP;
		os << endl << "y" << endl;

		for (int y=0; y<rows(); ++y) {
			os << (y+1) << SEP << SEP;

			for(int x=0; x<cols(); ++x) {
				if (colors[y][x]) os << colors[y][x];
				// else os << TERMINAL_MAKE_FAINT;
				os << grid[y][x];
				// if (x < cols() - 1) os << SEP;
				os << SEP;
				os << TERMINAL_RESET;
			}
			os << endl;
		}
	}
};

int main(int argc, char *argv[]) {
	init();
	const char *dict_path = "/usr/share/dict/words";
	switch (argc) {
	case 1:
		break;
	case 2:
		dict_path = argv[1];
		break;
	default:
		throw invalid_argument("USAGE: search [dict.txt] < grid.txt");
	}
	ifstream dict_file(dict_path);
	string w;
	Trie trie;
	while (dict_file >> w) {
		w.erase(remove_if(w.begin(), w.end(), [](char c){return !isalpha(c);}), w.end());
		if (w.size() >= MIN_LENGTH) trie.add(w.c_str());
	}
	vector<vector<char>> grid;
	string line;
	while (getline(cin, line)) {
		grid.emplace_back();
		istringstream iss(line);
		char c;
		while (iss >> c) {
			if (isalpha(c)) {
				c = toupper(c);
				grid.back().push_back(c);
			}
		}
		assert(grid.back().size() == grid.front().size());
	}
	WordSearch ws(trie, grid);

	WordSearchOptions options;
	options.max_turns = max_turns;
	options.skip_word = skip_word;
	options.skip_grid = skip_grid;

	ws.search(options);
	ws.print_grid();
	cout << endl;
	ws.print_results();

}
