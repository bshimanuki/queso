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
#include <type_traits>
#include <vector>

#include "cxxopts.hpp"

using namespace std;

constexpr char SEP[] = " ";
constexpr int INF = -1;
unsigned min_length = 5;

class WordSearchOptions {
public:
	int max_turns = 0;
	int inserted = 0;
	int removed = 0;
	bool allow_less = false;
	bool card_only = false;

	WordSearchOptions();
	~WordSearchOptions() {};
	WordSearchOptions(void*) {};

} search_options{nullptr};
WordSearchOptions::WordSearchOptions() : WordSearchOptions(search_options) {}

class DisplayOptions {
public:
	bool no_list = false;
	bool no_grid = false;
	bool show_artifacts = false;

	DisplayOptions();
	~DisplayOptions() {};
	DisplayOptions(void*) {};

} display_options{nullptr};
DisplayOptions::DisplayOptions() : DisplayOptions(display_options) {}

const string& to_string(const string &s) {return s;}
template <typename T> auto make_value(T &value, bool set_default=!is_same<bool,T>::value) {
	if (set_default) return cxxopts::value<T>(value)->default_value(to_string(value));
	return cxxopts::value<T>(value);
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

class FormatUInt {
	unsigned i, n;
public:
	FormatUInt(unsigned i, unsigned n) : i(i), n(n) {}
	~FormatUInt() {}
	friend ostream & operator<<(ostream &os, const FormatUInt &self) {
		unsigned i = self.i;
		unsigned n = self.n;
		while (n) {
			if (!i) os << '0';
			i /= 10;
			n /= 10;
		}
		os << self.i;
		return os;
	}
};

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

	string word_artifacts() const {
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
	string word_base() const {
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
	string repr(bool artifacts=false) const {
		ostringstream oss;
		oss << (start.first+1) << SEP << (start.second+1) << SEP << "<" << direction() << ">";
		oss << SEP;
		if (artifacts) oss << word_artifacts();
		else oss << word_base();
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
	inline size_t cols() const {return grid.empty() ? 0 : grid.front().size();}

	void go(size_t y, size_t x, Direction dir, const Trie *trie, const WordSearchOptions &options) {
		if (trie->end_of_word) {
			if (options.allow_less || (options.removed == 0 && options.inserted == 0)) {
				results.push_back(partial);
			}
		}
		if (options.removed) {
			for (size_t i=0; i<trie->children.size(); ++i) {
				Trie *child = trie->children[i];
				if (child) {
					partial.emplace_back(itoc(i), NOOP, true);
					WordSearchOptions next_options(options);
					--next_options.removed;
					go(y, x, dir, child, next_options);
					partial.pop_back();
				}
			}
		}
		if (y < 0 || y >= rows() || x < 0 || x >= cols()) return;
		if (used[y][x]) return;
		used[y][x] = true;
		partial.emplace_back(grid[y][x], dir, false);
		char idx = ctoi(grid[y][x]);
		Trie *child = trie->children[idx];
		for (Direction next : {NW, N, NE, W, E, SW, S, SE}) {
			if (options.max_turns == 0 && dir != NOOP && dir != next) continue;
			if (options.card_only && next & 1) continue;
			WordSearchOptions next_options(options);
			if (dir != NOOP && dir != next ) --next_options.max_turns;
			int ny, nx;
			make_next(y, x, next, &ny, &nx);
			if (options.inserted && dir != NOOP) {
				get<2>(partial.back()) = true;
				WordSearchOptions _next_options(next_options);
				--_next_options.inserted;
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
		stable_sort(results.begin(), results.end(), [](const Result &a, const Result &b){return a.size() > b.size();});
	}

	void print_results(ostream &os=cout) const {
		os << "y" << SEP << "x" << SEP << "direction" << SEP << "word" << endl;
		for (const Result &r : results) {
			os << r.repr(display_options.show_artifacts) << endl;
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
			bool color_next = false;
			for (auto &tup : r) {
				make_next(y, x, get<1>(tup), &y, &x);
				if (get<2>(tup) || color_next) {
					colors[y][x] = TERMINAL_MAKE_RED;
				}
				// highlight before and after removed letter
				color_next = get<2>(tup) && get<1>(tup) == NOOP;
			}
		}

		// print
		os << SEP << "x" << SEP;
		for(size_t x=0; x<cols(); ++x) os << FormatUInt(x + 1, cols()) << SEP;
		os << endl << "y" << endl;

		for (size_t y=0; y<rows(); ++y) {
			os << FormatUInt(y + 1, cols()) << SEP << SEP;

			for(size_t x=0; x<cols(); ++x) {
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

void read_dict(istream &is, Trie *trie) {
	string w;
	while (getline(is, w)) {
		w.erase(remove_if(w.begin(), w.end(), [](char c){return !isalpha(c);}), w.end());
		if (w.size() >= min_length) trie->add(w.c_str());
	}
}

void read_grid(istream &is, vector<vector<char>> *grid) {
	string line;
	while (getline(is, line)) if (!all_of(line.begin(), line.end(), static_cast<int (*)(int)>(isspace))) {
		grid->emplace_back();
		istringstream iss(line);
		char c;
		while (iss >> c) {
			if (isalpha(c)) {
				c = toupper(c);
				grid->back().push_back(c);
			}
		}
		assert(grid->back().size() == grid->front().size());
	}
}

int main(int argc, char *argv[]) {
	string dict_path = "/usr/share/dict/words";
	string grid_path = "-";
	cxxopts::Options options(
			"wordsearch",
			"Find words in a grid. Reads from stdin if GRID_FILE not specified. Prints the\n"
			"grid highlighted with entries followed by a list of results."
			);
	options.add_options()
		("h,help", "help")
		("d,dict", "dictionary file", make_value(dict_path), "DICT_FILE")
		("l,min_length", "min length of words to consider from dictionary", make_value(min_length), "LENGTH")
		("grid", "grid", make_value(grid_path), "GRID_FILE")
		;
	options.parse_positional({"grid"});
	options.allow_unrecognised_options();
	string positional_help = "[GRID_FILE]";
	options.positional_help(positional_help);

	options.add_options("search")
		("max_turns", "max turns to take, -1 for infinite", make_value(search_options.max_turns), "N_TURNS")
		("inserted", "number of letters inserted into words placed in the grid", make_value(search_options.inserted), "N_SKIPS")
		("removed", "number of letters removed from words placed in the grid", make_value(search_options.removed), "N_SKIPS")
		("allow_less", "allow less than the max letters inserted/removed", make_value(search_options.allow_less))
		("card_only", "search in cardinal directions only", make_value(search_options.card_only))
		;
	options.add_options("display")
		("no_list", "do not list results", make_value(display_options.no_list))
		("no_grid", "do not show the grid", make_value(display_options.no_grid))
		("show_artifacts", "show inserted and removed letters in list", make_value(display_options.show_artifacts))
		;

	auto args = options.parse(argc, argv);

	if (args.count("help")) {
		cerr << options.help({"", "search", "display"});
		return 0;
	}

	Trie trie;
	ifstream dict_fin(dict_path);
	read_dict(dict_fin, &trie);
	vector<vector<char>> grid;
	if (grid_path == "-") read_grid(cin, &grid);
	else {
		ifstream grid_fin(grid_path);
		read_grid(grid_fin, &grid);
	}
	WordSearch ws(trie, grid);

	WordSearchOptions opts;

	ws.search(opts);

	cerr << "Found " << ws.results.size() << " results." << endl;

	if (!display_options.no_grid) ws.print_grid();
	if (!display_options.no_grid && !display_options.no_list) cout << endl;
	if (!display_options.no_list) ws.print_results();

}
