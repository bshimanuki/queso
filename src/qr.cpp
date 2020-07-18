#include <cassert>
#include <cctype>
#include <climits>
#include <cstring>
#include <algorithm>
#include <array>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <tuple>
#include <type_traits>
#include <unordered_map>
#include <utility>
#include <vector>

#include "cxxopts.hpp"

using namespace std;

constexpr char BLACK[] = "1#+*@Xx";
constexpr char WHITE[] = " .0Oo_-";
constexpr char ERASURE[] = "=?";
constexpr char BLOCK[] = "█";

class Options {
public:
	bool all_format_options = false;
	bool debug = false;
	string path = "-";

	~Options() {};
	Options(void*) {};

} options{nullptr};

const string& to_string(const string &s) {return s;}
template <typename T> auto make_value(T &value, bool set_default=!is_same<bool,T>::value) {
	if (set_default) return cxxopts::value<T>(value)->default_value(to_string(value));
	return cxxopts::value<T>(value);
}

struct GF {
	uint8_t v;

	static constexpr uint16_t A = 2;
	static constexpr uint16_t MOD = 285;
	static constexpr uint16_t ORDER = 256; // order of GF256
	static constexpr uint16_t SUBORDER = 255; // order of multiplicative subgroup
	static constexpr auto _LOG_ANTILOG_PAIR = [] {
		array<uint16_t, ORDER> log{}, antilog{};
		uint16_t x = 1;
		for (int i=0; i<SUBORDER; ++i) {
			log[x] = i;
			antilog[i] = x;
			x *= A;
			if (x >= ORDER) x ^= MOD;
		}
		antilog[SUBORDER] = x;
		assert(x == 1);
		return make_pair(log, antilog);
	}();
	static constexpr array<uint16_t, ORDER> LOG = _LOG_ANTILOG_PAIR.first;
	static constexpr array<uint16_t, ORDER> ANTILOG = _LOG_ANTILOG_PAIR.second;

	GF(uint8_t v=0) : v(v) {}
	uint8_t operator()() const { return v; }
	bool operator==(const GF &rhs) const { return v == rhs.v; }
	bool operator!=(const GF &rhs) const { return !(v == rhs.v); }

	GF pow(int e) const {
		if (v == 0) {
			if (e >= 0) return 0;
			throw domain_error("inverse of 0");
		}
		return ANTILOG[((((int) LOG[v] * e) % SUBORDER) + SUBORDER) % SUBORDER];
	}
	GF operator-(const GF &rhs) const { return GF(v ^ rhs.v); }
	GF operator+(const GF &rhs) const { return *this - rhs; }
	GF operator*(const GF &rhs) const {
		if (v == 0 || rhs.v == 0) return 0;
		return GF(ANTILOG[(LOG[v] + LOG[rhs.v]) % SUBORDER]);
	}
	GF operator/(const GF &rhs) const { return *this * rhs.inv(); }
	GF operator-() const { return this->inv(); }
	GF inv() const {
		if (v == 0) throw domain_error("inverse of 0");
		return GF(ANTILOG[SUBORDER - LOG[v]]);
	}
	GF& operator-=(const GF &rhs) { return *this = *this - rhs; }
	GF& operator+=(const GF &rhs) { return *this = *this + rhs; }
	GF& operator*=(const GF &rhs) { return *this = *this * rhs; }
	GF& operator/=(const GF &rhs) { return *this = *this / rhs; }
	operator bool() const { return v; }
	friend ostream &operator<<(ostream &os, const GF &x);
} O = GF(0), I = GF(1);
ostream &operator<<(ostream &os, const GF &x) { return os << x.v; }

// Polynomial = c0 + c1 * x + ... + c{n-1} * x^{n-1}, where Poly[i] = c{i}
class Poly : public vector<GF> {
public:
	template<typename ...Args>
	explicit Poly(Args &&...args) : vector<GF>(forward<Args>(args)...) {}
	Poly(const GF &x) : vector<GF>(1, x) {}
	static Poly mono(size_t n, const GF &c=1) { Poly p = Poly(n); p.push_back(c); return p; }

	size_t deg() const {
		auto it = rbegin();
		while (it != rend() && !*it) ++it;
		return rend() - it;
	}

	// remove leading 0 coefficients
	void reduce() {
		auto it = rbegin();
		while (it != rend() && !*it) ++it;
		erase(it.base(), end());
	}
	GF operator()(const GF &x) const {
		GF y = 0;
		for (size_t i=0; i<size(); ++i) y += this->at(i) * x.pow(i);
		return y;
	}

	Poly operator-(const Poly &rhs) const {
		Poly p(*this);
		if (rhs.size() > p.size()) p.resize(rhs.size());
		transform(rhs.begin(), rhs.end(), p.begin(), p.begin(), minus<GF>());
		return p;
	}
	Poly operator+(const Poly &rhs) const { return *this - rhs; }
	Poly operator-() const { return Poly() - *this; }
	Poly operator*(const Poly &rhs) const {
		Poly p(size() + rhs.size() - 1);
		for (size_t i=0; i<size(); ++i) {
			for (size_t j=0; j<rhs.size(); ++j) {
				p[i+j] += this->at(i) * rhs.at(j);
			}
		}
		return p;
	}
	Poly operator/(const Poly &rhs) const { return divides(*this, rhs).first; }
	Poly operator%(const Poly &rhs) const { return divides(*this, rhs).second; }
	static pair<Poly, Poly> divides(Poly a, Poly b) {
		Poly q;
		Poly &r = a;
		b.reduce();
		r.reduce();
		if (b.empty()) throw domain_error("divide by empty polynomial");
		while (r.deg() >= b.deg()) {
			GF coef = r.back() / b.back();
			Poly factor = mono(r.deg() - b.deg(), coef);
			Poly multiple = b * factor;
			q += factor;
			r -= multiple;
			r.reduce();
		}
		return {q, r};
	}
	Poly operator<<(size_t n) { Poly p(n); p.insert(p.end(), begin(), end()); return p; }
	Poly operator>>(size_t n) { Poly p; if (size() > n) p.insert(p.end(), begin()+n, end()); return p; }
	Poly& operator+=(const Poly &rhs) { return *this = *this + rhs; }
	Poly& operator-=(const Poly &rhs) { return *this = *this - rhs; }
	Poly& operator*=(const Poly &rhs) { return *this = *this * rhs; }
	Poly& operator/=(const Poly &rhs) { return *this = *this / rhs; }
	Poly& operator%=(const Poly &rhs) { return *this = *this % rhs; }
	Poly& operator<<=(size_t n) { return *this = *this << n; }
	Poly& operator>>=(size_t n) { return *this = *this >> n; }
	operator bool() const { return any_of(begin(), end(), [](const auto &x){ return x; }); }
	friend ostream &operator<<(ostream &os, const Poly &p);
};
ostream &operator<<(ostream &os, const Poly &p) {
	os << "Poly[";
	for (auto it = p.begin(); it != p.end(); ++it) {
		if (it != p.begin()) os << ",";
		os << *it;
	}
	os << "]";
	return os;
}

enum class CellValue : uint8_t {
	BLACK,
	WHITE,
	ERASURE,
};

class QR {
	vector<vector<CellValue>> grid;
	size_t n;
	int v;
public:
	~QR() {};
	QR(const vector<vector<CellValue>> &grid) : grid(grid) {
		ostringstream err;
		n = grid.size();
		v = n / 4 - 4;
		if (n % 4 != 1 || v < 1 || v > 40) {
			err << "QR code has " << n << " rows but " << n << " is not a valid size";
			throw runtime_error(err.str());
		}
		for (size_t i=0; i<grid.size(); ++i) {
			if (grid[i].size() != n) {
				err << "QR code has " << n << " rows but row " << (i+1) << " has " << grid[i].size() << " columns";
				throw runtime_error(err.str());
			}
		}
	}
};

vector<vector<CellValue>> read_grid(istream &is) {
	unordered_map<char, CellValue> symbol_table;
	for (char c : BLACK) symbol_table[c] = CellValue::BLACK;
	for (char c : WHITE) symbol_table[c] = CellValue::WHITE;
	for (char c : ERASURE) symbol_table[c] = CellValue::ERASURE;
	vector<vector<CellValue>> grid;
	string s;
	while (getline(is, s)) {
		if (!s.empty()) {
			grid.emplace_back();
			for (char c : s) {
				auto it = symbol_table.find(c);
				if (it != symbol_table.end()) grid.back().push_back(it->second);
			}
		}
	}
	return grid;
}

int main(int argc, char *argv[]) {
	ostringstream helptext;
	helptext << "Attempt to decode a QR code from text input. Input should be lines of symbols, with ["
		<< BLACK << "] for black, ["
		<< WHITE << "] for white, and ["
		<< ERASURE << "] for unknown.";
	cxxopts::Options argparse("qr", helptext.str());
	argparse.add_options()
		("h,help", "help")
		("a,all_format_options", "try all 32 possible format", make_value(options.all_format_options))
		("d,debug", "show debugging steps", make_value(options.debug))
		("path", "text file representing QR code (default: stdin)", make_value(options.path), "PATH")
		;
	argparse.parse_positional({"grid"});
	argparse.allow_unrecognised_options();
	string positional_help = "[PATH]";
	argparse.positional_help(positional_help);

	auto args = argparse.parse(argc, argv);

	if (args.count("help")) {
		cerr << argparse.help({""});
		return 0;
	}

	vector<vector<CellValue>> grid;
	if (options.path == "-") grid = read_grid(cin);
	else {
		ifstream fin(options.path);
		grid = read_grid(fin);
	}

	QR qr(grid);
}
