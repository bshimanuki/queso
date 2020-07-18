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

class Formatter {
	ostringstream ss;
public:
	template<typename T>
	Formatter& operator<<(const T& value) {
		ss << value;
		return *this;
	}
	operator string() const { return ss.str(); }
	string str() const { return *this; }
};

class GF {
	uint8_t v;
public:
	static constexpr uint16_t A = 2;
	static constexpr uint16_t MOD = 285;
	static constexpr uint16_t ORDER = 256; // order of GF256
	static constexpr uint16_t SUBORDER = 255; // order of multiplicative subgroup
private:
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
public:
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
	GF& operator-=(const GF &rhs) { v ^= rhs.v; return *this; }
	GF& operator+=(const GF &rhs) { return *this -= rhs; }
	GF& operator*=(const GF &rhs) {
		if (v == 0 || rhs.v == 0) return *this = 0;
		return *this = ANTILOG[(LOG[v] + LOG[rhs.v]) % SUBORDER];
	}
	GF& operator/=(const GF &rhs) { return *this *= rhs.inv(); }
	GF operator-() const { return this->inv(); }
	GF inv() const {
		if (v == 0) throw domain_error("inverse of 0");
		return GF(ANTILOG[SUBORDER - LOG[v]]);
	}

	friend GF operator-(const GF &lhs, const GF &rhs);
	friend GF operator+(const GF &lhs, const GF &rhs);
	friend GF operator*(const GF &lhs, const GF &rhs);
	friend GF operator/(const GF &lhs, const GF &rhs);

	explicit operator bool() const { return v; }
	friend ostream &operator<<(ostream &os, const GF &x);
};
GF operator-(const GF &lhs, const GF &rhs) { return GF(lhs) -= rhs; }
GF operator+(const GF &lhs, const GF &rhs) { return GF(lhs) += rhs; }
GF operator*(const GF &lhs, const GF &rhs) { return GF(lhs) *= rhs; }
GF operator/(const GF &lhs, const GF &rhs) { return GF(lhs) /= rhs; }
ostream &operator<<(ostream &os, const GF &x) { return os << x.v; }

// Polynomial = c0 + c1 * x + ... + c{n-1} * x^{n-1}, where Poly[i] = c{i}
class Poly : public vector<GF> {
public:
	template<typename ...Args>
	explicit Poly(Args &&...args) : vector<GF>(forward<Args>(args)...) {}
	Poly(const GF &x) : vector<GF>(1, x) {}
	static Poly mono(size_t n, const GF &c=1) { Poly p = Poly(n); p.push_back(c); return p; }

	int deg() const {
		auto it = rbegin();
		while (it != rend() && !*it) ++it;
		return rend() - it - (int) 1;
	}

	// remove leading 0 coefficients
	void reduce() {
		auto it = rbegin();
		while (it != rend() && !*it) ++it;
		erase(it.base(), end());
	}
	GF operator()(const GF &x) const {
		GF y = 0;
		for (size_t i=0; i<size(); ++i) y += (*this)[i] * x.pow(i);
		return y;
	}

	Poly& operator-=(const Poly &rhs) {
		if (size() < rhs.size()) resize(rhs.size());
		transform(rhs.begin(), rhs.end(), begin(), begin(), minus<GF>());
		return *this;
	}
	Poly& operator+=(const Poly &rhs) { return *this -= rhs; }
	Poly operator-() const { return Poly() - *this; }
private:
	static Poly mul(const Poly &lhs, const Poly &rhs) {
		Poly p(lhs.size() + rhs.size() - 1);
		for (size_t i=0; i<lhs.size(); ++i) {
			for (size_t j=0; j<rhs.size(); ++j) {
				p[i+j] += lhs[i] * rhs[j];
			}
		}
		return p;
	}
	static void div(Poly *r, const Poly &b, Poly *q=nullptr) {
		if (q) q->clear();
		r->reduce();
		if (b.deg() < 0) throw domain_error("divide by empty polynomial");
		while (r->deg() >= b.deg()) {
			GF coef = r->back() / b.back();
			size_t deg = r->deg() - b.deg();
			Poly multiple = (coef * b) << deg;
			if (q) *q += mono(deg, coef);
			*r -= multiple;
			r->reduce();
		}
	}
public:
	Poly& operator*=(const Poly &rhs) { return *this = mul(*this, rhs); }
	Poly& operator/=(const Poly &rhs) { Poly q; div(this, rhs, &q); return *this = q; }
	Poly& operator%=(const Poly &rhs) { div(this, rhs); return *this; }
	Poly& operator<<=(size_t n) { insert(begin(), n, 0); return *this; }
	Poly& operator>>=(size_t n) { erase(begin(), min(begin()+n, end())); return *this; }
	Poly operator<<(size_t n) { Poly p(n); p.insert(p.end(), begin(), end()); return p; }
	Poly operator>>(size_t n) { return Poly(min(begin()+n, end()), end()); }

	friend Poly operator-(const Poly &lhs, const Poly &rhs);
	friend Poly operator+(const Poly &lhs, const Poly &rhs);
	friend Poly operator*(const Poly &lhs, const Poly &rhs);
	friend Poly operator/(const Poly &lhs, const Poly &rhs);
	friend Poly operator%(const Poly &lhs, const Poly &rhs);

	explicit operator bool() const { return any_of(begin(), end(), [](const auto &x){ return x; }); }
	friend ostream &operator<<(ostream &os, const Poly &p);
};
Poly operator-(const Poly &lhs, const Poly &rhs) { return Poly(lhs) -= rhs; }
Poly operator+(const Poly &lhs, const Poly &rhs) { return Poly(lhs) += rhs; }
Poly operator*(const Poly &lhs, const Poly &rhs) { return Poly::mul(lhs, rhs); }
Poly operator/(const Poly &lhs, const Poly &rhs) { return Poly(lhs) /= rhs; }
Poly operator%(const Poly &lhs, const Poly &rhs) { return Poly(lhs) %= rhs; }
ostream &operator<<(ostream &os, const Poly &p) {
	os << "Poly[";
	for (auto it = p.begin(); it != p.end(); ++it) {
		if (it != p.begin()) os << ",";
		os << *it;
	}
	os << "]";
	return os;
}

class Vector : private vector<GF> {
	friend class Matrix;

	template<typename ...Args>
	explicit Vector(Args &&...args) : vector<GF>(forward<Args>(args)...) {}

	auto begin() { return vector<GF>::begin(); }
	auto begin() const { return vector<GF>::begin(); }
	auto end() { return vector<GF>::end(); }
	auto end() const { return vector<GF>::end(); }
	auto rbegin() { return vector<GF>::rbegin(); }
	auto rbegin() const { return vector<GF>::rbegin(); }
	auto rend() { return vector<GF>::rend(); }
	auto rend() const { return vector<GF>::rend(); }
	auto cbegin() const { return vector<GF>::cbegin(); }
	auto cend() const { return vector<GF>::cend(); }
	auto crbegin() const { return vector<GF>::crbegin(); }
	auto crend() const { return vector<GF>::crend(); }
	auto size() const { return vector<GF>::size(); }
	auto empty() const { return vector<GF>::empty(); }
	auto operator[](size_t n) { return vector<GF>::operator[](n); }
	auto operator[](size_t n) const { return vector<GF>::operator[](n); }
	auto data() { return vector<GF>::data(); }
	auto data() const { return vector<GF>::data(); }

	Vector& operator-=(const Vector &rhs) {
		if (size() != rhs.size()) throw domain_error(Formatter() << "vector sizes " << size() << " and " << rhs.size() << " don't match");
		transform(rhs.begin(), rhs.end(), begin(), begin(), minus<GF>());
		return *this;
	}
	Vector& operator+=(const Vector &rhs) { return *this -= rhs; }
	Vector& operator-=(const GF &rhs) { for (GF &x : *this) x -= rhs; return *this; }
	Vector& operator+=(const GF &rhs) { for (GF &x : *this) x += rhs; return *this; }
	Vector& operator*=(const GF &rhs) { for (GF &x : *this) x *= rhs; return *this; }
	Vector& operator/=(const GF &rhs) { for (GF &x : *this) x /= rhs; return *this; }

	friend Vector operator-(const Vector &lhs, const Vector &rhs);
	friend Vector operator+(const Vector &lhs, const Vector &rhs);
	friend Vector operator-(const GF &lhs, const Vector &rhs);
	friend Vector operator-(const Vector &lhs, const GF &rhs);
	friend Vector operator+(const GF &lhs, const Vector &rhs);
	friend Vector operator+(const Vector &lhs, const GF &rhs);
	friend Vector operator*(const GF &lhs, const Vector &rhs);
	friend Vector operator*(const Vector &lhs, const GF &rhs);
	friend Vector operator/(const GF &lhs, const Vector &rhs);
	friend Vector operator/(const Vector &lhs, const GF &rhs);

	friend ostream &operator<<(ostream &os, const Vector &v);
};
Vector operator-(const Vector &lhs, const Vector &rhs) { return Vector(lhs) -= rhs; }
Vector operator+(const Vector &lhs, const Vector &rhs) { return Vector(lhs) += rhs; }
Vector operator-(const GF &lhs, const Vector &rhs) { return Vector(rhs) -= lhs; }
Vector operator-(const Vector &lhs, const GF &rhs) { return Vector(lhs) -= rhs; }
Vector operator+(const GF &lhs, const Vector &rhs) { return Vector(rhs) += lhs; }
Vector operator+(const Vector &lhs, const GF &rhs) { return Vector(lhs) += rhs; }
Vector operator*(const GF &lhs, const Vector &rhs) { return Vector(rhs) *= lhs; }
Vector operator*(const Vector &lhs, const GF &rhs) { return Vector(lhs) *= rhs; }
Vector operator/(const GF &lhs, const Vector &rhs) { return Vector(rhs) /= lhs; }
Vector operator/(const Vector &lhs, const GF &rhs) { return Vector(lhs) /= rhs; }
ostream &operator<<(ostream &os, const Vector &v) {
	os << "Vector[";
	for (auto it = v.begin(); it != v.end(); ++it) {
		if (it != v.begin()) os << ",";
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
		n = grid.size();
		v = n / 4 - 4;
		if (n % 4 != 1 || v < 1 || v > 40) {
			throw runtime_error(Formatter() << "QR code has " << n << " rows but " << n << " is not a valid size");
		}
		for (size_t i=0; i<grid.size(); ++i) {
			if (grid[i].size() != n) {
				throw runtime_error(Formatter() << "QR code has " << n << " rows but row " << (i+1) << " has " << grid[i].size() << " columns");
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
