#include <cassert>
#include <cctype>
#include <climits>
#include <cstring>
#include <algorithm>
#include <array>
#include <fstream>
#include <iomanip>
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
template<typename A, typename B>
ostream& operator<<(ostream& os, const pair<A, B> &p) {
	os << "(" << p.first << ", " << p.second << ")";
	return os;
}

struct identity {
	template<typename T>
	constexpr	auto operator()(T&& v) const noexcept { return forward<T>(v); }
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
		transform(begin(), end(), rhs.begin(), begin(), minus<GF>());
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

	explicit operator bool() const { return any_of(begin(), end(), identity()); }
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

template<typename T>
class FixedLengthVector : protected vector<T> {
protected:
	template<typename ...Args>
	explicit FixedLengthVector(Args &&...args) : vector<T>(forward<Args>(args)...) {}
public:
	auto begin() { return vector<T>::begin(); }
	auto begin() const { return vector<T>::begin(); }
	auto end() { return vector<T>::end(); }
	auto end() const { return vector<T>::end(); }
	auto rbegin() { return vector<T>::rbegin(); }
	auto rbegin() const { return vector<T>::rbegin(); }
	auto rend() { return vector<T>::rend(); }
	auto rend() const { return vector<T>::rend(); }
	auto cbegin() const { return vector<T>::cbegin(); }
	auto cend() const { return vector<T>::cend(); }
	auto crbegin() const { return vector<T>::crbegin(); }
	auto crend() const { return vector<T>::crend(); }
	auto size() const { return vector<T>::size(); }
	auto empty() const { return vector<T>::empty(); }
	auto& operator[](size_t n) { return vector<T>::operator[](n); }
	auto& operator[](size_t n) const { return vector<T>::operator[](n); }
	auto& at(size_t n) { return vector<T>::at(n); }
	auto& at(size_t n) const { return vector<T>::at(n); }
	auto& front() { return vector<T>::front(); }
	auto& front() const { return vector<T>::front(); }
	auto& back() { return vector<T>::back(); }
	auto& back() const { return vector<T>::back(); }
	auto& data() { return vector<T>::data(); }
	auto& data() const { return vector<T>::data(); }

	auto& operator()(size_t n) { return at(n); }
	auto& operator()(size_t n) const { return at(n); }
};

// Vector class that cannot be resized after construction
class Vector : public FixedLengthVector<GF> {
	friend class Matrix;

	void check_same_shape(const Vector &oth) const {
		if (size() != oth.size()) throw domain_error(Formatter() << "vector sizes " << size() << " and " << oth.size() << " don't match");
	}


public:
	template<typename ...Args>
	explicit Vector(Args &&...args) : FixedLengthVector<GF>(forward<Args>(args)...) {}

	Vector& operator-=(const Vector &rhs) {
		check_same_shape(rhs);
		transform(begin(), end(), rhs.begin(), begin(), minus<GF>());
		return *this;
	}
	Vector& operator+=(const Vector &rhs) { return *this -= rhs; }
	Vector& operator-=(const GF &rhs) { for (GF &x : *this) x -= rhs; return *this; }
	Vector& operator+=(const GF &rhs) { for (GF &x : *this) x += rhs; return *this; }
	Vector& operator*=(const GF &rhs) { for (GF &x : *this) x *= rhs; return *this; }
	Vector& operator/=(const GF &rhs) { for (GF &x : *this) x /= rhs; return *this; }

	friend Vector operator-(const Vector &lhs, const Vector &rhs);
	friend Vector operator+(const Vector &lhs, const Vector &rhs);
	friend GF operator*(const Vector &lhs, const Vector &rhs);

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
GF operator*(const Vector &lhs, const Vector &rhs) {
	lhs.check_same_shape(rhs);
	GF s = 0;
	for (size_t i=0; i<lhs.size(); ++i) {
		s -= lhs[i] * rhs[i];
	}
	return s;
}
Vector operator-(const GF &lhs, const Vector &rhs) { return Vector(rhs) -= lhs; }
Vector operator-(const Vector &lhs, const GF &rhs) { return Vector(lhs) -= rhs; }
Vector operator+(const GF &lhs, const Vector &rhs) { return Vector(rhs) += lhs; }
Vector operator+(const Vector &lhs, const GF &rhs) { return Vector(lhs) += rhs; }
Vector operator*(const GF &lhs, const Vector &rhs) { return Vector(rhs) *= lhs; }
Vector operator*(const Vector &lhs, const GF &rhs) { return Vector(lhs) *= rhs; }
Vector operator/(const GF &lhs, const Vector &rhs) { return Vector(rhs) /= lhs; }
Vector operator/(const Vector &lhs, const GF &rhs) { return Vector(lhs) /= rhs; }
ostream &operator<<(ostream &os, const Vector &v) {
	os << "[";
	for (auto it = v.begin(); it != v.end(); ++it) {
		if (it != v.begin()) os << " ";
		os << setw(3) << *it;
	}
	os << "]";
	return os;
}

class Matrix : public FixedLengthVector<Vector> {
	void check_same_shape(const Matrix &oth) const {
		if (shape() != oth.shape()) throw domain_error(Formatter() << "matrix shapes " << shape() << " and " << oth.shape() << " do not match");
	}
public:
	const size_t m, n;

	Matrix(size_t m, size_t n) : FixedLengthVector<Vector>(m, Vector(n)), m{m}, n{n} {}

	pair<size_t, size_t> shape() const { return {m, n}; }
	size_t size() const { return m * n; }

	Vector& operator()(size_t i) { return this->FixedLengthVector<Vector>::operator()(i); }
	const Vector& operator()(size_t i) const { return this->FixedLengthVector<Vector>::operator()(i); }
	GF& operator()(size_t i, size_t j) { return (*this)(i)(j); }
	const GF& operator()(size_t i, size_t j) const { return (*this)(i)(j); }
	Vector& row(size_t i) { return this->at(i); }
	const Vector& row(size_t i) const { return this->at(i); }
	Vector col(size_t j) const {
		Vector cv(m);
		transform(begin(), end(), cv.begin(), [&](const auto &rv){ return rv.at(j); });
		return cv;
	}

	// reduced row echelon form in place
	// returns rank
	size_t rref() {
		size_t i = 0;
		for (size_t j=0; j<n; ++j) {
			size_t ii = i;
			while (ii < m && (*this)(ii, j) == 0) ++ii;
			if (ii < m) {
				(*this)(i).swap((*this)(ii));
				(*this)(i) /= (*this)(i, j);
				for (auto &rv : (*this)) {
					if (&rv != &(*this)(i)) {
						rv -= rv(j) * (*this)(i);
					}
				}
				++i;
			}
		}
		return i;
	}

	// returns (solvable, solution)
	pair<bool, Vector> solve(const Vector &b) const {
		if (b.size() != n) throw domain_error(Formatter() << "vector of size " << b.size() << " does not match column vectors of matrix with shape " << shape());
		Matrix aug(m, n+1);
		for (size_t i=0; i<m; ++i) {
			for (size_t j=0; j<n; ++j) {
				aug(i, j) = (*this)(i,j);
			}
			aug(i, n) = b(i);
		}
		size_t rank = aug.rref();
		Vector solution(n);
		bool solvable = aug(rank).back() && none_of(aug(rank).begin(), aug(rank).end()-1, identity());
		if (solvable) {
			size_t j = 0;
			for (size_t i=0; i<m; ++i) {
				while (j < n && !aug(i, j)) ++j;
				if (j < n) {
					solution(j) = aug(i, n);
				}
			}
		}
		return {solvable, solution};
	}

	Matrix& operator-=(const Matrix &rhs) {
		check_same_shape(rhs);
		transform(begin(), end(), rhs.begin(), begin(), minus<Vector>());
		return *this;
	}
	Matrix& operator+=(const Matrix &rhs) { return *this -= rhs; }
	// Matrix and Vector addition / subtraction are row-wise
	Matrix& operator-=(const Vector &rhs) { for (auto &v : *this) v -= rhs; return *this; }
	Matrix& operator+=(const Vector &rhs) { for (auto &v : *this) v += rhs; return *this; }
	Matrix& operator-=(const GF &rhs) { for (auto &v : *this) v -= rhs; return *this; }
	Matrix& operator+=(const GF &rhs) { for (auto &v : *this) v += rhs; return *this; }
	Matrix& operator*=(const GF &rhs) { for (auto &v : *this) v *= rhs; return *this; }
	Matrix& operator/=(const GF &rhs) { for (auto &v : *this) v /= rhs; return *this; }

	friend Matrix operator*(const Matrix &lhs, const Matrix &rhs);
	friend Vector operator*(const Matrix &lhs, const Vector &rhs);
	friend Vector operator*(const Vector &lhs, const Matrix &rhs);

	friend Matrix operator-(const Vector &lhs, const Matrix &rhs);
	friend Matrix operator-(const Matrix &lhs, const Vector &rhs);
	friend Matrix operator+(const Vector &lhs, const Matrix &rhs);
	friend Matrix operator+(const Matrix &lhs, const Vector &rhs);
	friend Matrix operator-(const GF &lhs, const Matrix &rhs);
	friend Matrix operator-(const Matrix &lhs, const GF &rhs);
	friend Matrix operator+(const GF &lhs, const Matrix &rhs);
	friend Matrix operator+(const Matrix &lhs, const GF &rhs);
	friend Matrix operator*(const GF &lhs, const Matrix &rhs);
	friend Matrix operator*(const Matrix &lhs, const GF &rhs);
	friend Matrix operator/(const GF &lhs, const Matrix &rhs);
	friend Matrix operator/(const Matrix &lhs, const GF &rhs);

	friend ostream &operator<<(ostream &os, const Matrix &m);
};
Vector operator*(const Matrix &lhs, const Vector &rhs) {
	Vector cv(lhs.m);
	transform(lhs.begin(), lhs.end(), cv.begin(), [&](const auto &rv){ return rv * rhs; });
	return cv;
}
Vector operator*(const Vector &lhs, const Matrix &rhs) {
	Vector rv(rhs.n);
	for (size_t j=0; j<rv.size(); ++j) {
		rv[j] = lhs * rhs.col(j);
	}
	return rv;
}
Matrix operator*(const Matrix &lhs, const Matrix &rhs) {
	if (lhs.n != rhs.m) throw domain_error(Formatter() << "matrix shapes " << lhs.shape() << " and " << rhs.shape() << " cannot be multiplied");
	Matrix result(lhs.m, rhs.n);
	for (size_t i=0; i<result.m; ++i) {
		for (size_t j=0; j<result.n; ++j) {
			result(i, j) = lhs.row(i) * rhs.col(j);
		}
	}
	return result;
}
Matrix operator-(const Vector &lhs, const Matrix &rhs) { return Matrix(rhs) -= lhs; }
Matrix operator-(const Matrix &lhs, const Vector &rhs) { return Matrix(lhs) -= rhs; }
Matrix operator+(const Vector &lhs, const Matrix &rhs) { return Matrix(rhs) += lhs; }
Matrix operator+(const Matrix &lhs, const Vector &rhs) { return Matrix(lhs) += rhs; }
Matrix operator-(const GF &lhs, const Matrix &rhs) { return Matrix(rhs) -= lhs; }
Matrix operator-(const Matrix &lhs, const GF &rhs) { return Matrix(lhs) -= rhs; }
Matrix operator+(const GF &lhs, const Matrix &rhs) { return Matrix(rhs) += lhs; }
Matrix operator+(const Matrix &lhs, const GF &rhs) { return Matrix(lhs) += rhs; }
Matrix operator*(const GF &lhs, const Matrix &rhs) { return Matrix(rhs) *= lhs; }
Matrix operator*(const Matrix &lhs, const GF &rhs) { return Matrix(lhs) *= rhs; }
Matrix operator/(const GF &lhs, const Matrix &rhs) { return Matrix(rhs) /= lhs; }
Matrix operator/(const Matrix &lhs, const GF &rhs) { return Matrix(lhs) /= rhs; }
ostream &operator<<(ostream &os, const Matrix &m) {
	os << "[";
	for (auto it = m.begin(); it != m.end(); ++it) {
		if (it != m.begin()) os << "\n ";
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
