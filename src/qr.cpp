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
constexpr char UNKNOWN[] = "=?";
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
	static constexpr array<uint16_t, ORDER> LOG = _LOG_ANTILOG_PAIR.first;
	static constexpr array<uint16_t, ORDER> ANTILOG = _LOG_ANTILOG_PAIR.second;
public:
	static uint8_t log(const GF &x) {
		if (x == 0) throw domain_error("can't take log of 0");
		return LOG[x.v];
	}
	static GF antilog(int e) {
		e = ((e % SUBORDER) + SUBORDER) % SUBORDER;
		return ANTILOG[e];
	}

	GF(uint8_t v=0) : v(v) {}
	uint8_t operator()() const { return v; }
	bool operator==(const GF &rhs) const { return v == rhs.v; }
	bool operator!=(const GF &rhs) const { return !(v == rhs.v); }

	GF pow(int e) const {
		if (v == 0) {
			if (e >= 0) return 0;
			throw domain_error("inverse of 0");
		}
		return antilog(log(v) * e);
	}
	GF& operator-=(const GF &rhs) { v ^= rhs.v; return *this; }
	GF& operator+=(const GF &rhs) { return *this -= rhs; }
	GF& operator*=(const GF &rhs) {
		if (v == 0 || rhs.v == 0) return *this = 0;
		return *this = antilog(log(v) + log(rhs.v));
	}
	GF& operator/=(const GF &rhs) { return *this *= rhs.inv(); }
	GF operator-() const { return this->inv(); }
	GF inv() const {
		if (v == 0) throw domain_error("inverse of 0");
		return GF(antilog(SUBORDER - log(v)));
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

	const vector<GF>& vector() const { return *this; }
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


// Polynomial = c0 + c1 * x + ... + c{n-1} * x^{n-1}, where Poly[i] = c{i}
class Poly : public vector<GF> {
public:
	template<typename ...Args>
	explicit Poly(Args &&...args) : vector<GF>(forward<Args>(args)...) {}
	Poly(const GF &x) : vector<GF>{x} {}
	static Poly Mono(size_t n, const GF &c=1) { Poly p = Poly(n); p.push_back(c); return p; }
	static Poly FromBinary(uint64_t bin) {
		Poly p;
		for (size_t i=0; i<64; ++i) {
			if (bin & (1LL << i)) {
				p.resize(i);
				p.back() = 1;
			}
		}
		return p;
	}

	int deg() const {
		auto it = rbegin();
		while (it != rend() && !*it) ++it;
		return rend() - it - (int) 1;
	}
	void set(size_t n, const GF &c=1) {
		if (n >= size()) resize(n);
		(*this)[n] = c;
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
	// get coefficient, even if higher than the order of the polynomial
	GF coef(size_t n) const { return n < size() ? (*this)[n] : 0; }

	uint64_t to_binary() const {
		if (deg() > 63) throw overflow_error(Formatter() << "polynomial with degree " << deg() << " does not fit in a 64 bit integer");
		uint64_t bin = 0;
		for(size_t i=0; i<=size(); ++i) {
			if ((*this)[i] == 1) bin |= 1LL << i;
			else if ((*this)[i]) throw domain_error(Formatter() << "polynomial cannot be converted to binary because the coefficient for x^" << i << " is " << (*this)[i]);
		}
		return bin;
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
			if (q) *q += Mono(deg, coef);
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

	static Poly generator(size_t n) {
		if (n > GF::SUBORDER) throw overflow_error(Formatter() << "cannot create generator for size " << n);
		Poly gen = GF(1);
		for (size_t i=0; i<n; ++i) {
			// A^i + 1
			gen *= Poly(initializer_list<GF>{GF::antilog(i), 1});
		}
		return gen;
	}

	// perform error correction
	// err_codewords: number of error correcting code words
	// erasures: positions of known erasures
	Poly error_correct(size_t err_codewords, const vector<size_t> &erasures={}) const {
		Poly gen = generator(err_codewords);
		Vector syndromes(err_codewords);
		for (size_t i=0; i<err_codewords; ++i) syndromes[i] = (*this)(GF::antilog(i));
		// TODO: erasures
		size_t nu = err_codewords / 2; // number of corrections
		Matrix mat_locator(nu, nu);
		Vector b_locator(mat_locator.m);
		for (size_t i=0; i<mat_locator.m; ++i) {
			for (size_t j=0; j<mat_locator.n; ++j) {
				mat_locator(i, j) = syndromes(i + j);
			}
			b_locator(i) = syndromes(nu + i);
		}
		auto [locator_solvable, locator_coefs] = mat_locator.solve(b_locator);
		if (!locator_solvable) throw runtime_error("could not solve for locator polynomial");
		Poly locator{1}; // initialize with constant term
		// locator polynomial is in reverse order
		locator.insert(locator.end(), locator_coefs.rbegin(), locator_coefs.rend());
		vector<size_t> locations;
		for (size_t i=0; i<GF::SUBORDER; ++i) {
			if (locator(GF::antilog(i).inv()) == 0) locations.push_back(i);
		}
		nu = locations.size();
		Matrix mat_err(err_codewords, nu);
		Vector b_err(mat_err.m);
		for (size_t i=0; i<mat_err.m; ++i) {
			for (size_t j=0; j<mat_err.n; ++j) {
				mat_err(i, j) = syndromes(i + j);
			}
			b_err(i) = syndromes(nu + i);
		}
		auto [err_solvable, err_coefs] = mat_err.solve(b_err);
		if (!err_solvable) throw runtime_error("could not solve for error values");
		Poly err;
		for (size_t i=0; i<nu; ++i) {
			err.set(locations[i], err_coefs[i]);
		}
		return (*this) - err;
	}

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


enum class CellValue {
	BLACK,
	WHITE,
	UNKNOWN,
};
CellValue operator-(const CellValue &v) {
	switch (v) {
	case CellValue::BLACK:
		return CellValue::WHITE;
	case CellValue::WHITE:
		return CellValue::BLACK;
	case CellValue::UNKNOWN:
	default:
		return CellValue::UNKNOWN;
	}
}

class QR {
	vector<vector<CellValue>> grid;
	size_t n;
	int v;

	void validate() const {
		for (size_t i=0; i<grid.size(); ++i) {
			if (grid[i].size() != grid.size()) {
				throw runtime_error(Formatter() << "QR code has " << grid.size() << " rows but row " << (i+1) << " has " << grid[i].size() << " columns");
			}
		}
		if (grid.size() != n) throw runtime_error(Formatter() << "QR code of size " << grid.size() << " does not match specified size of " << n);
		if (v != (int) n / 4 - 4) throw runtime_error(Formatter() << "QR code version number " << v << " does not match grid size " << n);
		if (n % 4 != 1 || v < 1 || v > 40) {
			throw runtime_error(Formatter() << "QR code is " << n << "x" << n << " but " << n << " is not a valid size");
		}
	}

	static const Poly VERSION_GENERATOR_POLYNOMIAL;
	static const Poly FORMAT_GENERATOR_POLYNOMIAL;
	static constexpr int FORMAT_BITS = 15;
	static constexpr uint16_t FORMAT_MASK = 0b101010000010010;
	// offsets in grid
	static constexpr int FINDER_WIDTH = 8;
	static constexpr int ALIGNMENT = 6;
	static constexpr int ALIGNMENT_RADIUS = 2;
	static constexpr int FORMAT_OFFSET = 8;
	static constexpr int FORMAT_SPLIT = 7;
public:
	~QR() {};
	QR(const vector<vector<CellValue>> &grid) : grid(grid) {
		n = grid.size();
		v = n / 4 - 4;
		validate();
	}
	QR(size_t n) : QR(vector<vector<CellValue>>(n, vector<CellValue>(n, CellValue::UNKNOWN))) {}

	array<float, FORMAT_BITS> format_bit_proportions() const {
		auto f = [](CellValue v) -> double {
			switch(v) {
			case CellValue::BLACK:
				return 1;
			case CellValue::WHITE:
				return 0;
			case CellValue::UNKNOWN:
			default:
				return 0.5;
			}
		};
		array<float, FORMAT_BITS> bits = {};
		for (size_t c=0; c<FORMAT_SPLIT; ++c) bits[c] += f(grid[FORMAT_OFFSET][c + (c >= ALIGNMENT)]) / 2;
		for (size_t c=FORMAT_SPLIT; c<FORMAT_BITS; ++c) bits[c] += f(grid[FORMAT_OFFSET][n - (FORMAT_BITS - c)]) / 2;
		for (size_t r=0; r<FORMAT_SPLIT; ++r) bits[r] += f(grid[n - 1 - r][FORMAT_OFFSET]) / 2;
		for (size_t r=FORMAT_SPLIT; r<FORMAT_BITS; ++r) bits[r] += f(grid[FORMAT_BITS - 1 - r + (FORMAT_BITS - 1 - r >= ALIGNMENT)][FORMAT_OFFSET]) / 2;
		return bits;
	}

	uint8_t get_format() const {
		auto bits = format_bit_proportions();
		uint16_t guess = 0;
		for (size_t i=0; i<bits.size(); ++i) {
			if (bits[i] > 0.5) guess |= 1 << i;
		}
		Poly p = Poly::FromBinary(guess);
		p = p.error_correct(10);
		uint64_t format = p.to_binary();
		if (format ^ (format & 0x7fff)) throw runtime_error(Formatter() << "computed invalid format string " << hex << format);
		format ^= FORMAT_MASK;
		return format >> 10;
	}

	void apply_mask(uint8_t mask) {
		auto f = [mask](size_t r, size_t c) {
			switch (mask) {
			case 0:
				return (r + c) % 2 == 0;
			case 1:
				return r % 2 == 0;
			case 2:
				return c % 2 == 0;
			case 3:
				return (r + c) % 3 == 0;
			case 4:
				return (r/2 + c/3) % 2 == 0;
			case 5:
				return (r * c) % 2 + (r * c) % 3 == 0;
			case 6:
				return ((r * c) % 2 + (r * c) % 3) % 2 == 0;
			case 7:
				return ((r + c) % 2 + (r * c) % 3) % 2 == 0;
			default:
				return false;
			}
		};
		for (size_t r=0; r<n; ++r) {
			for (size_t c=0; c<n; ++c) {
				if (is_data_bit(r, c).first) {
					if (f(r, c)) grid[r][c] = -grid[r][c];
				}
			}
		}
	}

	// returns (codewords, erasures), without accounting for interleaving
	// any masking should be done before this is called
	pair<vector<uint8_t>, vector<size_t>> get_codewords() const {
		vector<uint8_t> codewords;
		vector<size_t> erasures;
		uint8_t codeword = 0;
		size_t bit = 0;
		bool erasure = false;
		size_t r = n - 1;
		size_t c = n - 1;
		bool right = true; // state for whether on right side of column pair
		bool up = true; // state for whether on traversing upwards
		while (c >= 0) {
			if (c == ALIGNMENT) {
				// shift and do not change column state
				--c;
				continue;
			}
			auto [is_data, value] = is_data_bit(r, c);
			if (is_data) {
				uint8_t v = 0;
				switch (value) {
				case CellValue::BLACK:
					v = 1;
					break;
				case CellValue::WHITE:
					break;
				case CellValue::UNKNOWN:
					erasure = true;
					break;
				}
				codeword |= v << bit++;
				if (bit == 8) {
					if (erasure) erasures.push_back(codewords.size());
					erasure = false;
					codewords.push_back(codeword);
					codeword = 0;
					bit = 0;
				}
			}
			c += right ? -1 : 1;
			if (!right) {
				if (up ? r == 0 : r == n - 1) {
					up = !up;
					c -= 2;
				} else {
					r += up ? -1 : 1;
				}
			}
			right = !right;
		}
		return {codewords, erasures};
	}

	string solve() const {
		uint8_t format = get_format();
		uint8_t level = format >> 3;
		uint8_t mask = format & 7;
		QR qr(*this); // copy
		qr.apply_mask(mask);
		auto [codewords, erasures] = qr.get_codewords();
		// TODO
		return "";
	}

	bool is_format_bit(size_t r, size_t c) const {
		if (r == ALIGNMENT || c == ALIGNMENT) return false;
		if (r == FORMAT_OFFSET) {
			if (c < FORMAT_SPLIT + 1) return true;
			if (c >= n - (FORMAT_BITS - FORMAT_SPLIT)) return true;
		}
		if (c == FORMAT_OFFSET) {
			if (r >= n - 1 - FORMAT_SPLIT) return true;
			if (r < FORMAT_BITS - FORMAT_SPLIT) return true;
		}
		return false;
	}

	pair<bool, CellValue> is_data_bit(size_t r, size_t c) const {
		CellValue value = function_pattern_value(r, c);
		bool b = value == CellValue::UNKNOWN && !is_format_bit(r, c);
		return {b, value};
	}

	CellValue function_pattern_value(size_t r, size_t c) const {
		// dark module
		if (r == n - FINDER_WIDTH) return CellValue::BLACK;
		// top left finder pattern
		if (r < FINDER_WIDTH && c < FINDER_WIDTH) {
			switch (max(abs((int)r-3), abs((int)c-3))) {
			case 2:
			case 4:
				return CellValue::WHITE;
			default:
				return CellValue::BLACK;
			}
		}
		// top right / bottom left finder patterns
		if (min(r, c) < FINDER_WIDTH && max(r, c) >= n - FINDER_WIDTH) {
			return function_pattern_value(min(r, n - 1 - r), min(c, n - 1 - c));
		}
		// timing patterns
		if (r == ALIGNMENT) return c % 2 ? CellValue::BLACK : CellValue::WHITE;
		if (c == ALIGNMENT) return function_pattern_value(c, r);
		// alignment patterns
		if (v > 1) {
			// this calculates the correct values for the spec, though the spec itself is questionable
			int n_patterns = v / 7 + 2;
			int first_pos = ALIGNMENT;
			int last_pos = n - 1 - ALIGNMENT;
			int diff = last_pos - first_pos;
			int offset = (diff + (n_patterns - 1) / 2) / (n_patterns - 1); // rounded divide
			offset = (offset + 1) >> 1 << 1; // round up to even
			int second_pos = last_pos - (n_patterns - 2) * offset;
			auto f = [second_pos, offset](int x){
				if(x <= ALIGNMENT + ALIGNMENT_RADIUS) return ALIGNMENT;
				return second_pos + (x - second_pos + ALIGNMENT_RADIUS) / offset * offset;
			};
			int vr = f(r);
			int vc = f(c);
			if ((vr != ALIGNMENT && vc != ALIGNMENT) ||
					min(vr, (int) n - vr - 1) != ALIGNMENT ||
					min(vc, (int) n - vc - 1) != ALIGNMENT) {
				int dist = max(abs((int) r - vr), abs((int) c - vc));
				if (dist <= ALIGNMENT_RADIUS) {
					return dist == 1 ? CellValue::WHITE : CellValue::BLACK;
				}
			}
		}
		// version bits
		if (v >= 7) {
			if (min(r, c) < ALIGNMENT && max(r, c) >= n - FINDER_WIDTH - 3) {
				if (r > c) return function_pattern_value(c, r);
				c -= n - FINDER_WIDTH - 3;
				size_t i = r * 3 + c;
				if (i >= 12) {
					// bit directly from version number
					return v & (1 << (i - 12)) ? CellValue::BLACK : CellValue::WHITE;
				} else {
					// bit from error correction
					Poly p = Poly::FromBinary(v) << 12;
					p %= VERSION_GENERATOR_POLYNOMIAL;
					return p.coef(i) ? CellValue::BLACK : CellValue::WHITE;
				}
			}
		}
		// format and data bits
		return CellValue::UNKNOWN;
	}
};
// x^12 + x^11 + x^10 + x^9 + x^8 + x^5 + x^2 + 1
const Poly QR::VERSION_GENERATOR_POLYNOMIAL = Poly::FromBinary(0b1111100100101);
// x^10 + x^8 + x^5 + x^4 + x^2 + x + 1
const Poly QR::FORMAT_GENERATOR_POLYNOMIAL = Poly::FromBinary(0b10100110111);


vector<vector<CellValue>> read_grid(istream &is) {
	unordered_map<char, CellValue> symbol_table;
	for (char c : BLACK) symbol_table[c] = CellValue::BLACK;
	for (char c : WHITE) symbol_table[c] = CellValue::WHITE;
	for (char c : UNKNOWN) symbol_table[c] = CellValue::UNKNOWN;
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
		<< UNKNOWN << "] for unknown.";
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
	cout << qr.solve() << endl;;
}
