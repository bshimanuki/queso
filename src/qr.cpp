#include <algorithm>
#include <array>
#include <cassert>
#include <cctype>
#include <climits>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <tuple>
#include <type_traits>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "cxxopts.hpp"

constexpr char BLACK[] = "1#+*@Xx";
constexpr char WHITE[] = " .0Oo_-";
constexpr char UNKNOWN[] = "=?";
constexpr char BLOCK[] = "█";


class Options {
public:
	bool all_format_options = false;
	bool debug = false;
	std::string path = "-";

	~Options() {};
	Options(void*) {};

} options{nullptr};


namespace std {
	const std::string& to_string(const std::string &s) {return s;}
}
template <typename T> auto make_value(T &value, bool set_default=!std::is_same<bool,T>::value) {
	if (set_default) return cxxopts::value<T>(value)->default_value(std::to_string(value));
	return cxxopts::value<T>(value);
}


class Formatter {
	std::ostringstream ss;
public:
	template<typename T>
	Formatter& operator<<(const T& value) {
		ss << value;
		return *this;
	}
	operator std::string() const { return ss.str(); }
	std::string str() const { return *this; }
};
template<typename A, typename B>
std::ostream& operator<<(std::ostream& os, const std::pair<A, B> &p) {
	os << "(" << p.first << ", " << p.second << ")";
	return os;
}


struct identity {
	template<typename T>
	constexpr	auto operator()(T&& v) const noexcept { return std::forward<T>(v); }
};


template <uint16_t _ORDER, uint16_t GENERATOR, uint16_t A=2, uint16_t _C=0>
class GF {
	uint8_t v;
public:
	static constexpr uint16_t ORDER = _ORDER;
	static_assert(ORDER <= 256, "Order greater than 256 does not fit in uint8_t");
	static constexpr uint16_t C = _C;
	static constexpr uint16_t SUBORDER = ORDER - 1; // order of multiplicative subgroup
private:
	static constexpr auto _LOG_ANTILOG_PAIR = [] {
		std::array<uint16_t, ORDER> log{}, antilog{};
		uint16_t x = 1;
		for (int i=0; i<SUBORDER; ++i) {
			log[x] = i;
			antilog[i] = x;
			x *= A;
			if (x >= ORDER) x ^= GENERATOR;
		}
		antilog[SUBORDER] = x;
		return make_pair(log, antilog);
	}();
	static constexpr std::array<uint16_t, ORDER> LOG = _LOG_ANTILOG_PAIR.first;
	static constexpr std::array<uint16_t, ORDER> ANTILOG = _LOG_ANTILOG_PAIR.second;
	static_assert(ANTILOG[SUBORDER] == 1);
public:
	static uint8_t log(const GF &x) {
		if (x == 0) throw std::domain_error("can't take log of 0");
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
			throw std::domain_error("inverse of 0");
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
		if (v == 0) throw std::domain_error("inverse of 0");
		return GF(antilog(SUBORDER - log(v)));
	}

	friend GF operator-(const GF &lhs, const GF &rhs) { return GF(lhs) -= rhs; }
	friend GF operator+(const GF &lhs, const GF &rhs) { return GF(lhs) += rhs; }
	friend GF operator*(const GF &lhs, const GF &rhs) { return GF(lhs) *= rhs; }
	friend GF operator/(const GF &lhs, const GF &rhs) { return GF(lhs) /= rhs; }

	explicit operator bool() const { return v; }
	friend std::ostream &operator<<(std::ostream &os, const GF &x) { return os << (int) x(); }
};


template<typename T>
class FixedLengthVector : protected std::vector<T> {
protected:
	template<typename ...Args>
	explicit FixedLengthVector(Args &&...args) : std::vector<T>(std::forward<Args>(args)...) {}
public:
	auto begin() { return std::vector<T>::begin(); }
	auto begin() const { return std::vector<T>::begin(); }
	auto end() { return std::vector<T>::end(); }
	auto end() const { return std::vector<T>::end(); }
	auto rbegin() { return std::vector<T>::rbegin(); }
	auto rbegin() const { return std::vector<T>::rbegin(); }
	auto rend() { return std::vector<T>::rend(); }
	auto rend() const { return std::vector<T>::rend(); }
	auto cbegin() const { return std::vector<T>::cbegin(); }
	auto cend() const { return std::vector<T>::cend(); }
	auto crbegin() const { return std::vector<T>::crbegin(); }
	auto crend() const { return std::vector<T>::crend(); }
	auto size() const { return std::vector<T>::size(); }
	auto empty() const { return std::vector<T>::empty(); }
	auto& operator[](size_t n) { return std::vector<T>::operator[](n); }
	auto& operator[](size_t n) const { return std::vector<T>::operator[](n); }
	auto& at(size_t n) { return std::vector<T>::at(n); }
	auto& at(size_t n) const { return std::vector<T>::at(n); }
	auto& front() { return std::vector<T>::front(); }
	auto& front() const { return std::vector<T>::front(); }
	auto& back() { return std::vector<T>::back(); }
	auto& back() const { return std::vector<T>::back(); }
	auto& data() { return std::vector<T>::data(); }
	auto& data() const { return std::vector<T>::data(); }

	auto& operator()(size_t n) { return at(n); }
	auto& operator()(size_t n) const { return at(n); }
};


template<typename T> class Matrix; // forward declaration
// Vector class that cannot be resized after construction
template <typename T>
class Vector : public FixedLengthVector<T> {
	friend class Matrix<T>;

	void check_same_shape(const Vector &oth) const {
		if (this->size() != oth.size()) throw std::domain_error(Formatter() << "vector sizes " << this->size() << " and " << oth.size() << " don't match");
	}

public:
	template<typename ...Args>
	explicit Vector(Args &&...args) : FixedLengthVector<T>(std::forward<Args>(args)...) {}

	const std::vector<T>& vector() const { return *this; }
	Vector& operator-=(const Vector &rhs) {
		check_same_shape(rhs);
		std::transform(this->begin(), this->end(), rhs.begin(), this->begin(), std::minus<T>());
		return *this;
	}
	Vector& operator+=(const Vector &rhs) { return *this -= rhs; }
	Vector& operator-=(const T rhs) { for (T &x : *this) x -= rhs; return *this; }
	Vector& operator+=(const T rhs) { for (T &x : *this) x += rhs; return *this; }
	Vector& operator*=(const T rhs) { for (T &x : *this) x *= rhs; return *this; }
	Vector& operator/=(const T rhs) { for (T &x : *this) x /= rhs; return *this; }

	friend Vector operator-(const Vector &lhs, const Vector &rhs) { return Vector(lhs) -= rhs; }
	friend Vector operator+(const Vector &lhs, const Vector &rhs) { return Vector(lhs) += rhs; }
	friend T operator*(const Vector &lhs, const Vector &rhs) {
		lhs.check_same_shape(rhs);
		T s = 0;
		for (size_t i=0; i<lhs.size(); ++i) {
			s -= lhs(i) * rhs(i);
		}
		return s;
	}

	friend Vector operator-(const T &lhs, const Vector &rhs) { return Vector(rhs) -= lhs; }
	friend Vector operator-(const Vector &lhs, const T &rhs) { return Vector(lhs) -= rhs; }
	friend Vector operator+(const T &lhs, const Vector &rhs) { return Vector(rhs) += lhs; }
	friend Vector operator+(const Vector &lhs, const T &rhs) { return Vector(lhs) += rhs; }
	friend Vector operator*(const T &lhs, const Vector &rhs) { return Vector(rhs) *= lhs; }
	friend Vector operator*(const Vector &lhs, const T &rhs) { return Vector(lhs) *= rhs; }
	friend Vector operator/(const T &lhs, const Vector &rhs) { return Vector(rhs) /= lhs; }
	friend Vector operator/(const Vector &lhs, const T &rhs) { return Vector(lhs) /= rhs; }

	friend std::ostream &operator<<(std::ostream &os, const Vector &v) {
		os << "[ ";
		for (auto it = v.begin(); it != v.end(); ++it) {
			if (it != v.begin()) os << " ";
			os << std::setw(3) << *it;
		}
		os << " ]";
		return os;
	}
};


template <typename T>
class Matrix : public FixedLengthVector<Vector<T>> {
	void check_same_shape(const Matrix &oth) const {
		if (shape() != oth.shape()) throw std::domain_error(Formatter() << "matrix shapes " << shape() << " and " << oth.shape() << " do not match");
	}
public:
	using Vec = Vector<T>;
	const size_t m, n;

	Matrix(size_t m, size_t n) : FixedLengthVector<Vec>(m, Vec(n)), m{m}, n{n} {}

	std::pair<size_t, size_t> shape() const { return {m, n}; }
	size_t size() const { return m * n; }

	Vec& operator()(size_t i) { return this->FixedLengthVector<Vec>::operator()(i); }
	const Vec& operator()(size_t i) const { return this->FixedLengthVector<Vec>::operator()(i); }
	T& operator()(size_t i, size_t j) { return (*this)(i)(j); }
	const T& operator()(size_t i, size_t j) const { return (*this)(i)(j); }
	Vec& row(size_t i) { return this->at(i); }
	const Vec& row(size_t i) const { return this->at(i); }
	Vec col(size_t j) const {
		Vec cv(m);
		std::transform(this->begin(), this->end(), cv.begin(), [&](const auto &rv){ return rv.at(j); });
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
	std::pair<bool, Vec> solve(const Vec &b) const {
		if (b.size() != m) throw std::domain_error(Formatter() << "vector of size " << b.size() << " does not match column vectors of matrix with shape " << shape());
		Matrix aug(m, n+1);
		for (size_t i=0; i<m; ++i) {
			for (size_t j=0; j<n; ++j) {
				aug(i, j) = (*this)(i,j);
			}
			aug(i, n) = b(i);
		}
		size_t rank = aug.rref();
		Vec solution(n);
		bool solvable = rank == 0 || !(aug(rank - 1).back() && none_of(aug(rank - 1).begin(), aug(rank - 1).end()-1, identity()));
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
		std::transform(this->begin(), this->end(), rhs.begin(), this->begin(), std::minus<Vec>());
		return *this;
	}
	Matrix& operator+=(const Matrix &rhs) { return *this -= rhs; }
	// Matrix and Vec addition / subtraction are row-wise
	Matrix& operator-=(const Vec rhs) { for (auto &v : *this) v -= rhs; return *this; }
	Matrix& operator+=(const Vec rhs) { for (auto &v : *this) v += rhs; return *this; }
	Matrix& operator-=(const T rhs) { for (auto &v : *this) v -= rhs; return *this; }
	Matrix& operator+=(const T rhs) { for (auto &v : *this) v += rhs; return *this; }
	Matrix& operator*=(const T rhs) { for (auto &v : *this) v *= rhs; return *this; }
	Matrix& operator/=(const T rhs) { for (auto &v : *this) v /= rhs; return *this; }

	friend Vec operator*(const Matrix &lhs, const Vec &rhs) {
		Vec cv(lhs.m);
		std::transform(lhs.begin(), lhs.end(), cv.begin(), [&](const auto &rv){ return rv * rhs; });
		return cv;
	}
	friend Vec operator*(const Vec &lhs, const Matrix &rhs) {
		Vec rv(rhs.n);
		for (size_t j=0; j<rv.size(); ++j) {
			rv[j] = lhs * rhs.col(j);
		}
		return rv;
	}
	friend Matrix operator*(const Matrix &lhs, const Matrix &rhs) {
		if (lhs.n != rhs.m) throw std::domain_error(Formatter() << "matrix shapes " << lhs.shape() << " and " << rhs.shape() << " cannot be multiplied");
		Matrix result(lhs.m, rhs.n);
		for (size_t i=0; i<result.m; ++i) {
			for (size_t j=0; j<result.n; ++j) {
				result(i, j) = lhs.row(i) * rhs.col(j);
			}
		}
		return result;
	}

	friend Matrix operator-(const Vec &lhs, const Matrix &rhs) { return Matrix(rhs) -= lhs; }
	friend Matrix operator-(const Matrix &lhs, const Vec &rhs) { return Matrix(lhs) -= rhs; }
	friend Matrix operator+(const Vec &lhs, const Matrix &rhs) { return Matrix(rhs) += lhs; }
	friend Matrix operator+(const Matrix &lhs, const Vec &rhs) { return Matrix(lhs) += rhs; }
	friend Matrix operator-(const T &lhs, const Matrix &rhs) { return Matrix(rhs) -= lhs; }
	friend Matrix operator-(const Matrix &lhs, const T &rhs) { return Matrix(lhs) -= rhs; }
	friend Matrix operator+(const T &lhs, const Matrix &rhs) { return Matrix(rhs) += lhs; }
	friend Matrix operator+(const Matrix &lhs, const T &rhs) { return Matrix(lhs) += rhs; }
	friend Matrix operator*(const T &lhs, const Matrix &rhs) { return Matrix(rhs) *= lhs; }
	friend Matrix operator*(const Matrix &lhs, const T &rhs) { return Matrix(lhs) *= rhs; }
	friend Matrix operator/(const T &lhs, const Matrix &rhs) { return Matrix(rhs) /= lhs; }
	friend Matrix operator/(const Matrix &lhs, const T &rhs) { return Matrix(lhs) /= rhs; }

	friend std::ostream &operator<<(std::ostream &os, const Matrix &m) {
		os << "[ ";
		for (auto it = m.begin(); it != m.end(); ++it) {
			if (it != m.begin()) os << "\n ";
			os << *it;
		}
		os << " ]";
		return os;
	}
};


// Polynomial = c0 + c1 * x + ... + c{n-1} * x^{n-1}, where Poly[i] = c{i}
template<typename T>
class Poly : public std::vector<T> {
public:
	Poly(const T &x) : std::vector<T>{x} {}
	// ugly template patterns to default to above constructor
	explicit Poly() : std::vector<T>() {}
	template<typename First, std::enable_if_t<!std::is_same_v<std::remove_reference_t<std::remove_const_t<First>>, T>, int> = 0>
	explicit Poly(First &&first) : std::vector<T>(std::forward<First>(first)) {}
	template<typename First, typename Second, typename ...Args>
	explicit Poly(First &&first, Second &&second, Args &&...args) : std::vector<T>(std::forward<First>(first), std::forward<Second>(second), std::forward<Args>(args)...) {}
	static Poly Mono(size_t n, const T &c=1) { Poly p = Poly(n); p.push_back(c); return p; }
	static Poly FromBinary(uint64_t bin) {
		Poly p;
		for (size_t i=0; i<64; ++i) {
			if (bin & (1LL << i)) {
				p.resize(i + 1, 0);
				p.back() = 1;
			}
		}
		return p;
	}

	int deg() const {
		auto it = this->rbegin();
		while (it != this->rend() && !*it) ++it;
		return this->rend() - it - (int) 1;
	}
	void set(size_t n, const T &c=1) {
		if (n >= this->size()) {
			if (c == 0) return;
			this->resize(n + 1, 0);
		}
		(*this)[n] = c;
	}

	// remove leading 0 coefficients
	void reduce() {
		auto it = this->rbegin();
		while (it != this->rend() && !*it) ++it;
		this->erase(it.base(), this->end());
	}
	T operator()(const T &x) const {
		T y = 0;
		for (size_t i=0; i<this->size(); ++i) y += (*this)[i] * x.pow(i);
		return y;
	}
	// get coefficient, even if higher than the order of the polynomial
	T coef(size_t n) const { return n < this->size() ? (*this)[n] : 0; }

	uint64_t to_binary() const {
		if (deg() > 63) throw std::overflow_error(Formatter() << "polynomial with degree " << deg() << " does not fit in a 64 bit integer");
		uint64_t bin = 0;
		for(size_t i=0; i<this->size(); ++i) {
			if ((*this)[i] == 1) bin |= 1LL << i;
			else if ((*this)[i]) throw std::domain_error(Formatter() << "polynomial cannot be converted to binary because the coefficient for x^" << i << " is " << (*this)[i]);
		}
		return bin;
	}

	Poly& operator-=(const Poly &rhs) {
		if (this->size() < rhs.size()) this->resize(rhs.size(), 0);
		std::transform(this->begin(), this->begin() + std::min(this->size(), rhs.size()), rhs.begin(), this->begin(), std::minus<T>());
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
		if (b.deg() < 0) throw std::domain_error("divide by empty polynomial");
		while (r->deg() >= b.deg()) {
			T coef = r->back() / b.back();
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
	Poly& operator<<=(size_t n) { this->insert(this->begin(), n, 0); return *this; }
	Poly& operator>>=(size_t n) { this->erase(this->begin(), std::min(this->begin()+n, this->end())); return *this; }
	Poly operator<<(size_t n) { Poly p(n); p.insert(p.end(), this->begin(), this->end()); return p; }
	Poly operator>>(size_t n) { return Poly(std::min(this->begin()+n, this->end()), this->end()); }

	friend Poly operator-(const Poly &lhs, const Poly &rhs) { return Poly(lhs) -= rhs; }
	friend Poly operator+(const Poly &lhs, const Poly &rhs) { return Poly(lhs) += rhs; }
	friend Poly operator*(const Poly &lhs, const Poly &rhs) { return Poly::mul(lhs, rhs); }
	friend Poly operator/(const Poly &lhs, const Poly &rhs) { return Poly(lhs) /= rhs; }
	friend Poly operator%(const Poly &lhs, const Poly &rhs) { return Poly(lhs) %= rhs; }

	static Poly generator(size_t n) {
		if (n > T::SUBORDER) throw std::overflow_error(Formatter() << "cannot create generator for size " << n);
		Poly gen = T(1);
		for (size_t i=0; i<n; ++i) {
			// A^i + 1
			gen *= Poly(std::initializer_list<T>{T::antilog(i), 1});
		}
		return gen;
	}

	// perform error correction
	// num_syndromes: the hamming distance (equal to the number of error correcting words in a perfect code)
	// erasures: positions of known erasures
	Poly error_correct(size_t num_syndromes, const std::vector<size_t> &erasures={}) const {
		Vector<T> syndromes(num_syndromes);
		bool has_errors = false;
		for (size_t i=0; i<num_syndromes; ++i) {
			syndromes[i] = (*this)(T::antilog(i + T::C));
			has_errors |= (bool) syndromes[i];
		}
		if (!has_errors) return *this; // short-circuit
		if (erasures.size() > num_syndromes) throw std::runtime_error("more erasures than error correction redundancy");
		// max number of corrections
		const size_t s = erasures.size();
		const size_t e = (num_syndromes - s) / 2;
		size_t nu = s + e;
		Matrix<T> mat_locator(nu, nu);
		Vector<T> b_locator(mat_locator.m);
		for (size_t i=0; i<mat_locator.m; ++i) {
			if (i < e) {
				for (size_t j=0; j<mat_locator.n; ++j) {
					mat_locator(i, j) = syndromes(i + j);
				}
				b_locator(i) = syndromes(nu + i);
			} else {
				const size_t ii = i - e;
				for (size_t j=0; j<mat_locator.n; ++j) {
					mat_locator(i, j) = T::antilog(erasures[ii]).inv().pow(nu - j);
				}
				b_locator(i) = 1;
			}
		}
		auto [locator_solvable, locator_coefs] = mat_locator.solve(b_locator);
		if (!locator_solvable) throw std::runtime_error("could not solve for locator polynomial");
		Poly locator = T(1); // initialize with constant term
		// locator polynomial is in reverse order
		locator.insert(locator.end(), locator_coefs.rbegin(), locator_coefs.rend());
		std::vector<size_t> locations;
		for (size_t i=0; i<T::SUBORDER; ++i) {
			if (locator(T::antilog(i).inv()) == 0) locations.push_back(i);
		}
		nu = locations.size();
		if (nu == 0) throw std::runtime_error("locator polynomial does not have roots");
		Matrix<T> mat_err(num_syndromes, nu);
		Vector<T> b_err(mat_err.m);
		for (size_t i=0; i<mat_err.m; ++i) {
			for (size_t j=0; j<mat_err.n; ++j) {
				mat_err(i, j) = T::antilog(locations[j]).pow(i + T::C);
			}
			b_err(i) = syndromes(i);
		}
		auto [err_solvable, err_coefs] = mat_err.solve(b_err);
		if (!err_solvable) throw std::runtime_error("could not solve for error values");
		Poly err;
		for (size_t i=0; i<nu; ++i) {
			err.set(locations[i], err_coefs[i]);
		}
		return (*this) - err;
	}

	explicit operator bool() const { return any_of(this->begin(), this->end(), identity()); }
	friend std::ostream &operator<<(std::ostream &os, const Poly &p) {
		os << "Poly[";
		for (auto it = p.begin(); it != p.end(); ++it) {
			if (it != p.begin()) os << ",";
			os << *it;
		}
		os << "]";
		return os;
	}
};


struct Version {
	struct VersionLevel {
		struct Group {
			size_t blocks;
			size_t datawords;
		};
		char level = 0;
		size_t errorwords; // per block
		std::vector<Group> groups;
		VersionLevel(size_t errorwords, const std::vector<Group> &groups) :
			errorwords{errorwords},
			groups{groups} {}
		VersionLevel(char level, const VersionLevel &v) : VersionLevel(v) {
			this->level = level;
		}
	};
	int version;
	std::vector<size_t> alignments;
	std::array<VersionLevel, 4> levels;
	Version(int version, const std::vector<size_t> &alignments,
			const VersionLevel &l, const VersionLevel &m, const VersionLevel &q, const VersionLevel &h) :
		version{version},
		alignments{alignments},
		levels{
			VersionLevel{'L', l},
			VersionLevel{'M', m},
			VersionLevel{'Q', q},
			VersionLevel{'H', h}} {}
};

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

using GF256 = GF<256, 0b100011101, 2, 0>;
using GF16 = GF<16, 0b10011, 2, 1>;
using GF2 = GF<2, 0b10, 1>;
class QR {
	std::vector<std::vector<CellValue>> grid;
	size_t n;
	int v;

	void validate() const {
		for (size_t i=0; i<grid.size(); ++i) {
			if (grid[i].size() != grid.size()) {
				throw std::runtime_error(Formatter() << "QR code has " << grid.size() << " rows but row " << (i+1) << " has " << grid[i].size() << " columns");
			}
		}
		if (grid.size() != n) throw std::runtime_error(Formatter() << "QR code of size " << grid.size() << " does not match specified size of " << n);
		if (v != (int) n / 4 - 4) throw std::runtime_error(Formatter() << "QR code version number " << v << " does not match grid size " << n);
		if (n % 4 != 1 || v < 1 || v > 40) {
			throw std::runtime_error(Formatter() << "QR code is " << n << "x" << n << " but " << n << " is not a valid size");
		}
	}

	static const Poly<GF2> VERSION_GENERATOR_POLYNOMIAL;
	static const Poly<GF16> FORMAT_GENERATOR_POLYNOMIAL;
	static constexpr int FORMAT_BITS = 15;
	static constexpr uint16_t FORMAT_MASK = 0b101010000010010;
	// bytes
	static constexpr int PAD_BYTES[] = {236, 17};
	static constexpr size_t ALPHANUMERIC_SIZE = 45;
	static constexpr char ALPHANUMERIC_TABLE[ALPHANUMERIC_SIZE + 1] = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ $%*+-./:";
	// offsets in grid
	static constexpr int FINDER_WIDTH = 8;
	static constexpr int ALIGNMENT = 6;
	static constexpr int ALIGNMENT_RADIUS = 2;
	static constexpr int FORMAT_OFFSET = 8;
	static constexpr int FORMAT_SPLIT = 7;
public:
	static const std::vector<Version> VERSIONS;

	~QR() {};
	QR(const std::vector<std::vector<CellValue>> &grid) : grid(grid) {
		n = grid.size();
		v = n / 4 - 4;
		validate();
	}
	QR(size_t n) : QR(std::vector<std::vector<CellValue>>(n, std::vector<CellValue>(n, CellValue::UNKNOWN))) {}

	std::array<float, FORMAT_BITS> format_bit_proportions() const {
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
		std::array<float, FORMAT_BITS> bits = {};
		for (size_t c=0; c<FORMAT_SPLIT; ++c) bits[c] += f(grid[FORMAT_OFFSET][c + (c >= ALIGNMENT)]) / 2;
		for (size_t c=FORMAT_SPLIT; c<FORMAT_BITS; ++c) bits[c] += f(grid[FORMAT_OFFSET][n - (FORMAT_BITS - c)]) / 2;
		for (size_t r=0; r<FORMAT_SPLIT; ++r) bits[r] += f(grid[n - 1 - r][FORMAT_OFFSET]) / 2;
		for (size_t r=FORMAT_SPLIT; r<FORMAT_BITS; ++r) bits[r] += f(grid[FORMAT_BITS - 1 - r + (FORMAT_BITS - 1 - r >= ALIGNMENT)][FORMAT_OFFSET]) / 2;
		std::reverse(bits.begin(), bits.end()); // QR code order is high to low
		return bits;
	}

	uint8_t get_format() const {
		auto bits = format_bit_proportions();
		uint16_t guess = 0;
		std::vector<size_t> erasures;
		for (size_t i=0; i<bits.size(); ++i) {
			if (bits[i] > 0.5) guess |= 1 << i;
			if (0.4 < bits[i] && bits[i] < 0.6) erasures.push_back(i);
		}
		guess ^= FORMAT_MASK;
		Poly p = Poly<GF16>::FromBinary(guess);
		p = p.error_correct(6, erasures);
		uint64_t format = p.to_binary();
		if (format ^ (format & 0x7fff)) throw std::runtime_error(Formatter() << "computed invalid format std::string " << std::hex << format);
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
	std::pair<std::vector<uint8_t>, std::vector<size_t>> get_raw_codewords() const {
		std::vector<uint8_t> codewords;
		std::vector<size_t> erasures;
		uint8_t codeword = 0;
		size_t bit = 8;
		bool erasure = false;
		int r = n - 1;
		int c = n - 1;
		bool right = true; // state for whether on right side of column std::pair
		bool up = true; // state for whether on traversing upwards
		while (c >= 0) {
			if (c == ALIGNMENT) {
				// shift and do not change column state
				--c;
				continue;
			}
			bool is_data = is_data_bit(r, c).first;
			if (is_data) {
				const CellValue &value = grid[r][c];
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
				codeword |= v << --bit;
				if (bit == 0) {
					if (erasure) erasures.push_back(codewords.size());
					erasure = false;
					codewords.push_back(codeword);
					codeword = 0;
					bit = 8;
				}
			}
			c += right ? -1 : 1;
			if (!right) {
				if (up ? r == 0 : r == (int) n - 1) {
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

	const Version::VersionLevel& get_version(uint8_t level) const {
		const Version &versions = VERSIONS.at(v - 1);
		switch (level) {
		case 1:
			return versions.levels[0];
		case 0:
			return versions.levels[1];
		case 3:
			return versions.levels[2];
		case 2:
			return versions.levels[3];
		default:
			throw std::domain_error("invalid QR level");
		}
	}

	// returns {(codewords, erasures), ...} arranged by block
	// any masking should be done before this is called
	std::vector<std::pair<std::vector<uint8_t>, std::vector<size_t>>> get_codewords(const Version::VersionLevel &version) const {
		auto [_raw_codewords, _raw_erasures] = get_raw_codewords();
		auto &raw_codewords = _raw_codewords; // needed to bind in lambda
		std::unordered_set<size_t> raw_erasures(_raw_erasures.begin(), _raw_erasures.end());
		size_t num_blocks = 0;
		for (const auto &group : version.groups) {
			num_blocks += group.blocks;
		}
		std::vector<std::pair<std::vector<uint8_t>, std::vector<size_t>>> blockdata(num_blocks);
		size_t done = 0;
		size_t iteration = 0;
		size_t codeword_i = 0;
		auto next = [&codeword_i, &raw_codewords]() {
			if (codeword_i == raw_codewords.size()) throw std::out_of_range(Formatter() << "out of range when attempting to read past " << raw_codewords.size() << " codewords");
			return codeword_i++;
		};
		// data codewords
		while (done < num_blocks) {
			size_t block_i = 0;
			for (const auto &group : version.groups) {
				for (size_t block=0; block<group.blocks; ++block) {
					if (iteration < group.datawords) {
						size_t i = next();
						auto& [codewords, erasures] = blockdata[block_i];
						if (raw_erasures.count(i)) erasures.push_back(codewords.size());
						codewords.push_back(raw_codewords[i]);
					} else if (iteration == group.datawords) {
						++done;
					}
					++block_i;
				}
			}
			++iteration;
		}
		// error correction codewords
		for (iteration=0; iteration<version.errorwords; ++iteration) {
			size_t block_i = 0;
			for (const auto &group : version.groups) {
				for (size_t block=0; block<group.blocks; ++block) {
					size_t i = next();
					auto& [codewords, erasures] = blockdata[block_i];
					if (raw_erasures.count(i)) erasures.push_back(codewords.size());
					codewords.push_back(raw_codewords[i]);
					++block_i;
				}
			}
		}
		return blockdata;
	}

	std::string solve() const {
		uint8_t format = get_format();
		uint8_t level = format >> 3;
		uint8_t mask = format & 7;
		const Version::VersionLevel &version = get_version(level);
		QR qr(*this); // copy
		qr.apply_mask(mask);
		auto all_codewords = qr.get_codewords(version);
		std::vector<uint8_t> datawords;
		size_t group_i = 0;
		for (size_t i=0, block_i=0; i<all_codewords.size(); ++i, ++block_i) {
			const auto& [codewords, _erasures] = all_codewords[i];
			if (block_i == version.groups[group_i].blocks) {
				block_i = 0;
				++group_i;
			}
			const auto &versionlevel = version.groups[group_i];
			// first codewords are highest order terms
			Poly<GF256> p(codewords.rbegin(), codewords.rend());
			std::vector<size_t> erasures(_erasures.size());
			std::transform(_erasures.rbegin(), _erasures.rend(), erasures.begin(), [&p](size_t i) { return p.size() - i; });
			p = p.error_correct(version.errorwords, erasures);
			p >>= version.errorwords;
			if (p.size() > versionlevel.datawords) throw std::runtime_error(Formatter() << "error correction yielded bytes at out of range locations");
			p.resize(versionlevel.datawords, 0);
			std::transform(p.rbegin(), p.rend(), std::back_inserter(datawords), [](GF256 x){return x();} );
		}
		return decode(datawords, v);
	}

	static std::string decode(const std::vector<uint8_t> &datawords, size_t version) {
		std::ostringstream os;
		uint64_t buffer = 0;
		size_t bit_length = 0;
		size_t index = 0;
		auto read = [&](size_t n) {
			while (bit_length < n) {
				buffer <<= 8;
				buffer |= datawords.at(index++);
				bit_length += 8;
			}
			uint64_t value = buffer & ((1 << bit_length) - 1);
			value >>= bit_length - n;
			bit_length -= n;
			return value;
		};
		size_t versionset = version < 10 ? 0 : version < 27 ? 1 : 2;
		uint8_t mode = 0;
		size_t length = 0;
		while (index < datawords.size()) {
			if (length) {
				switch (mode) {
				case 0b0001: { // numeric
						size_t digits = std::min((size_t) 3, length);
						uint64_t v = read(3 * digits + 1);
						length -= digits;
						os << std::setfill('0') << std::setw(digits) << v;
					}
					break;
				case 0b0010: { // alphanumeric
						size_t chars = std::min((size_t) 2, length);
						uint64_t v = read(5 * chars + 1);
						length -= chars;
						for (size_t i=0; i<chars; ++i) {
							uint64_t vv = v;
							for (size_t ii=0; ii<chars-i-1; ++ii) vv /= ALPHANUMERIC_SIZE;
							os << ALPHANUMERIC_TABLE[vv % ALPHANUMERIC_SIZE];
						}
					}
					break;
				case 0b0100: { // byte
						char c = read(8);
						os << c;
						--length;
					}
					break;
				case 0b1000: { // kanji
						// character range 0x8140-0x9ffc has offset 0x8140
						// character range 0xe040-0xebbf has offset 0xc140
						wchar_t wc;
						wc = read(13);
						if (wc >= 0x1f00) wc += 0xc140;
						else wc += 0x8140;
						os << wc;
					}
					break;
				default:
					throw std::runtime_error(Formatter() << "unsupported mode " << mode);
					break;
				}
			} else {
				// new mode
				mode = read(4);
				switch (mode) {
				case 0b0000: { // terminator
						// skip to end
						index = datawords.size();
					}
					break;
				case 0b0001: { // numeric
						length = read(std::array{10, 12, 14}.at(versionset));
					}
					break;
				case 0b0010: { // alphanumeric
						length = read(std::array{9, 11, 13}.at(versionset));
					}
					break;
				case 0b0100: { // byte
						length = read(std::array{8, 16, 16}.at(versionset));
					}
					break;
				case 0b1000: { // kanji
						length = read(std::array{8, 10, 12}.at(versionset));
					}
					break;
				case 0b0111: // ECI
				default:
					throw std::runtime_error(Formatter() << "unsupported mode " << mode);
					break;
				}
			}
		}
		return os.str();
	}

	bool is_format_bit(size_t r, size_t c) const {
		if (r == ALIGNMENT || c == ALIGNMENT) return false;
		if (r == FORMAT_OFFSET) {
			if (c < FORMAT_SPLIT + 1) return true;
			if (c >= n - (FORMAT_BITS - FORMAT_SPLIT)) return true;
		}
		if (c == FORMAT_OFFSET) {
			if (r >= n - 1 - FORMAT_SPLIT) return true;
			if (r < FORMAT_BITS - FORMAT_SPLIT + 1) return true;
		}
		return false;
	}

	std::pair<bool, CellValue> is_data_bit(int r, int c) const {
		CellValue value = function_pattern_value(r, c);
		bool b = value == CellValue::UNKNOWN && !is_format_bit(r, c);
		return {b, value};
	}

	CellValue function_pattern_value(int r, int c) const {
		if (r < 0 || c < 0 || r >= (int) n || c >= (int) n) throw std::out_of_range(Formatter() << "(" << r << ", " << c << ") is not in the " << n << "x" << n << " grid");
		// dark module
		if (r == (int) n - FINDER_WIDTH && c == FINDER_WIDTH) return CellValue::BLACK;
		// top left finder pattern
		if (r < FINDER_WIDTH && c < FINDER_WIDTH) {
			switch (std::
					max(abs((int)r-3), abs((int)c-3))) {
			case 2:
			case 4:
				return CellValue::WHITE;
			default:
				return CellValue::BLACK;
			}
		}
		// top right / bottom left finder patterns
		if (std::min(r, c) < FINDER_WIDTH && std::max<int>(r, c) >= (int) n - FINDER_WIDTH) {
			return function_pattern_value(std::min<int>(r, n - 1 - r), std::min<int>(c, n - 1 - c));
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
					std::min(vr, (int) n - vr - 1) != ALIGNMENT ||
					std::min(vc, (int) n - vc - 1) != ALIGNMENT) {
				int dist = std::max(abs((int) r - vr), abs((int) c - vc));
				if (dist <= ALIGNMENT_RADIUS) {
					return dist == 1 ? CellValue::WHITE : CellValue::BLACK;
				}
			}
		}
		// version bits
		if (v >= 7) {
			if (std::min<int>(r, c) < ALIGNMENT && std::max<int>(r, c) >= (int) n - FINDER_WIDTH - 3) {
				if (r > c) return function_pattern_value(c, r);
				c -= n - FINDER_WIDTH - 3;
				size_t i = r * 3 + c;
				if (i >= 12) {
					// bit directly from version number
					return v & (1 << (i - 12)) ? CellValue::BLACK : CellValue::WHITE;
				} else {
					// bit from error correction
					Poly p = Poly<GF2>::FromBinary(v) << 12;
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
const Poly<GF2> QR::VERSION_GENERATOR_POLYNOMIAL = Poly<GF2>::FromBinary(0b1111100100101);
// x^10 + x^8 + x^5 + x^4 + x^2 + x + 1
const Poly<GF16> QR::FORMAT_GENERATOR_POLYNOMIAL = Poly<GF16>::FromBinary(0b10100110111);


std::vector<std::vector<CellValue>> read_grid(std::istream &is) {
	std::unordered_map<char, CellValue> symbol_table;
	for (char c : BLACK) symbol_table[c] = CellValue::BLACK;
	for (char c : WHITE) symbol_table[c] = CellValue::WHITE;
	for (char c : UNKNOWN) symbol_table[c] = CellValue::UNKNOWN;
	std::vector<std::vector<CellValue>> grid;
	std::string s;
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
	std::ostringstream helptext;
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
	argparse.parse_positional({"path"});
	argparse.allow_unrecognised_options();
	std::string positional_help = "[PATH]";
	argparse.positional_help(positional_help);

	auto args = argparse.parse(argc, argv);

	if (args.count("help")) {
		std::cerr << argparse.help({""});
		return 0;
	}

	std::vector<std::vector<CellValue>> grid;
	if (options.path == "-") grid = read_grid(std::cin);
	else {
		std::ifstream fin(options.path);
		grid = read_grid(fin);
	}

	QR qr(grid);
	std::cout << qr.solve() << std::endl;;
}

const std::vector<Version> QR::VERSIONS = {
	Version(1, {},
			{7, {{1, 19}}},
			{10, {{1, 16}}},
			{13, {{1, 13}}},
			{17, {{1, 9}}}),
	Version(2, {6, 18},
			{10, {{1, 34}}},
			{16, {{1, 28}}},
			{22, {{1, 22}}},
			{28, {{1, 16}}}),
	Version(3, {6, 22},
			{15, {{1, 55}}},
			{26, {{1, 44}}},
			{18, {{2, 17}}},
			{22, {{2, 13}}}),
	Version(4, {6, 26},
			{20, {{1, 80}}},
			{18, {{2, 32}}},
			{26, {{2, 24}}},
			{16, {{4, 9}}}),
	Version(5, {6, 30},
			{26, {{1, 108}}},
			{24, {{2, 43}}},
			{18, {{2, 15}, {2, 16}}},
			{22, {{2, 11}, {2, 12}}}),
	Version(6, {6, 34},
			{18, {{2, 68}}},
			{16, {{4, 27}}},
			{24, {{4, 19}}},
			{28, {{4, 15}}}),
	Version(7, {6, 22, 38},
			{20, {{2, 78}}},
			{18, {{4, 31}}},
			{18, {{2, 14}, {4, 15}}},
			{26, {{4, 13}, {1, 14}}}),
	Version(8, {6, 24, 42},
			{24, {{2, 97}}},
			{22, {{2, 38}, {2, 39}}},
			{22, {{4, 18}, {2, 19}}},
			{26, {{4, 14}, {2, 15}}}),
	Version(9, {6, 26, 46},
			{30, {{2, 116}}},
			{22, {{3, 36}, {2, 37}}},
			{20, {{4, 16}, {4, 17}}},
			{24, {{4, 12}, {4, 13}}}),
	Version(10, {6, 28, 50},
			{18, {{2, 68}, {2, 69}}},
			{26, {{4, 43}, {1, 44}}},
			{24, {{6, 19}, {2, 20}}},
			{28, {{6, 15}, {2, 16}}}),
	Version(11, {6, 30, 54},
			{20, {{4, 81}}},
			{30, {{1, 50}, {4, 51}}},
			{28, {{4, 22}, {4, 23}}},
			{24, {{3, 12}, {8, 13}}}),
	Version(12, {6, 32, 58},
			{24, {{2, 92}, {2, 93}}},
			{22, {{6, 36}, {2, 37}}},
			{26, {{4, 20}, {6, 21}}},
			{28, {{7, 14}, {4, 15}}}),
	Version(13, {6, 34, 62},
			{26, {{4, 107}}},
			{22, {{8, 37}, {1, 38}}},
			{24, {{8, 20}, {4, 21}}},
			{22, {{12, 11}, {4, 12}}}),
	Version(14, {6, 26, 46, 66},
			{30, {{3, 115}, {1, 116}}},
			{24, {{4, 40}, {5, 41}}},
			{20, {{11, 16}, {5, 17}}},
			{24, {{11, 12}, {5, 13}}}),
	Version(15, {6, 26, 48, 70},
			{22, {{5, 87}, {1, 88}}},
			{24, {{5, 41}, {5, 42}}},
			{30, {{5, 24}, {7, 25}}},
			{24, {{11, 12}, {7, 13}}}),
	Version(16, {6, 26, 50, 74},
			{24, {{5, 98}, {1, 99}}},
			{28, {{7, 45}, {3, 46}}},
			{24, {{15, 19}, {2, 20}}},
			{30, {{3, 15}, {13, 16}}}),
	Version(17, {6, 30, 54, 78},
			{28, {{1, 107}, {5, 108}}},
			{28, {{10, 46}, {1, 47}}},
			{28, {{1, 22}, {15, 23}}},
			{28, {{2, 14}, {17, 15}}}),
	Version(18, {6, 30, 56, 82},
			{30, {{5, 120}, {1, 121}}},
			{26, {{9, 43}, {4, 44}}},
			{28, {{17, 22}, {1, 23}}},
			{28, {{2, 14}, {19, 15}}}),
	Version(19, {6, 30, 58, 86},
			{28, {{3, 113}, {4, 114}}},
			{26, {{3, 44}, {11, 45}}},
			{26, {{17, 21}, {4, 22}}},
			{26, {{9, 13}, {16, 14}}}),
	Version(20, {6, 34, 62, 90},
			{28, {{3, 107}, {5, 108}}},
			{26, {{3, 41}, {13, 42}}},
			{30, {{15, 24}, {5, 25}}},
			{28, {{15, 15}, {10, 16}}}),
	Version(21, {6, 28, 50, 72, 94},
			{28, {{4, 116}, {4, 117}}},
			{26, {{17, 42}}},
			{28, {{17, 22}, {6, 23}}},
			{30, {{19, 16}, {6, 17}}}),
	Version(22, {6, 26, 50, 74, 98},
			{28, {{2, 111}, {7, 112}}},
			{28, {{17, 46}}},
			{30, {{7, 24}, {16, 25}}},
			{24, {{34, 13}}}),
	Version(23, {6, 30, 54, 78, 102},
			{30, {{4, 121}, {5, 122}}},
			{28, {{4, 47}, {14, 48}}},
			{30, {{11, 24}, {14, 25}}},
			{30, {{16, 15}, {14, 16}}}),
	Version(24, {6, 28, 54, 80, 106},
			{30, {{6, 117}, {4, 118}}},
			{28, {{6, 45}, {14, 46}}},
			{30, {{11, 24}, {16, 25}}},
			{30, {{30, 16}, {2, 17}}}),
	Version(25, {6, 32, 58, 84, 110},
			{26, {{8, 106}, {4, 107}}},
			{28, {{8, 47}, {13, 48}}},
			{30, {{7, 24}, {22, 25}}},
			{30, {{22, 15}, {13, 16}}}),
	Version(26, {6, 30, 58, 86, 114},
			{28, {{10, 114}, {2, 115}}},
			{28, {{19, 46}, {4, 47}}},
			{28, {{28, 22}, {6, 23}}},
			{30, {{33, 16}, {4, 17}}}),
	Version(27, {6, 34, 62, 90, 118},
			{30, {{8, 122}, {4, 123}}},
			{28, {{22, 45}, {3, 46}}},
			{30, {{8, 23}, {26, 24}}},
			{30, {{12, 15}, {28, 16}}}),
	Version(28, {6, 26, 50, 74, 98, 122},
			{30, {{3, 117}, {10, 118}}},
			{28, {{3, 45}, {23, 46}}},
			{30, {{4, 24}, {31, 25}}},
			{30, {{11, 15}, {31, 16}}}),
	Version(29, {6, 30, 54, 78, 102, 126},
			{30, {{7, 116}, {7, 117}}},
			{28, {{21, 45}, {7, 46}}},
			{30, {{1, 23}, {37, 24}}},
			{30, {{19, 15}, {26, 16}}}),
	Version(30, {6, 26, 52, 78, 104, 130},
			{30, {{5, 115}, {10, 116}}},
			{28, {{19, 47}, {10, 48}}},
			{30, {{15, 24}, {25, 25}}},
			{30, {{23, 15}, {25, 16}}}),
	Version(31, {6, 30, 56, 82, 108, 134},
			{30, {{13, 115}, {3, 116}}},
			{28, {{2, 46}, {29, 47}}},
			{30, {{42, 24}, {1, 25}}},
			{30, {{23, 15}, {28, 16}}}),
	Version(32, {6, 34, 60, 86, 112, 138},
			{30, {{17, 115}}},
			{28, {{10, 46}, {23, 47}}},
			{30, {{10, 24}, {35, 25}}},
			{30, {{19, 15}, {35, 16}}}),
	Version(33, {6, 30, 58, 86, 114, 142},
			{30, {{17, 115}, {1, 116}}},
			{28, {{14, 46}, {21, 47}}},
			{30, {{29, 24}, {19, 25}}},
			{30, {{11, 15}, {46, 16}}}),
	Version(34, {6, 34, 62, 90, 118, 146},
			{30, {{13, 115}, {6, 116}}},
			{28, {{14, 46}, {23, 47}}},
			{30, {{44, 24}, {7, 25}}},
			{30, {{59, 16}, {1, 17}}}),
	Version(35, {6, 30, 54, 78, 102, 126, 150},
			{30, {{12, 121}, {7, 122}}},
			{28, {{12, 47}, {26, 48}}},
			{30, {{39, 24}, {14, 25}}},
			{30, {{22, 15}, {41, 16}}}),
	Version(36, {6, 24, 50, 76, 102, 128, 154},
			{30, {{6, 121}, {14, 122}}},
			{28, {{6, 47}, {34, 48}}},
			{30, {{46, 24}, {10, 25}}},
			{30, {{2, 15}, {64, 16}}}),
	Version(37, {6, 28, 54, 80, 106, 132, 158},
			{30, {{17, 122}, {4, 123}}},
			{28, {{29, 46}, {14, 47}}},
			{30, {{49, 24}, {10, 25}}},
			{30, {{24, 15}, {46, 16}}}),
	Version(38, {6, 32, 58, 84, 110, 136, 162},
			{30, {{4, 122}, {18, 123}}},
			{28, {{13, 46}, {32, 47}}},
			{30, {{48, 24}, {14, 25}}},
			{30, {{42, 15}, {32, 16}}}),
	Version(39, {6, 26, 54, 82, 110, 138, 166},
			{30, {{20, 117}, {4, 118}}},
			{28, {{40, 47}, {7, 48}}},
			{30, {{43, 24}, {22, 25}}},
			{30, {{10, 15}, {67, 16}}}),
	Version(40, {6, 30, 58, 86, 114, 142, 170},
			{30, {{19, 118}, {6, 119}}},
			{28, {{18, 47}, {31, 48}}},
			{30, {{34, 24}, {34, 25}}},
			{30, {{20, 15}, {61, 16}}}),
};
