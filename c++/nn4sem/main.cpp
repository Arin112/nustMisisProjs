#include <bits/stdc++.h>

using namespace std;

class AssertionFail{};

void cstAssert(bool cond, string sCond, int l, string fName){
    if(!cond){
        cerr << "Assertion '"<< sCond<< "' failed at line "<<l<<" in function "<<fName<<endl;
        throw AssertionFail(); // just for throw
    }
}

#define asrt(cond) cstAssert(cond, #cond, __LINE__, __FUNCTION__)

#define all(x) begin(x), end(x)

string to_string(const string& s) {
    return '"' + s + '"';
}
 
string to_string(const char* s) {
    return to_string((string) s);
}
 
string to_string(bool b) {
    return (b ? "true" : "false");
}

template <size_t N>
string to_string(bitset<N> v) {
    return []<size_t... I>(index_sequence<I...>, auto v){
        return (string("") + ... +  static_cast<char>('0' + v[I]));
    }(make_index_sequence<N>{}, v);
}

template <typename A>
string to_string(A v) {
    return accumulate(++begin(v), end(v), "{ "+to_string(*begin(v)),
        [](auto l, auto r){return l+", "+to_string(r);}) + " }";
}
 
template <typename A, typename B>
string to_string(pair<A, B> p) {
    return "(" + to_string(p.first) + ", " + to_string(p.second) + ")";
}

template <typename ... Args, size_t... N>
string to_string_impl_tuple(tuple<Args...> p, index_sequence<N...>){
    return string("(") + ( ((N ? ", " : "") + to_string(get<N>(p)) ) + ... ) + ")";
}
 
template <typename ... Args>
string to_string(tuple<Args...> p){
    return to_string_impl_tuple(p, make_index_sequence<sizeof...(Args)>{});
}
 
template <typename... Tail>
void debug_out(Tail... T) {
  (cerr << ... << (to_string(T) + " ") )<<endl;
}
 
#define dbg(...) cerr << "[" << #__VA_ARGS__ << "]:",  debug_out(__VA_ARGS__)

template<typename Type>
class Matrix{
    public:

    vector<vector<Type>> matr; // yep, non aligned & non flat memory
                            // because rules was use only plain C/C++
                            // so only self written code
    
    Matrix(int n, int m, Type val = 0) // n rows, m columns
        : matr(n, vector<Type>(m, val)){
        asrt(n>0);asrt(m>0);
    }
    
    Matrix(pair<int, int> s, Type val = 0):Matrix(s.first, s.second, val){}
    
    Matrix(const Matrix& mt):matr(mt.matr){}
    
    Matrix& operator=(const Matrix& mt){
        matr = mt.matr;
        return *this;
    }
    
    Matrix(Matrix && mt)
        : matr (std::move(mt.matr)){}
    
    Matrix(vector<vector<Type>> v):matr(v){
        asrt(n()>0);
        asrt(m()>0);
        asrt(all_of(all(matr), [&](const auto& row){return row.size()==m();}));
    }
    
    int n()const{
        return matr.size();
    }
    
    int m()const{
        return matr[0].size();
    }
    
    pair<int, int> shape()const{
        return {n(), m()};
    }
    
    Matrix T(){
        Matrix ans(m(), n());
        for(int i=0;i<n();i++){
            for(int j=0;j<m();j++){
                ans[j][i] = matr[i][j];
            }
        }
        return ans;
    }
    
    vector<Type>& operator[](int idx){
        return matr[idx];
    }
    
    Matrix operator+(const Matrix & mt){
        asrt(shape() == mt.shape()); // check r & c
        Matrix ans(shape());
        for(int i=0;i<n();i++){
            for(int j=0;j<m();j++){
                ans[i][j] = matr[i][j] + mt.matr[i][j];
            }
        }
        return (ans);
    }
    
    Matrix operator-(const Matrix& mt){
        asrt(shape() == mt.shape()); // check r & c
        Matrix ans(shape());
        for(int i=0;i<n();i++){
            for(int j=0;j<m();j++){
                ans[i][j] = matr[i][j] - mt.matr[i][j];
            }
        }
        return (ans);
    }
    
    Matrix operator-(){
        Matrix ans(shape());
        for(int i=0;i<n();i++){
            for(int j=0;j<m();j++){
                ans[i][j] = -matr[i][j];
            }
        }
        return (ans);
    }
    
    Matrix operator*(double d){
        Matrix ans(shape());
        for(int i=0;i<n();i++){
            for(int j=0;j<m();j++){
                ans[i][j] = matr[i][j] * d;
            }
        }
        return (ans);
    }

    Matrix operator/(double d){
        Matrix ans(shape());
        for(int i=0;i<n();i++){
            for(int j=0;j<m();j++){
                ans[i][j] = matr[i][j] / d;
            }
        }
        return (ans);
    }
    
    Matrix operator*(const Matrix &mt){
        asrt(m() == mt.n());
        Matrix ans(n(), mt.m());
        for(int i = 0; i < ans.n(); i++){
            for(int j = 0; j < ans.m(); j++){
                for(int k = 0; k < m(); k++){
                    ans[i][j] += matr[i][k] * mt.matr[k][j];
                }
            }
        }
        return (ans);
    }
    
    Matrix operator^(const Matrix &mt){
        asrt(shape() == mt.shape());
        Matrix ans(n(), m());
        for(int i = 0; i < ans.n(); i++){
            for(int j = 0; j < ans.m(); j++){
                ans[i][j] = matr[i][j] * mt.matr[i][j];
            }
        }
        return (ans);
    }
    
};

template<typename Type>
class NN{
    public:
    vector<Matrix<Type>> w, b;
    
    NN(vector<int> layers){
        asrt(all_of(all(layers), [](int v){return v>0;}));
        for(int i=1;i<layers.size();i++){
            b.push_back(Matrix<Type>(layers[i], 1));
            w.push_back(Matrix<Type>(layers[i], layers[i-1]));
        }
    }
    
    static Matrix<Type> sigm( Matrix<Type> in){
        Matrix<Type> ans (in.shape());
        for(int i=0;i<ans.n();i++){
            for(int j=0;j<ans.m();j++){
                ans[i][j] = 1./(1.+exp(-in[i][j]));
            }
        }
        return ans;
    }
    
    static Matrix<Type> sigmPrime( Matrix<Type> in){
        auto sgm = [](auto v){return 1./(1.+exp(-v));};
        Matrix<Type> ans (in);
        for(int i=0;i<ans.n();i++){
            for(int j=0;j<ans.m();j++){
                ans[i][j] = sgm(in[i][j])*(1-sgm(in[i][j]));
            }
        }
        return ans;
    }
    
    Matrix<Type> operator()(Matrix<Type> in){
        for(int i=0;i<w.size();i++){
            in = sigm(w[i]*in + b[i]);
        }
        return in;
    }
    
    void feed( vector<pair<Matrix<Type>, Matrix<Type>>> data, int n, int batch, double alpha){
        asrt(n > 0);
        asrt(batch <= data.size());
        auto dre = default_random_engine{1338 /*random_device{}()*/};
        for(int i = 0; i < n; i++){
            shuffle(all(data), dre);
            for(int j = 0; j < data.size(); j += batch){
                batch_train(data.cbegin()+j, (j+batch>=data.size())?data.cend():data.cbegin()+(j+batch), alpha);
            }
        }
    }
    
    void batch_train(auto lhs, auto rhs, double alpha){
        vector<Matrix<Type>> nabla_w, nabla_b;
        auto sz = distance(lhs, rhs);
        for(const auto& tw : w)
            nabla_w.push_back(Matrix<Type>(tw.shape()));
        for(const auto& tb : b)
            nabla_b.push_back(Matrix<Type>(tb.shape()));

        for(;lhs!=rhs;lhs++){
            auto [dtNabla_w, dtNabla_b] = backprop(*lhs);
            for(int i=0;i<nabla_w.size();i++){
                nabla_w[i] = nabla_w[i] + dtNabla_w[i];
                nabla_b[i] = nabla_b[i] + dtNabla_b[i];
            }
        }
        for(int i=0;i<w.size();i++){
            w[i] = w[i] - nabla_w[i]*(alpha/sz);
            b[i] = b[i] - nabla_b[i]*(alpha/sz);
        }
    }
    
    pair<vector<Matrix<Type>>, vector<Matrix<Type>>> backprop(pair<Matrix<Type>, Matrix<Type>> xy){
        vector<Matrix<Type>> nabla_w, nabla_b;
        for(const auto& tw : w)
            nabla_w.push_back(Matrix<Type>(tw.shape()));
        for(const auto& tb : b)
            nabla_b.push_back(Matrix<Type>(tb.shape()));
        Matrix<Type> act = xy.first;
        vector<Matrix<Type>> acts{act};
        vector<Matrix<Type>> zs;

        for(int i=0;i<w.size();i++){
            zs.push_back(w[i]*act + b[i]);
            act = sigm(zs.back());
            acts.push_back(act);
        }
        auto dt = (acts.back()-xy.second) ^ sigmPrime(zs.back());

        nabla_b.back() = dt;
        nabla_w.back() = dt * (*(acts.end()-2)).T();
        auto md = [](auto a, auto b){return (b + (a%b)) % b;};
        int sz = w.size();
        for(int i=2;i<sz+1;i++){
            dt = (w[md(-i+1, sz)].T()*dt)^sigmPrime(zs[md(-i, sz)]);
            nabla_b[md(-i, sz)] = dt;
            nabla_w[md(-i, sz)] = dt*(acts[md(-i-1, acts.size())].T());
        }
        return make_pair(nabla_w, nabla_b);
    }
};
template<typename T>
string to_string(const Matrix<T> & m){
    return to_string(m.matr);
}

template<typename T>
string to_string(const NN<T> & nn){
    return to_string(nn.matr);
}

using Matrixd = Matrix<double>;
using NNd = NN<double>;

vector<Matrixd> irisX{{{{5.1, 3.5, 1.4, 0.2}}},{{{4.9, 3.0, 1.4, 0.2}}},{{{4.7, 3.2, 1.3, 0.2}}},{{{4.6, 3.1, 1.5, 0.2}}},{{{5.0, 3.6, 1.4, 0.2}}},{{{5.4, 3.9, 1.7, 0.4}}},{{{4.6, 3.4, 1.4, 0.3}}},{{{5.0, 3.4, 1.5, 0.2}}},{{{4.4, 2.9, 1.4, 0.2}}},{{{4.9, 3.1, 1.5, 0.1}}},{{{5.4, 3.7, 1.5, 0.2}}},{{{4.8, 3.4, 1.6, 0.2}}},{{{4.8, 3.0, 1.4, 0.1}}},{{{4.3, 3.0, 1.1, 0.1}}},{{{5.8, 4.0, 1.2, 0.2}}},{{{5.7, 4.4, 1.5, 0.4}}},{{{5.4, 3.9, 1.3, 0.4}}},{{{5.1, 3.5, 1.4, 0.3}}},{{{5.7, 3.8, 1.7, 0.3}}},{{{5.1, 3.8, 1.5, 0.3}}},{{{5.4, 3.4, 1.7, 0.2}}},{{{5.1, 3.7, 1.5, 0.4}}},{{{4.6, 3.6, 1.0, 0.2}}},{{{5.1, 3.3, 1.7, 0.5}}},{{{4.8, 3.4, 1.9, 0.2}}},{{{5.0, 3.0, 1.6, 0.2}}},{{{5.0, 3.4, 1.6, 0.4}}},{{{5.2, 3.5, 1.5, 0.2}}},{{{5.2, 3.4, 1.4, 0.2}}},{{{4.7, 3.2, 1.6, 0.2}}},{{{4.8, 3.1, 1.6, 0.2}}},{{{5.4, 3.4, 1.5, 0.4}}},{{{5.2, 4.1, 1.5, 0.1}}},{{{5.5, 4.2, 1.4, 0.2}}},{{{4.9, 3.1, 1.5, 0.2}}},{{{5.0, 3.2, 1.2, 0.2}}},{{{5.5, 3.5, 1.3, 0.2}}},{{{4.9, 3.6, 1.4, 0.1}}},{{{4.4, 3.0, 1.3, 0.2}}},{{{5.1, 3.4, 1.5, 0.2}}},{{{5.0, 3.5, 1.3, 0.3}}},{{{4.5, 2.3, 1.3, 0.3}}},{{{4.4, 3.2, 1.3, 0.2}}},{{{5.0, 3.5, 1.6, 0.6}}},{{{5.1, 3.8, 1.9, 0.4}}},{{{4.8, 3.0, 1.4, 0.3}}},{{{5.1, 3.8, 1.6, 0.2}}},{{{4.6, 3.2, 1.4, 0.2}}},{{{5.3, 3.7, 1.5, 0.2}}},{{{5.0, 3.3, 1.4, 0.2}}},{{{7.0, 3.2, 4.7, 1.4}}},{{{6.4, 3.2, 4.5, 1.5}}},{{{6.9, 3.1, 4.9, 1.5}}},{{{5.5, 2.3, 4.0, 1.3}}},{{{6.5, 2.8, 4.6, 1.5}}},{{{5.7, 2.8, 4.5, 1.3}}},{{{6.3, 3.3, 4.7, 1.6}}},{{{4.9, 2.4, 3.3, 1.0}}},{{{6.6, 2.9, 4.6, 1.3}}},{{{5.2, 2.7, 3.9, 1.4}}},{{{5.0, 2.0, 3.5, 1.0}}},{{{5.9, 3.0, 4.2, 1.5}}},{{{6.0, 2.2, 4.0, 1.0}}},{{{6.1, 2.9, 4.7, 1.4}}},{{{5.6, 2.9, 3.6, 1.3}}},{{{6.7, 3.1, 4.4, 1.4}}},{{{5.6, 3.0, 4.5, 1.5}}},{{{5.8, 2.7, 4.1, 1.0}}},{{{6.2, 2.2, 4.5, 1.5}}},{{{5.6, 2.5, 3.9, 1.1}}},{{{5.9, 3.2, 4.8, 1.8}}},{{{6.1, 2.8, 4.0, 1.3}}},{{{6.3, 2.5, 4.9, 1.5}}},{{{6.1, 2.8, 4.7, 1.2}}},{{{6.4, 2.9, 4.3, 1.3}}},{{{6.6, 3.0, 4.4, 1.4}}},{{{6.8, 2.8, 4.8, 1.4}}},{{{6.7, 3.0, 5.0, 1.7}}},{{{6.0, 2.9, 4.5, 1.5}}},{{{5.7, 2.6, 3.5, 1.0}}},{{{5.5, 2.4, 3.8, 1.1}}},{{{5.5, 2.4, 3.7, 1.0}}},{{{5.8, 2.7, 3.9, 1.2}}},{{{6.0, 2.7, 5.1, 1.6}}},{{{5.4, 3.0, 4.5, 1.5}}},{{{6.0, 3.4, 4.5, 1.6}}},{{{6.7, 3.1, 4.7, 1.5}}},{{{6.3, 2.3, 4.4, 1.3}}},{{{5.6, 3.0, 4.1, 1.3}}},{{{5.5, 2.5, 4.0, 1.3}}},{{{5.5, 2.6, 4.4, 1.2}}},{{{6.1, 3.0, 4.6, 1.4}}},{{{5.8, 2.6, 4.0, 1.2}}},{{{5.0, 2.3, 3.3, 1.0}}},{{{5.6, 2.7, 4.2, 1.3}}},{{{5.7, 3.0, 4.2, 1.2}}},{{{5.7, 2.9, 4.2, 1.3}}},{{{6.2, 2.9, 4.3, 1.3}}},{{{5.1, 2.5, 3.0, 1.1}}},{{{5.7, 2.8, 4.1, 1.3}}},{{{6.3, 3.3, 6.0, 2.5}}},{{{5.8, 2.7, 5.1, 1.9}}},{{{7.1, 3.0, 5.9, 2.1}}},{{{6.3, 2.9, 5.6, 1.8}}},{{{6.5, 3.0, 5.8, 2.2}}},{{{7.6, 3.0, 6.6, 2.1}}},{{{4.9, 2.5, 4.5, 1.7}}},{{{7.3, 2.9, 6.3, 1.8}}},{{{6.7, 2.5, 5.8, 1.8}}},{{{7.2, 3.6, 6.1, 2.5}}},{{{6.5, 3.2, 5.1, 2.0}}},{{{6.4, 2.7, 5.3, 1.9}}},{{{6.8, 3.0, 5.5, 2.1}}},{{{5.7, 2.5, 5.0, 2.0}}},{{{5.8, 2.8, 5.1, 2.4}}},{{{6.4, 3.2, 5.3, 2.3}}},{{{6.5, 3.0, 5.5, 1.8}}},{{{7.7, 3.8, 6.7, 2.2}}},{{{7.7, 2.6, 6.9, 2.3}}},{{{6.0, 2.2, 5.0, 1.5}}},{{{6.9, 3.2, 5.7, 2.3}}},{{{5.6, 2.8, 4.9, 2.0}}},{{{7.7, 2.8, 6.7, 2.0}}},{{{6.3, 2.7, 4.9, 1.8}}},{{{6.7, 3.3, 5.7, 2.1}}},{{{7.2, 3.2, 6.0, 1.8}}},{{{6.2, 2.8, 4.8, 1.8}}},{{{6.1, 3.0, 4.9, 1.8}}},{{{6.4, 2.8, 5.6, 2.1}}},{{{7.2, 3.0, 5.8, 1.6}}},{{{7.4, 2.8, 6.1, 1.9}}},{{{7.9, 3.8, 6.4, 2.0}}},{{{6.4, 2.8, 5.6, 2.2}}},{{{6.3, 2.8, 5.1, 1.5}}},{{{6.1, 2.6, 5.6, 1.4}}},{{{7.7, 3.0, 6.1, 2.3}}},{{{6.3, 3.4, 5.6, 2.4}}},{{{6.4, 3.1, 5.5, 1.8}}},{{{6.0, 3.0, 4.8, 1.8}}},{{{6.9, 3.1, 5.4, 2.1}}},{{{6.7, 3.1, 5.6, 2.4}}},{{{6.9, 3.1, 5.1, 2.3}}},{{{5.8, 2.7, 5.1, 1.9}}},{{{6.8, 3.2, 5.9, 2.3}}},{{{6.7, 3.3, 5.7, 2.5}}},{{{6.7, 3.0, 5.2, 2.3}}},{{{6.3, 2.5, 5.0, 1.9}}},{{{6.5, 3.0, 5.2, 2.0}}},{{{6.2, 3.4, 5.4, 2.3}}},{{{5.9, 3.0, 5.1, 1.8}}}};
vector<Matrixd> irisY{{{{1, 0, 0}}},{{{1, 0, 0}}},{{{1, 0, 0}}},{{{1, 0, 0}}},{{{1, 0, 0}}},{{{1, 0, 0}}},{{{1, 0, 0}}},{{{1, 0, 0}}},{{{1, 0, 0}}},{{{1, 0, 0}}},{{{1, 0, 0}}},{{{1, 0, 0}}},{{{1, 0, 0}}},{{{1, 0, 0}}},{{{1, 0, 0}}},{{{1, 0, 0}}},{{{1, 0, 0}}},{{{1, 0, 0}}},{{{1, 0, 0}}},{{{1, 0, 0}}},{{{1, 0, 0}}},{{{1, 0, 0}}},{{{1, 0, 0}}},{{{1, 0, 0}}},{{{1, 0, 0}}},{{{1, 0, 0}}},{{{1, 0, 0}}},{{{1, 0, 0}}},{{{1, 0, 0}}},{{{1, 0, 0}}},{{{1, 0, 0}}},{{{1, 0, 0}}},{{{1, 0, 0}}},{{{1, 0, 0}}},{{{1, 0, 0}}},{{{1, 0, 0}}},{{{1, 0, 0}}},{{{1, 0, 0}}},{{{1, 0, 0}}},{{{1, 0, 0}}},{{{1, 0, 0}}},{{{1, 0, 0}}},{{{1, 0, 0}}},{{{1, 0, 0}}},{{{1, 0, 0}}},{{{1, 0, 0}}},{{{1, 0, 0}}},{{{1, 0, 0}}},{{{1, 0, 0}}},{{{1, 0, 0}}},{{{0, 1, 0}}},{{{0, 1, 0}}},{{{0, 1, 0}}},{{{0, 1, 0}}},{{{0, 1, 0}}},{{{0, 1, 0}}},{{{0, 1, 0}}},{{{0, 1, 0}}},{{{0, 1, 0}}},{{{0, 1, 0}}},{{{0, 1, 0}}},{{{0, 1, 0}}},{{{0, 1, 0}}},{{{0, 1, 0}}},{{{0, 1, 0}}},{{{0, 1, 0}}},{{{0, 1, 0}}},{{{0, 1, 0}}},{{{0, 1, 0}}},{{{0, 1, 0}}},{{{0, 1, 0}}},{{{0, 1, 0}}},{{{0, 1, 0}}},{{{0, 1, 0}}},{{{0, 1, 0}}},{{{0, 1, 0}}},{{{0, 1, 0}}},{{{0, 1, 0}}},{{{0, 1, 0}}},{{{0, 1, 0}}},{{{0, 1, 0}}},{{{0, 1, 0}}},{{{0, 1, 0}}},{{{0, 1, 0}}},{{{0, 1, 0}}},{{{0, 1, 0}}},{{{0, 1, 0}}},{{{0, 1, 0}}},{{{0, 1, 0}}},{{{0, 1, 0}}},{{{0, 1, 0}}},{{{0, 1, 0}}},{{{0, 1, 0}}},{{{0, 1, 0}}},{{{0, 1, 0}}},{{{0, 1, 0}}},{{{0, 1, 0}}},{{{0, 1, 0}}},{{{0, 1, 0}}},{{{0, 1, 0}}},{{{0, 0, 1}}},{{{0, 0, 1}}},{{{0, 0, 1}}},{{{0, 0, 1}}},{{{0, 0, 1}}},{{{0, 0, 1}}},{{{0, 0, 1}}},{{{0, 0, 1}}},{{{0, 0, 1}}},{{{0, 0, 1}}},{{{0, 0, 1}}},{{{0, 0, 1}}},{{{0, 0, 1}}},{{{0, 0, 1}}},{{{0, 0, 1}}},{{{0, 0, 1}}},{{{0, 0, 1}}},{{{0, 0, 1}}},{{{0, 0, 1}}},{{{0, 0, 1}}},{{{0, 0, 1}}},{{{0, 0, 1}}},{{{0, 0, 1}}},{{{0, 0, 1}}},{{{0, 0, 1}}},{{{0, 0, 1}}},{{{0, 0, 1}}},{{{0, 0, 1}}},{{{0, 0, 1}}},{{{0, 0, 1}}},{{{0, 0, 1}}},{{{0, 0, 1}}},{{{0, 0, 1}}},{{{0, 0, 1}}},{{{0, 0, 1}}},{{{0, 0, 1}}},{{{0, 0, 1}}},{{{0, 0, 1}}},{{{0, 0, 1}}},{{{0, 0, 1}}},{{{0, 0, 1}}},{{{0, 0, 1}}},{{{0, 0, 1}}},{{{0, 0, 1}}},{{{0, 0, 1}}},{{{0, 0, 1}}},{{{0, 0, 1}}},{{{0, 0, 1}}},{{{0, 0, 1}}},{{{0, 0, 1}}}};

void test1(){
    NNd nn(vector<int>{4, 16, 16, 3});

    vector<pair<Matrixd, Matrixd>> trainData;
    for(int i=0;i<irisX.size();i++){
        trainData.push_back({irisX[i].T(), irisY[i].T()});
    }

    nn.feed(trainData, 200, 10, 3);
    
    dbg(nn(irisX[  0].T()), irisY[  0].T());
    dbg(nn(irisX[ 50].T()), irisY[ 50].T());
    dbg(nn(irisX[149].T()), irisY[149].T());
	
	cout << "total accuracy: "<<inner_product(all(irisX), irisY.begin(), 0, [](int l, int r){return l+r;}, [&](auto l, auto r){
		auto mIndex = [](auto v)->int{return distance(max_element(all(v)), v.begin());};
		return mIndex(nn(l.T()).T()[0]) == mIndex(r[0]);
	}) / double(irisX.size());
}

int main(){
    
    test1();
    
    return 0;
}