#include <iostream>
#include <cmath>
#include <chrono>
#include <vector>
#include <omp.h>

#define N 512

// вариант 4
double u(double x) {
	return cos(2 * x); 
}

// вариант 4
double q(double x) {
	return x * x; 
}

// вариант 4
double f(double x) {
	return (4 + x * x) * cos(2 * x); 
}

// Метод прогонки
void solveEquationByProgonka() {
	double* y = new double[N + 1]; // y0, y2, ... , yn
	double* P = new double[N - 1]; // P0, P1, ... , Pn-2
	double* Q = new double[N - 1]; // Q0, Q1, ... , Qn-2
	double* x = new double[N + 1]; // x0, x1, ... , xn

	double a = 0.0, b = 2; // вариант 4

	y[0] = u(a);
	y[N] = u(b);
	double h = (b - a)/N;
	double hh = h * h;

	// создаём сетку
	for (int i = 0; i <= N; i++)
		x[i] = a + (i + 1) * h;

	std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();

	// start from P[1] : Q[1]
	P[0] = 1 / (2 + hh * q(x[1]));
	Q[0] = (hh * f(x[1]) + y[0]) / (2 + hh * q(x[1]));

	// Прямой ход прогонки
	for (int i = 1; i < N - 1; i++) {
		double divisor = (2 + hh * q(x[i+1])) - P[i - 1];
		P[i] = 1 / divisor;
		Q[i] = (hh * f(x[i + 1]) + Q[i - 1]) / divisor;
	}

	// Обратный ход прогонки
	for (int i = N - 1; i >= 1; i--) {
		y[i] = P[i - 1] * y[i + 1] + Q[i - 1];
	}

	std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();

	// Выцисление равновесной нормы
	double equilibriumNorm = 0.0;
	for (int i = 1; i < N - 1; i++) {
		double diff = std::fabs(y[i] - u(x[i]));
		equilibriumNorm = std::max(equilibriumNorm, diff);
	}

	std::cout << "Time for solveEquationByProgonka = " 
		<< std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count() 
		<< " microseconds" << std::endl
		<< " norma value = " << equilibriumNorm << std::endl;


	delete[] x;
	delete[] P;
	delete[] Q;
	delete[] y;
	return;
}

void cyclicReduction(std::vector<double>& a, std::vector<double>& b, std::vector<double>& c, 
					std::vector<double>& fx, std::vector<double>& yx, int q) {
    int n = static_cast<int>(b.size());

    int sdvig = 1; // шаг в глубину по дереву
    int hag = 2; // шаг, нижний индекс

	// поднимаемся по дереву
#pragma omp parallel for
	for (int k = 0; k < q - 1; ++k) {
#pragma omp parallel
        for (int i = hag; i < n - hag; i += hag) {
            double P = a[i] / b[i - sdvig];
            double Q = c[i] / b[i + sdvig];

            a[i] = P * a[i - sdvig];
            c[i] = Q * c[i + sdvig];
            b[i] = b[i] - P * c[i - sdvig] - Q * a[i + sdvig];
            fx[i] = fx[i] + P * fx[i - sdvig] + Q * fx[i + sdvig];
        }

        sdvig = 2 * sdvig;
        hag = 2 * hag;
    }

	// спускаемся по дереву
    for (int k = q - 1; k >= 0; --k) {
		sdvig = sdvig / 2;
        hag = hag / 2;

#pragma omp parallel for
        for (int i = hag; i < n - hag; i += hag) {
			yx[i] = (fx[i] + a[i] * yx[i-sdvig] + c[i] * yx[i+sdvig]) / b[i];
        }
    }
}


void solveEquationByСyclicReduction() {
    const int n = N; // Размерность СЛАУ (2^q)

    const int qq = log2(n); // Значение q

	double aa = 0.0, bb = 2;	// вариант 4

	double h = (bb - aa)/n;
	double hh = h * h;
	// создаём сетку
    std::vector<double> x(n + 1, 0.0);
	for (int i = 0; i <= n; i++)
		x[i] = aa + i * h;

	double a_= 1.0 / (hh); 
	double c_ = 1.0 / (hh);
	std::vector<double> aVec(n+1, a_);
	std::vector<double> cVec(n+1, c_);
	std::vector<double> bVec(n+1, 0.0);
	std::vector<double> fx(n+1, 0.0);

	std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();

	// Выцисление коэффичиентов и правой части
	for (int i = 0; i < n; i++) {
		bVec[i] = (2.0 + hh * q(x[i])) / hh;
		fx[i] = f(x[i]);         
	}

	// solution
	std::vector<double> yx(n+1, 0.0);
	yx[0] = u(x[0]);
	yx[n - 1] = u(x[n - 1]);

	cyclicReduction(aVec, bVec, cVec, fx, yx, qq);

	std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();

	// Выцисление равновесной нормы
	double equilibriumNorm = 0.0;
	for (int i = 1; i < N - 1; i++) {
		double diff = std::fabs(yx[i] - u(x[i]));
		equilibriumNorm = std::max(equilibriumNorm, diff);
	}

	std::cout << "Time for solveEquationByProgonka = " 
		<< std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count() 
		<< " microseconds" << std::endl
		<< " norma value = " << equilibriumNorm << std::endl;

    // // Выводим результаты
    // std::cout << "Results:" << std::endl;
    // for (int i = 0; i < n; ++i) {
    //     std::cout << "f[" << i << "] = " << yx[i] << " - " << u(x[i]) << "=" 
	// 		<< yx[i] - u(x[i]) << std::endl;
    // }
}

int main(void)
{
	solveEquationByProgonka();
	solveEquationByСyclicReduction();
}


void cyclicReduction3(std::vector<double>& a, std::vector<double>& b, std::vector<double>& c,
					 std::vector<double>& fx, std::vector<double>& yx,
					 int q) {
    int n = static_cast<int>(b.size());

    int sdvig = 1; // шаг в глубину по дереву
    int hag = 2; // шаг, нижний индекс

	// поднимаемся по дереву
#pragma omp parallel for
	for (int k = 0; k < q - 1; ++k) {
#pragma omp parallel
        for (int i = hag; i < n - hag; i += hag) {
            double P = a[i] / b[i - sdvig];
            double Q = c[i] / b[i + sdvig];

            a[i] = P * a[i - sdvig];
            c[i] = Q * c[i + sdvig];
            b[i] = b[i] - P * c[i - sdvig] - Q * a[i + sdvig];
            fx[i] = fx[i] + P * fx[i - sdvig] + Q * fx[i + sdvig];
        }

        sdvig = 2 * sdvig;
        hag = 2 * hag;
    }

	// спускаемся по дереву
    for (int k = q - 1; k >= 0; --k) {
		sdvig = sdvig / 2;
        hag = hag / 2;
#pragma omp parallel for
        for (int i = hag; i < n - hag; i += hag) {
			yx[i] = (fx[i] + a[i]*yx[i-sdvig] +c[i] * yx[i+sdvig]) / b[i];
        }
    }
}

// Функция для решения СЛАУ методом циклической редукции
void cyclicReduction2(std::vector<double>& a, std::vector<double>& b, std::vector<double>& c,
					  std::vector<double>& d, std::vector<double>& x) {
    int n = static_cast<int>(a.size());

    // Прямой ход
    for (int i = 1; i < n; i++) {
        double m = a[i] / b[i - 1];
        b[i] -= m * c[i - 1];
        d[i] -= m * d[i - 1];
    }

    // Обратный ход
    x[n - 1] = d[n - 1] / b[n - 1];
    for (int i = n - 2; i >= 0; i--) {
        x[i] = (d[i] - c[i] * x[i + 1]) / b[i];
    }
}

void gaussElimination(std::vector<std::vector<double>>&& matrix, std::vector<double>& x, std::vector<double>& y);

void Reduce(std::vector<double>& a, std::vector<double>& b, std::vector<double>& c, 
					 std::vector<double>& fx, std::vector<double>& yx, int n);
std::vector<std::vector<double>>   compress(std::vector<double>& a, std::vector<double>& b, std::vector<double>& c);

int main2() {
    const int n = 256; // Размерность СЛАУ (2^q)

    const int qq = log2(n); // Значение q

	double aa = 0.0, bb = 2;	// вариант 4

	double h = (bb - aa)/N;
	double hh = h * h;
	// создаём сетку
    std::vector<double> x(n, 0.0);
	for (int i = 0; i <= n; i++)
		x[i] = aa + (i + 1) * h;

	double a_ = 1; // 1.0 / hh;
	
	double c_ = a_;


    std::vector<double> a(n, a_);	// 0-го эл-та нет
    std::vector<double> c(n, c_); 	// n-того элемента нет

	//double b_ = 2.0 + hh;
	std::vector<double> b(n, 0.0);
    for(size_t i = 0; i < b.size(); i++) {
		b[i] = -2 * q(x[i]); //(2 + hh * q(x[i])) / hh;
	}

	std::vector<double> fx(n,0);
	for(size_t i = 0; i < fx.size(); i++) {
		fx[i] = hh * f(x[i]);
	}


	std::vector<double> yx(n, 0.0);
	yx[0] = u(x[0]);
	yx[n-1] = u(x[n-1]);
    // Вызываем прямой ход циклической редукции
	// Reduce(a, b, c, fx, yx, qq);
	// void gaussElimination(std::vector<std::vector<double>>& matrix, std::vector<double>& x, std::vector<double>& y)
	// gaussElimination(compress(a,b,c),fx,yx);
    // cyclicReduction(a, b, c, fx, yx, qq);
    cyclicReduction2(a, b, c, fx, yx);

    // Выводим результаты
    std::cout << "Results:" << std::endl;
    for (int i = 0; i < n; ++i) {
        std::cout << "f[" << i << "] = " << yx[i] << " - " << u(x[i]) << "=" 
			<< yx[i] - u(x[i]) << std::endl;
    }

    return 0;
}


// std::vector<double> cyclicReduction(int cN) {
//     std::vector<double> A(N, 2.0); // Матрица коэффициентов
//     std::vector<double> B(N, 1.0); // Вектор свободных членов

// #pragma omp parallel for
//     for (int stride = cN / 2; stride >= 1; stride /= 2) {
// #pragma omp for
//         for (int i = stride; i < cN; i += stride * 2) {
//             double ratio = A[i] / A[i - stride];
//             A[i] -= ratio * A[i - stride];
//             B[i] -= ratio * B[i - stride];
//         }
//     }

//     // Обратный ход циклической редукции
//     for (int stride = 1; stride < cN; stride *= 2) {
// #pragma omp parallel for
//         for (int i = 2 * stride - 1; i < cN; i += 2 * stride) {
//             double ratio = A[i] / A[i - stride];
//             B[i] -= ratio * B[i - stride];
//         }
//     }

//     // Вычисление решения
//     for (int i = 0; i < N; ++i) {
//         B[i] /= A[i];
//     }
//     return B;
// }



/* 
  Алгоритм редукции для решения трехдиагональных систем.
	Описание:
   void Reduce(double *u,double *a,double *b,double *c,double *f,int n);
	Параметры:
   u - разрешение (размера n), на выходе;
   a,b,c,f - массивы коэффициентов (как в алгоритмах, описанных выше, размера n)
   n - размер разрешения и массива коэффициентов;
   Примечание:
   Алгоритм реализован без проверки на производительность. 
*/


void Reduce(std::vector<double>& a, std::vector<double>& b, std::vector<double>& c, 
					 std::vector<double>& F, std::vector<double>& yx, int q){

    // Forward Reduction
    for (int i = 0; i < log2(N + 1); ++i) {
        int offset = pow(2, i); // TODO hag + step * 2
        for (int j = offset*2 - 1; j < N; j += offset*2) {
            int id1 = j - offset;
            int id2 = j + offset;

            double alpha = a[i] / b[i];
            double beta = c[i] / b[i];

            a[j] = -a[id1]*alpha;
            b[j] = b[j] - c[id1]*alpha -a[id2]*beta;
            c[j] = -c[id1] * beta;
		}
    }

    // Backward Substitution
    for (int i = log2(N + 1) - 2; i >= 0; --i) {
        int offset = pow(2, i); // TODO hag + step * 2
        for (int j = offset*2 - 1; j < N; j += offset*2) {
            int id1 = j - offset;
            int id2 = j + offset;

            if (id1 - offset < 0) {
                yx[id1] = (F[id1] - c[id1] * yx[id1+offset])/ b[id1];
            } else {
                yx[id1] = (F[id1] - a[id1] * yx[id1 - offset] - c[id1] * yx[id1 + offset]) / b[id1];
            }

            if (id2 + offset >= N) {
                yx[id2] = (F[id2] - a[id1] * yx[id2 - offset]) / b[id2];
            } else {
                yx[id2] = (F[id2] - a[id2] * yx[id2 - offset] - c[id2] * yx[id2 + offset]) / b[id2];
            }
        }
    }
}

void gaussElimination(std::vector<std::vector<double>>&& matrix, std::vector<double>& x, std::vector<double>& y) {
    int n = matrix.size();

    // Прямой ход метода Гаусса
    for (int i = 0; i < n; ++i) {
        // Приведение матрицы к треугольному виду
        for (int k = i + 1; k < n; ++k) {
            double factor = matrix[k][i] / matrix[i][i];
            for (int j = i; j < n + 1; ++j) {
                matrix[k][j] -= factor * matrix[i][j];
            }
        }
    }

    // Обратный ход метода Гаусса
    for (int i = n - 1; i >= 0; --i) {
        x[i] = matrix[i][n] / matrix[i][i];
        y[i] = x[i];

        for (int k = i - 1; k >= 0; --k) {
            matrix[k][n] -= matrix[k][i] * matrix[i][n];
            matrix[k][i] = 0;
        }
    }
}

std::vector<std::vector<double>>  compress(std::vector<double>& a, std::vector<double>& b, std::vector<double>& c){

	std::vector<std::vector<double>> v(b.size());
	for (auto &a:v){
		a.resize(b.size());
	}
	v[0][0] = b[0];
	v[0][1] = c[0];
	v[b.size()-1][b.size()-1] = b[b.size()-1];
	v[b.size()-1][b.size()-2] = a[b.size()-1];
	for(size_t i = 1; i < b.size()-1; ++i){
		for(size_t j = 1; j < b.size()-1; ++j){
			v[i][j-1] = a[i];
			v[i][j] = 	b[i];
			v[i][j+1] = c[i];
		}
	}
	return v;
}


void CycleReductionMethod(double *x, double a_0, double a_1, double a_2,
                         double *a, double *b, double *c, double *f,
                         int n, int q) {
    // Прямой ход циклической редукции
    a[0] = a_0;
    b[0] = a_1;
    c[0] = a_2;
    f[0] = 0;
    f[n] = 0;
    x[0] = u(0);
    x[n] = u(2);
    int start = 2;
    int elementsNum = n;
    int step = 1;

    for (int j = 0; j < q; ++j) {
        double alpha = -a[j] / b[j];
        double beta = -c[j] / b[j];
        a[j + 1] = alpha * a[j];
        b[j + 1] = b[j] + 2 * alpha * c[j];
        c[j + 1] = beta * c[j];
        elementsNum = (elementsNum - 1) / 2;

        for (int i = 0; i <= elementsNum; ++i) {
            int k = start * (i + 1);
            f[k] = alpha * f[k - step] + f[k] + beta * f[k + step];
        }

        start = 2 * start;
        step = 2 * step;
    }

    // Обратный ход циклической редукции
    start = n / 2;
    step = start;
    elementsNum = 1;

    for (int j = q - 1; j >= 0; --j) {
        double alpha = -a[j] / b[j];
        double beta = -c[j] / b[j];

        for (int i = 0; i < elementsNum; ++i) {
            int k = start * (2 * i + 1);
            x[k] = f[k] / b[j] + alpha * x[k - step] + beta * x[k + step];
        }

        start = start / 2;
        step = start;
        elementsNum = elementsNum * 2;
    }
}
