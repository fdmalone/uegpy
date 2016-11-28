#include <stdio.h>
#include <math.h>
#include <iostream>
#include <vector>
#include <complex>

using namespace std;

static double const PI = 3.141592653589793;

double trapezoid(vector<double> &f, double dx) {

    // Trapezoidal rule.
    //
    // In:
    //     f: vector
    //        function to be integrated.
    //     dx: double
    //         spacing between data points in f.

    double res;

    res = f[0] + f[f.size()-1];

    for (int i=1; i < f.size()-1; i++) {
        res += 2.0*f[i];
    }

    return 0.5 * dx * res;

}

double fermi_factor(double ek, double mu, double beta) {

    return 1.0 / (exp(beta*(ek-mu))+1.0);

}

double bose_factor(double ek, double beta) {

    double nb;

    nb = 1.0 / (exp(beta*ek)-1);

    if (nb > 1e5) {
        return 1e5;
    }
    else {
        return nb;
    }
}

double re_lind_integrand(double k, double omega, double q, double beta, double mu) {

    double k_pl, k_mi;

    k_pl = (0.5*q*q + omega) / q;
    k_mi = (0.5*q*q - omega) / q;

    return (k * fermi_factor(0.5*k*k, mu, beta) * (log(fabs((k_pl+k)/(k_pl-k))) + log(fabs((k_mi+k)/(k_mi-k)))));

}

double re_lind(double omega, double q, double beta, double mu, double kmax=10, double dk=0.01) {

    double k, integral;
    int its;
    vector<double> I;

    k = 0.001;
    its = (int)(kmax/dk) + 1;

    for (int i = 0; i < its; i++) {
        I.push_back(re_lind_integrand(k, omega, q, beta, mu));
        k = k + dk;
    }

    integral = trapezoid(I, dk);

    return (-1.0/(4*PI*PI*q) * integral);

}

double im_lind(double omega, double q, double beta, double mu) {

    double eq, e_pl, e_mi;

    eq = 0.5*q*q;
    e_pl = (eq + omega)*(eq + omega)/(4*eq);
    e_mi = (eq - omega)*(eq - omega)/(4*eq);

    return ((-1.0/(4.0*PI*beta*q)) * log((1.0+exp(-beta*(e_mi-mu)))/ (1.0+exp(-beta*(e_pl-mu)))));

}

double im_eps_inv(double omega, double q, double beta, double mu, double kmax=10, double eta=0.001) {

    double vq;
    double denom, re_eps, im_eps;

    vq = 4.0*PI / (q*q);
    re_eps = 1.0 - vq * re_lind(omega, q, beta, mu, kmax);
    im_eps = -vq * im_lind(omega, q, beta, mu);
    denom = re_eps*re_eps + im_eps*im_eps + eta*eta;

    complex<double> eps(re_eps, im_eps+omega/abs(omega) * eta);
    //return -im_eps / denom;
    return (1.0 / eps).imag();
}

double angular_integral(double k, double omega, double q, double beta, double mu, double kmax=10, double eta=0.001) {

    vector<double> I;
    double du = 0.005, u, ekq;
    int imax;

    imax = (int)(2/du) + 1;

    u = -1.0;
    for (int i = 0; i < imax; i++) {
        ekq = 0.5*k*k + 0.5*q*q + k*q*u - mu;
        I.push_back(im_eps_inv(ekq-omega, q, beta, mu, kmax, eta) * (bose_factor(ekq-omega, beta)+fermi_factor(ekq, 0, beta)));
        //cout << "uint: " << u << "    " << I[i] << endl;
        u = u + du;
    }

    return trapezoid(I, du);

}

double im_sigma_rpa(double omega, double q, double beta, double mu, double qmax=10) {

    double result, error;
    double dq, k;
    vector<double> I;

    int imax = 1000;

    dq = qmax/imax;
    k = 0.001;
    for (int i = 0; i < imax; i++) {
        I.push_back(angular_integral(k, omega, q, beta, mu, qmax, dq));
        k += dq;
    }

    return (1.0/PI) * trapezoid(I, dq);

}

int main() {

    double mu = 1.84158;
    double beta = 10/mu;
    double k, omega, kf, im, re, inv, im_eps_inv, domega;
    vector<double> I;
    int omax;

    omega = -5.0*mu;
    kf = sqrt((2*mu));
    omax = 2*abs(omega) / domega + 1;
    domega = 0.1;
    for (int o = 0; o < omax; o++) {
        k = 0.4;
        //im = 4*PI/(k*k)*im_lind(omega, k, beta, mu);
        //re = 1-4*PI/(k*k)*re_lind(omega, k, beta, mu, kf);
        im = im_sigma_rpa(omega, k, beta, mu, 10*kf);
        //im_eps_inv = im_eps_inv(omega, k, beta, mu);
        //std::cout << omega << "   " << im << "    " << re << "   " << im/(im*im+re*re+0.02) <<  endl;
        std::cout << omega << "   " << im << endl;
        std::cerr << omega << "   " << im << endl;
        omega += domega;
    }
    k = 0.0;

}
