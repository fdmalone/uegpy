#include <stdio.h>
#include <math.h>
#include <gsl/gsl_integration.h>
#include <gsl/gsl_errno.h>
#include <iostream>

using namespace std;

static double const PI = 3.141592653589793;

struct Lindhard {
    double beta, mu, q, omega;
};

struct Angular {
    double k, q, xi, beta, mu;
};

struct QInt {
    double k, xi, beta, mu;
};

double fermi_factor(double ek, double mu, double beta) {

    return 1.0 / (exp(beta*(ek-mu))+1.0);

}

double bose_factor(double ek, double beta) {

    return 1.0 / (exp(beta*ek)-1);

}

double re_lind_integrand(double k, void *p) {

    double k_pl, k_mi, omega, q, beta, mu;
    Lindhard &params = *reinterpret_cast<Lindhard *>(p);

    omega = params.omega;
    q = params.q;
    beta = params.beta;
    mu = params.mu;

    k_pl = (0.5*q*q + omega) / q;
    k_mi = (0.5*q*q - omega) / q;

    //std::cout << k_pl << "   " << k_mi << "   " << q << std::endl;

    return (k * fermi_factor(0.5*k*k, mu, beta) * (log(fabs((k_pl+k)/(k_pl-k))) + log(fabs((k_mi+k)/(k_mi-k)))));

}

double re_lind(double omega, double q, double beta, double mu) {

    double result, error;
    Lindhard params;
    gsl_integration_workspace * w = gsl_integration_workspace_alloc(1000);
    gsl_function F;

    params.omega = omega;
    params.q = q;
    params.beta = beta;
    params.mu = mu;

    F.function = &re_lind_integrand;
    F.params = reinterpret_cast<void *>(&params);

    gsl_integration_qags(&F, 0, 200, 0, 1e-6, 1000, w, &result, &error);

    gsl_integration_workspace_free(w);

    return (-1.0/(4*PI*PI*q*q) * result);

}

double im_lind(double omega, double q, double beta, double mu) {

    double eq, e_pl, e_mi;

    eq = 0.5*q*q;
    e_pl = (eq + omega)*(eq + omega)/(4*eq);
    e_mi = (eq - omega)*(eq - omega)/(4*eq);

    return ((-1.0/(4.0*PI*beta*q)) * log((1.0+exp(-beta*(e_mi-mu)))/ (1.0+exp(-beta*(e_pl-mu)))));

}

double im_chi_rpa(double omega, double q, double beta, double mu) {

    double vq, num, denom, re, im;

    vq = 4.0*PI / (q*q);
    im = im_lind(omega, q, beta, mu);
    re = 1.0 - vq*re_lind(omega, q, beta, mu);
    denom = re*re + (vq*im)*(vq*im);

    return (im / denom);

}

double angular_integrand(double u, void *args) {

    double omega, k, q, beta, mu, xi;
    Angular &params =  *reinterpret_cast<Angular *>(args);

    k = params.k;
    q = params.q;
    xi = params.xi;
    beta = params.beta;
    mu = params.mu;

    omega = 0.5*k*k + 0.5*q*q + k*q*u;

    //std::cout << u << "   "<< omega << "   " << k << "   " << q << std::endl;
    return (4*PI/q*q) * im_chi_rpa(omega-xi, q, beta, mu) * (bose_factor(omega-xi, beta)+fermi_factor(omega, mu, beta));

}

double q_integrand(double q, void *args) {

    double omega, k, beta, mu, xi, result, error;
    QInt &params = *reinterpret_cast<QInt *>(args);
    Angular p2;

    gsl_integration_workspace * w = gsl_integration_workspace_alloc(1000);
    gsl_function F;

    k = params.k;
    xi = params.xi;
    beta = params.beta;
    mu = params.mu;

    p2.k = k;
    p2.q = q;
    p2.xi = xi;
    p2.beta = beta;
    p2.mu = mu;

    //std::cout << "qint: " << q << std::endl;

    F.function = &angular_integrand;
    F.params = reinterpret_cast<void *>(&p2);

    gsl_integration_qags(&F, -1, 1, 0, 1e-3, 1000, w, &result, &error);
    //std::cout << q << "    " << result << std::endl;

    gsl_integration_workspace_free(w);

    return result;

}

double im_sigma_rpa(double xi, double k, double beta, double mu) {

    double result, error;
    QInt params;
    gsl_integration_workspace * w = gsl_integration_workspace_alloc(1000);
    gsl_function F;

    params.k = k;
    params.xi = xi;
    params.beta = beta;
    params.mu = mu;

    F.function = &q_integrand;
    F.params = reinterpret_cast<void *>(&params);

    gsl_integration_qags(&F, 0.01, 100, 0, 1e-3, 1000, w, &result, &error);
    //q_integrand(0.1, F.params);

    //double dq = 0.1;
    //for (int qv = 0; qv < 2000; qv++) {
        ////std::cout << dq << std::endl;
        //result = q_integrand(dq, F.params);
        ////result = angular_integrand(dq, F.params);
        //std::cout << dq << "   " << result << std::endl;
        //dq += 0.001;
    //}
    gsl_integration_workspace_free(w);

    return (1.0/PI) * result;

}

int main() {

    double mu = -0.039521;
    double beta = 0.5430107179652064;
    double k, omega;

    omega = 2.0;
    gsl_set_error_handler_off();
    for (int o = 0; o < 1; o++) {
        omega -= 0.1;
        k = 0.2;
        for (int kit = 0; kit < 1; kit++) {
            //k += 0.5;
            std::cout << k << "   " << omega << "   " << im_sigma_rpa(omega, k, beta, mu) << endl;
        }
    }
    //std::cout << im_sigma_rpa(2, 1, beta, mu) << std::endl;

}
