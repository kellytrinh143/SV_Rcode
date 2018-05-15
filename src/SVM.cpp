// -*- mode: C++; c-indent-level: 4; c-basic-offset: 4; indent-tabs-mode: nil; -*-

// we only include RcppArmadillo.h which pulls Rcpp.h in for us
#include "RcppArmadillo.h"
using namespace arma;
// via the depends attribute we tell Rcpp to create hooks for
// RcppArmadillo so that the build process will know what to do
//
// [[Rcpp::depends(RcppArmadillo)]]
// Define some functions used in AR-MH algorithm
// TO DO: Need to ask Conrad to see if there are efficeint ways to create the funtions.
// It is not a smart way now :(
// This function is used to see the structure of matrix  for debugging purpose
// Log posterior of ht
inline
  double lph (const vec& x, const vec& deltah, const vec& y, const mat& HiSH, const double mu, const double alp){
    double out = -.5* as_scalar((x-deltah).t()*HiSH*(x-deltah))-.5*as_scalar(sum(x))
    -
      .5* dot(exp(-x).t(),pow(y-mu-alp*exp(x),2));
    return out;
  }

// Conditional posterior for phih

inline
  double gphih (const double& x, const double& omegah2, const vec& h, const double& muh){
    double out = -.5*log(omegah2/pow((1-x),2))
    -
      .5*(1-pow(x,2))/omegah2*pow(as_scalar(h.row(0))- muh,2);
    return out;
  }
// This function is to estimate Stochastic Volatility in Mean
// [[Rcpp::export]]
Rcpp:: List
  SVM(const arma::vec& y, const arma::uword nloop, const arma::uword burnin, const arma::uword skip)
  {
    const bool debug = true;

    if(debug)
    {
      Rcpp::Rcout << "SVM(); using Armadillo " << arma_version::as_string() << endl;
      Rcpp::Rcout << "size(y): " << size(y)<< endl;
      //Rcpp::Rcout << "phih0:   " << phih0 << endl; // Ideally need to pass the values outside of the function
      //Rcpp::Rcout << "Vphi:  "  << Vphih  << endl;
      //Rcpp::Rcout << "mu0:   "  << mu0    << endl;
      //Rcpp::Rcout << "Vmu:  "   << Vmu    << endl;
      //Rcpp::Rcout << "Vmuh:  "  << Vmuh   << endl;
      //Rcpp::Rcout << "nuh:   "  << nuh    << endl;
      //Rcpp::Rcout << "Sh:  "    << Sh     << endl;
      //Rcpp::Rcout << "alp0:   " << alp0   << endl;
      //Rcpp::Rcout << "Valp0:  " << Valp0  << endl;
      Rcpp::Rcout << "nloop:   " << nloop << endl;
      Rcpp::Rcout << "burnin:  " << burnin<< endl;
      Rcpp::Rcout << "skip:  "   << skip  <<endl; // To improve converegence in MCMC algorithm
    }
    // Todo: Need to ask Conrad how to pass negative values to a function. Ideally the prior
    // should be passed from outside of the function.

    // Prior
    const uword T   = y.n_elem;
    double phih0    = 0.97;
    double Vphih    = pow(0.1,2);
    double mu0      = 0;
    double Vmu      = 10;
    double muh0     = -10;
    double Vmuh     = 10;
    double nuh      = 5;
    double Sh       = pow(0.2,2)*(nuh-1);
    double alp0     = 0;
    double Valp     = pow(100,2);


    // initialise the Markov chain
    double mu       = mean(y);
    double phih     = 0.98;
    double omegah2  = pow(0.2, 2);
    double alp      = 0;
    double muh      = log(var(y));


    vec h           = muh + sqrt(omegah2)*randn<mat>(T,1);
    vec exph        = exp(h);
    mat Hphi        = eye(T,T) - diagmat(phih*ones(T-1,1), -1 );


    // Storage for MCMC algorithm
    mat store_h     = zeros(nloop,T);
    vec theta       = vec{mu,alp,muh,phih,omegah2};
    mat store_theta = zeros (nloop, theta.n_elem);
    double counth   = 0;
    double countphih= 0;


    Rcpp::Rcout << "Starting SV-M..." << endl;

    wall_clock timer;
    timer.tic();
    uword  totalloop = nloop*skip + burnin;
    for (uword iloop = 0; iloop < totalloop ; ++iloop)
    {


      // Block 1: Sample mu and alp


      mat     X         = join_rows(ones<mat>(T,1), exph);
      mat     invSy     = diagmat(1/exph);
      mat     XinvSy    = X.t()*invSy;

      mat     invDbeta  = diagmat(join_cols((1/Vmu)*ones(1,1),(1/Valp)*ones(1,1)))
        +
          XinvSy*X;
      vec     betahat   = solve(invDbeta, join_cols(mu0/Vmu*ones(1,1),alp0/Valp*ones(1,1))
                                  +
                                    XinvSy*y);
      mat     CinvDbeta = chol(invDbeta, "upper");
      vec     beta      = betahat + solve(trimatu(CinvDbeta), randn<mat>(2,1), solve_opts::fast);
      double  mu        = beta(0);
      double  alp       = beta(1);

      // Block 2: Sample h

      vec    temp       = join_cols((1-pow(phih,2))/omegah2*ones(1,1),1.0/omegah2*ones(T-1,1)); // TODO: Double check with Conrad to see if there is better way to construct HiSH
      mat    HiSH       = Hphi.t()*diagmat(temp)*Hphi;

      vec     dmuh      = join_cols(muh*ones(1,1),(1-phih)*muh*ones(T-1,1));
      vec     deltah    = solve(Hphi,dmuh, solve_opts::fast);
      vec     HinvSHdeltah = HiSH*deltah;
      vec     s2        = pow((y-mu),2);
      vec     ht        = h;

      // Define some variables which will be used after while lo op

      mat    Kh         = zeros<mat>(T, T);
      int   flag        = 0;
      while (flag < 1) {
        vec   expht       = exp(ht);
        vec   sinvexpht   = s2/expht;
        vec   lam2expht   = pow(alp,2)*expht;
        vec   fh          = -.5 + .5*sinvexpht -.5*lam2expht;
        //Rcpp::Rcout << "fh" << fh.rows(0,5) << endl;
        vec   Gh          = .5*sinvexpht + .5*lam2expht;
        //Rcpp::Rcout << "Gh" << Gh.rows(0,5) << endl;
        Kh          = HiSH + diagmat(Gh);
        //  Rcpp::Rcout << "Kh " << Kh.rows(0,5) << endl;

        vec   newht       = solve(Kh, fh + Gh%ht + HinvSHdeltah);
        //Rcpp::Rcout << "newht before while loop" << newht.rows(0,10) << endl;
        double   errh     = max (abs(newht - ht));
        if (errh < 0.0001){
          flag=1;
        }
        ht          = newht;
      }
      mat cholHh = chol(Kh, "upper");


      // AR step
      vec hstar  = ht;
      double logc   = lph(ht, deltah, y, HiSH, mu, alp) + log (3.0);
      int flag1     = 0;

      // Define some variables before the while loop

      vec     hc    = ht + solve(trimatu(cholHh),randn<vec>(T,1), solve_opts::fast);
      double alpARc = 0;
      double alpMH  = 0;
      int testcount = 0;

      while (flag1==0)
      {
        hc        = ht + solve(trimatu(cholHh),randn<vec>(T,1), solve_opts::fast);
        alpARc    = lph(hc, deltah, y, HiSH, mu, alp)
          +
            double(.5)*as_scalar((hc-ht).t()*Kh*(hc-ht))-logc;
        testcount ++;
        if (alpARc > double(log(randu()))){
          flag1=1;
        }
      }



      // MH step

      double alpAR   = lph(h, deltah, y, HiSH, mu, alp)
        +
          .5*as_scalar((h-ht).t()*Kh*(h-ht))-logc;

        if (alpAR<0){
          alpMH = 1;
        }
        else if (alpARc<0){
          alpMH = -alpAR;
        }
        else {
          alpMH = alpARc - alpAR;
        }


        if (alpMH > log (randu()) || iloop <1) {
          h = hc;
          exph = exp(h);
          counth = counth + 1;
        }


        // Rcpp::Rcout << "Finish sampling ht" << h.rows(0,10)<<endl;

        // Block 3: Sample omegah2
        // Rcpp::Rcout << "muh  " << muh << endl;
        vec eh            = join_cols((h.row(0)-muh)*sqrt(1-pow(phih,2)), h.rows(1, T-1)
                                        -
                                          (1-phih)*muh - phih*h.rows(0, T-2));
        // Rcpp::Rcout << "eh" << eh.rows(0,10) << endl;

        double newSh      = Sh + as_scalar(sum(pow(eh,2)))/2;
        //Rcpp::Rcout << "newSh  " << newSh <<endl;
        double  newnuh    = T/2 + nuh;
        omegah2   = 1.0/as_scalar(randg(1,distr_param(newnuh, 1.0/newSh)));
        // Rcpp::Rcout << "Finish sampling omegah2  " << omegah2 << endl;

        //Block 4: Sample phih

        vec Xphi          = h.rows(0, T-2) - muh;
        vec yphi          = h.rows(1, T-1) - muh;
        double Dphi       = 1.0/(1.0/Vphih + dot(Xphi.t(),Xphi)/omegah2);
        double phihat     = Dphi*(phih0/Vphih + dot(Xphi.t(),yphi)/omegah2);
        double phihc      = phihat + sqrt(Dphi)*randn();
        // Rcpp::Rcout << "Finish sampling phih  " << phih << endl;

        // Check stationary of phihc

        if (abs(phihc) < double(.999)) {
          alpMH = gphih (phihc, omegah2, h,  muh) - gphih (phih, omegah2, h,  muh);
          if (alpMH > log(randu())){
            phih = phihc;
            countphih = countphih + 1;
            Hphi  =  eye(T,T)- phih*diagmat(ones<vec>(T-1,1),-1);

          }
        }

        // Block 5: Sample muh

        double dDmuh       = 1.0/Vmuh + ((T-1)*pow(1-phih,2)+ (1-pow(phih,2)))/omegah2;
        double Dmuh        = 1.0/dDmuh;
        double dmuhat      = muh0/Vmuh + (1-pow(phih,2))/omegah2*h(0)
          +
            (1-phih)/omegah2*sum(h.rows(1,T-1)-phih*h.rows(0, T-2));
        double muhhat      = Dmuh*dmuhat;
        muh                = muhhat + sqrt(Dmuh)*randn();


        if ((iloop % 100) ==0 && iloop>0) {
          Rcpp::Rcout << "iloopth" << iloop << endl;
        }

        // Store the results


        if (iloop >=burnin && ((iloop-burnin) % skip)==0){
          uword i              = (iloop - burnin)/skip;
          store_h.row(i)       = h.t();
          rowvec theta         = rowvec({mu,alp,muh,phih,omegah2});
          store_theta.row (i)  = theta;
        }
    }
    Rcpp::Rcout << "timer.toc(): " << timer.toc() <<endl;
    mat meanh                = mean(store_h).t();
    mat meanTheta            = mean(store_theta).t();
    double accep_h           = counth/(nloop*skip+burnin);
    double accep_phih        = countphih/(nloop*skip+burnin);
    return Rcpp::List::create(Rcpp::Named("meanTheta")= meanTheta,
                              //Rcpp::Named("meanh")= meanh,
                              Rcpp::Named("The acceptance rate for ht") = accep_h,
                              Rcpp::Named("The acceptance rate for phih") = accep_phih);
  }

// TO DO: Create a nice output in R
// Ask Conrad how to call other dependecy packages in R, save all the results .
// How to know which block work slowly
// join_cols: Do I really need to type ones(1,1)

