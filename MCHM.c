#include <stdio.h>
#include <math.h>
#include <gsl/gsl_integration.h>

/* These are the input parameters                                  */
/* mSQ's are resonances masses, fSQ is the compositeness scale     */
/* eSQ is a dimensionful mixing parameter and SQ and SS are signs  */
/* Dimensionful parameters are all squared and in units of TeV^2   */
/* They ought to be 500 GeV - Few TeV (squared)                    */
/* SQ and SQ can each be +1 or -1                                  */
struct input_params { double fSQ; double mrhoSQ; double maSQ;
                      double mQSQ; double mSSQ; double eSQ; int SQ; int SS; };

/* The bulk of the code defines the functions that need to be integrated */
/* The hardcoded parameters are SM gauge couplings and the W mass        */
double d_gamma_g (double pSQ, void * p) {

   struct input_params * params = (struct input_params *)p;
   double fSQ = (params->fSQ);
   double mrhoSQ = (params->mrhoSQ);
   double maSQ = (params->maSQ);
   double mQSQ = (params->mQSQ);
   double mSSQ = (params->mSSQ);
   double eSQ = (params->eSQ);
   int SQ = (params->SQ);
   int SS = (params->SS);
   
   double g2SQ = 0.424984;
   double g1SQ = 0.21296;
   double g20SQ = g2SQ / ( 1-g2SQ*fSQ*maSQ/( 2*mrhoSQ*(maSQ-mrhoSQ) ) );
   double g10SQ = g1SQ / ( 1-g1SQ*fSQ*maSQ/( 2*mrhoSQ*(maSQ-mrhoSQ) ) );
   
   double d_gamma_g = -(3 / (8*16*M_PI*M_PI) ) * ( fSQ*maSQ*mrhoSQ*pSQ / ((maSQ+pSQ)*(mrhoSQ+pSQ)) ) * (
                        g10SQ / (( fSQ*g10SQ*mrhoSQ*pSQ / ((2*maSQ-2*mrhoSQ)*(maSQ+pSQ)) )+pSQ ) +
                        3*g20SQ / (( fSQ*g20SQ*mrhoSQ*pSQ / ((2*maSQ-2*mrhoSQ)*(maSQ+pSQ)) )+pSQ ) );
   return d_gamma_g;
}

double d_beta_g (double pSQ, void * p) {

   struct input_params * params = (struct input_params *)p;
   double fSQ = (params->fSQ);
   double mrhoSQ = (params->mrhoSQ);
   double maSQ = (params->maSQ);
   double mQSQ = (params->mQSQ);
   double mSSQ = (params->mSSQ);
   double eSQ = (params->eSQ);
   int SQ = (params->SQ);
   int SS = (params->SS);
   
   double g2SQ = 0.424984;
   double g1SQ = 0.21296;
   double g20SQ = g2SQ / ( 1-g2SQ*fSQ*maSQ/( 2*mrhoSQ*(maSQ-mrhoSQ) ) );
   double g10SQ = g1SQ / ( 1-g1SQ*fSQ*maSQ/( 2*mrhoSQ*(maSQ-mrhoSQ) ) );
   
   double d_beta_g = (3*pow(fSQ,2)*pow(maSQ,2)*pow(mrhoSQ,2)*pSQ*((-3*pow(g20SQ,2))/
                     pow(pSQ + (fSQ*g20SQ*mrhoSQ*pSQ)/((2*maSQ - 2*mrhoSQ)*(maSQ + pSQ)),2) -
                     (4*g10SQ*pow(maSQ - mrhoSQ,2)*pow(maSQ + pSQ,2)*
                     (4*g20SQ*(maSQ - mrhoSQ)*(maSQ + pSQ) +
                     g10SQ*(3*fSQ*g20SQ*mrhoSQ + 2*(maSQ - mrhoSQ)*(maSQ + pSQ))))/
                     (pow(pSQ,2)*pow(fSQ*g10SQ*mrhoSQ + 2*(maSQ - mrhoSQ)*(maSQ + pSQ),2)*
                     (fSQ*g20SQ*mrhoSQ + 2*(maSQ - mrhoSQ)*(maSQ + pSQ)))))/
                     (1024.*pow(M_PI,2)*pow(maSQ + pSQ,2)*pow(mrhoSQ + pSQ,2));
   return d_beta_g;
}

double d_gamma_f (double pSQ, void * p) {

   struct input_params * params = (struct input_params *)p;
   double fSQ = (params->fSQ);
   double mrhoSQ = (params->mrhoSQ);
   double maSQ = (params->maSQ);
   double mQSQ = (params->mQSQ);
   double mSSQ = (params->mSSQ);
   double eSQ = (params->eSQ);
   int SQ = (params->SQ);
   int SS = (params->SS);
   
   double g2SQ = 0.424984;
   double g1SQ = 0.21296;
   double g20SQ = g2SQ / ( 1-g2SQ*fSQ*maSQ/( 2*mrhoSQ*(maSQ-mrhoSQ) ) );
   double g10SQ = g1SQ / ( 1-g1SQ*fSQ*maSQ/( 2*mrhoSQ*(maSQ-mrhoSQ) ) );
   
   double d_gamma_f = (3*pow(eSQ,2)*(-2*pow(mQSQ,1.5)*sqrt(mSSQ)*(mSSQ + pSQ)*SQ*SS - 
                      2*sqrt(mQSQ)*sqrt(mSSQ)*pSQ*(mSSQ + pSQ)*SQ*SS +
                      pow(mQSQ,2)*(pSQ + mSSQ*pow(SS,2)) +
                      mSSQ*pSQ*(2*mSSQ + pSQ + pSQ*pow(SS,2)) +
                      mQSQ*(pow(mSSQ,2)*pow(SQ,2) + pow(pSQ,2)*(-1 + pow(SQ,2)) +
                      mSSQ*pSQ*(-3 + 2*pow(SQ,2) + 2*pow(SS,2)))))/
                      (16.*pow(M_PI,2)*(mQSQ + pSQ)*(eSQ + mQSQ + pSQ)*(mSSQ + pSQ)*
                      (eSQ + 2*(mSSQ + pSQ)));
   return d_gamma_f;
}

double d_beta_f (double pSQ, void * p) {

   struct input_params * params = (struct input_params *)p;
   double fSQ = (params->fSQ);
   double mrhoSQ = (params->mrhoSQ);
   double maSQ = (params->maSQ);
   double mQSQ = (params->mQSQ);
   double mSSQ = (params->mSSQ);
   double eSQ = (params->eSQ);
   int SQ = (params->SQ);
   int SS = (params->SS);
   
   double g2SQ = 0.424984;
   double g1SQ = 0.21296;
   double g20SQ = g2SQ / ( 1-g2SQ*fSQ*maSQ/( 2*mrhoSQ*(maSQ-mrhoSQ) ) );
   double g10SQ = g1SQ / ( 1-g1SQ*fSQ*maSQ/( 2*mrhoSQ*(maSQ-mrhoSQ) ) );
   
   double d_beta_f = (3*pow(eSQ,2)*(pSQ/pow(eSQ + mQSQ + pSQ,2) +
                     (pSQ*pow(mQSQ + pSQ,2))/(pow(eSQ + mQSQ + pSQ,2)*pow(mSSQ + pSQ,2)) -
                     (2*pSQ*(mQSQ + pSQ))/(pow(eSQ + mQSQ + pSQ,2)*(mSSQ + pSQ)) +
                     pSQ/(pow(mQSQ + pSQ,2)*pow(1 + eSQ/(2.*(mSSQ + pSQ)),2)) +
                     (4*pSQ)/pow(eSQ + 2*(mSSQ + pSQ),2) -
                     (8*pSQ*(mSSQ + pSQ))/((mQSQ + pSQ)*pow(eSQ + 2*(mSSQ + pSQ),2)) -
                     (4*eSQ*mQSQ*(mSSQ + pSQ)*pow(SQ,2))/
                     ((mQSQ + pSQ)*(eSQ + mQSQ + pSQ)*pow(eSQ + 2*(mSSQ + pSQ),2)) +
                     (4*eSQ*mQSQ*pow(mSSQ + pSQ,2)*pow(SQ,2))/
                     (pow(mQSQ + pSQ,2)*(eSQ + mQSQ + pSQ)*pow(eSQ + 2*(mSSQ + pSQ),2)) +
                     (2*eSQ*mQSQ*pow(SQ,2))/(pow(eSQ + mQSQ + pSQ,2)*(eSQ + 2*(mSSQ + pSQ))) -
                     (2*eSQ*mQSQ*(mSSQ + pSQ)*pow(SQ,2))/
                     ((mQSQ + pSQ)*pow(eSQ + mQSQ + pSQ,2)*(eSQ + 2*(mSSQ + pSQ))) +
                     (4*mQSQ*(mSSQ + pSQ)*pow(SQ,2))/
                     ((mQSQ + pSQ)*(eSQ + mQSQ + pSQ)*(eSQ + 2*(mSSQ + pSQ))) +
                     (pow(eSQ,2)*pow(mQSQ,2)*pow(mSSQ + pSQ,2)*pow(SQ,4))/
                     (pSQ*pow(mQSQ + pSQ,2)*pow(eSQ + mQSQ + pSQ,2)*
                     pow(eSQ + 2*(mSSQ + pSQ),2)) +
                     (8*eSQ*sqrt(mQSQ)*sqrt(mSSQ)*SQ*SS)/
                     ((eSQ + mQSQ + pSQ)*pow(eSQ + 2*(mSSQ + pSQ),2)) -
                     (8*eSQ*sqrt(mQSQ)*sqrt(mSSQ)*(mSSQ + pSQ)*SQ*SS)/
                     ((mQSQ + pSQ)*(eSQ + mQSQ + pSQ)*pow(eSQ + 2*(mSSQ + pSQ),2)) +
                     (4*eSQ*sqrt(mQSQ)*sqrt(mSSQ)*SQ*SS)/
                     (pow(eSQ + mQSQ + pSQ,2)*(eSQ + 2*(mSSQ + pSQ))) -
                     (8*sqrt(mQSQ)*sqrt(mSSQ)*SQ*SS)/((eSQ + mQSQ + pSQ)*(eSQ + 2*(mSSQ + pSQ))) -
                     (4*eSQ*sqrt(mQSQ)*sqrt(mSSQ)*(mQSQ + pSQ)*SQ*SS)/
                     (pow(eSQ + mQSQ + pSQ,2)*(mSSQ + pSQ)*(eSQ + 2*(mSSQ + pSQ))) -
                     (4*pow(eSQ,2)*pow(mQSQ,1.5)*sqrt(mSSQ)*(mSSQ + pSQ)*pow(SQ,3)*SS)/
                     (pSQ*(mQSQ + pSQ)*pow(eSQ + mQSQ + pSQ,2)*pow(eSQ + 2*(mSSQ + pSQ),2)) +
                     (4*eSQ*mSSQ*pow(SS,2))/((eSQ + mQSQ + pSQ)*pow(eSQ + 2*(mSSQ + pSQ),2)) -
                     (4*eSQ*mSSQ*(mQSQ + pSQ)*pow(SS,2))/
                     ((eSQ + mQSQ + pSQ)*(mSSQ + pSQ)*pow(eSQ + 2*(mSSQ + pSQ),2)) +
                     (2*eSQ*mSSQ*pow(mQSQ + pSQ,2)*pow(SS,2))/
                     (pow(eSQ + mQSQ + pSQ,2)*pow(mSSQ + pSQ,2)*(eSQ + 2*(mSSQ + pSQ))) -
                     (2*eSQ*mSSQ*(mQSQ + pSQ)*pow(SS,2))/
                     (pow(eSQ + mQSQ + pSQ,2)*(mSSQ + pSQ)*(eSQ + 2*(mSSQ + pSQ))) +
                     (4*mSSQ*(mQSQ + pSQ)*pow(SS,2))/
                     ((eSQ + mQSQ + pSQ)*(mSSQ + pSQ)*(eSQ + 2*(mSSQ + pSQ))) +
                     (6*pow(eSQ,2)*mQSQ*mSSQ*pow(SQ,2)*pow(SS,2))/
                     (pSQ*pow(eSQ + mQSQ + pSQ,2)*pow(eSQ + 2*(mSSQ + pSQ),2)) -
                     (4*pow(eSQ,2)*sqrt(mQSQ)*pow(mSSQ,1.5)*(mQSQ + pSQ)*SQ*pow(SS,3))/
                     (pSQ*pow(eSQ + mQSQ + pSQ,2)*(mSSQ + pSQ)*pow(eSQ + 2*(mSSQ + pSQ),2)) +
                     (pow(eSQ,2)*pow(mSSQ,2)*pow(mQSQ + pSQ,2)*pow(SS,4))/
                     (pSQ*pow(eSQ + mQSQ + pSQ,2)*pow(mSSQ + pSQ,2)*
                     pow(eSQ + 2*(mSSQ + pSQ),2))))/(64.*pow(M_PI,2));
   return d_beta_f;
}

int
main (void)
{
  gsl_integration_workspace * w = gsl_integration_workspace_alloc (1000);
  
  double result, error;
  gsl_function F;
  
  /* This is where we set the parameters we want to scan over */
  double vfSQ = 1.0;
  struct input_params params = { vfSQ, 3.0, 5.7, 2.89, 1.44, 1.0, -1, 1 };
  double MWSQ = 0.00192214;
  
  /* Now we just have the perform the integrals */
  F.function = &d_gamma_g;
  F.params = &params;
  gsl_integration_qagiu (&F, 0, 0, 1e-7, 1000, w, &result, &error);
  double gamma_g = result;
  
  F.function = &d_beta_g;
  F.params = &params;
  gsl_integration_qagiu (&F, MWSQ, 0, 1e-7, 1000, w, &result, &error);
  double beta_g = result;
  
  F.function = &d_gamma_f;
  F.params = &params;
  gsl_integration_qagiu (&F, 0, 0, 1e-7, 1000, w, &result, &error);
  double gamma_f = result;
  
  F.function = &d_beta_f;
  F.params = &params;
  gsl_integration_qagiu (&F, MWSQ, 0, 1e-7, 1000, w, &result, &error);
  double beta_f = result;
  
  gsl_integration_workspace_free (w);
  
  printf ("gamma_g = % .18f\n", gamma_g);
  printf ("beta_g  = % .18f\n", beta_g);
  printf ("gamma_f = % .18f\n", gamma_f);
  printf ("beta_f  = % .18f\n", beta_f);
  
  /* Finally output the Higgs VEV and mass (if EWSB happens) */
  if ( 0<gamma_g+gamma_f && gamma_g+gamma_f<2*(beta_g+beta_f) ) {
     double xi = (gamma_g+gamma_f) / (2*(beta_g+beta_f));
     double v = 1e3*sqrt( vfSQ*xi );
     double mH = 1e3*sqrt( 8*(beta_g+beta_f)*xi*(1-xi) / vfSQ );
     printf ("v   = % .18f GeV\n", v);
     printf ("mH  = % .18f GeV\n", mH);
  }
  else {
     printf ("No EWSB\n");
  }

  return 0;
}