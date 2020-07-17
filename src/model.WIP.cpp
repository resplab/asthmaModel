// -*- mode: C++; c-indent-level: 4; c-basic-offset: 4; indent-tabs-mode: nil; -*-

#include <RcppArmadillo.h>
// [[Rcpp::depends(RcppArmadillo)]]
using namespace Rcpp;

/*
Layout:
1. Basic
2. Settings
3. Radom
4. Input
5. Output
6. Agent
7. Event
8. Model
*/


#define OUTPUT_EX_BIOMETRICS 1 //height, weight etc;
#define OUTPUT_EX_SMOKING 2
#define OUTPUT_EX_COPD 16
#define OUTPUT_EX_MORTALITY 64
#define OUTPUT_EX_POPULATION 256

#define OUTPUT_EX 65535


#define MAX_AGE 111


enum errors
{
ERR_INCORRECT_SETTING_VARIABLE=-1,
ERR_INCORRECT_VECTOR_SIZE=-2,
ERR_INCORRECT_INPUT_VAR=-3,
ERR_EVENT_STACK_FULL=-4,
ERR_MEMORY_ALLOCATION_FAILED=-5
} errors;
/*** R
errors<-c(
  ERR_INCORRECT_SETTING_VARIABLE=-1,
  ERR_INCORRECT_VECTOR_SIZE=-2,
  ERR_INCORRECT_INPUT_VAR=-3,
  ERR_EVENT_STACK_FULL=-4,
  ERR_MEMORY_ALLOCATION_FAILED=-5
)
*/



enum record_mode
{
record_mode_none=0,
record_mode_agent=1,
record_mode_event=2,
record_mode_some_event=3
};
/*** R
record_mode<-c(
  record_mode_none=0,
  record_mode_agent=1,
  record_mode_event=2,
  record_mode_some_event=3
)
*/



enum agent_creation_mode
{
agent_creation_mode_one=0,
agent_creation_mode_all=1,
agent_creation_mode_pre=2
};
/*** R
agent_creation_mode<-c(
  agent_creation_mode_one=0,
  agent_creation_mode_all=1,
  agent_creation_mode_pre=2
)
*/
























/////////////////////////////////////////////////////////////////////BASICS//////////////////////////////////////////////
#define max(a,b)            \
({ __typeof__ (a) _a = (a); \
__typeof__ (b) _b = (b);  \
_a > _b ? _a : _b; })     \


#define min(a,b)              \
({ __typeof__ (a) _a = (a); \
__typeof__ (b) _b = (b);    \
_a > _b ? _b : _a; })       \


double calendar_time;
int last_id;

//' Samples from a multivariate normal
//' @param n number of samples to be taken
//' @param mu the mean
//' @param sigma the covariance matrix
//' @return the multivariate normal sample
//' @export
// [[Rcpp::export]]
arma::mat mvrnormArma(int n, arma::vec mu, arma::mat sigma) {
  int ncols = sigma.n_cols;
  arma::mat Y = arma::randn(n, ncols);
  return arma::repmat(mu, 1, n).t() + Y * arma::chol(sigma);
}


NumericMatrix array_to_Rmatrix(std::vector<double> x, int nCol)
{
  int nRow=x.size()/nCol;
  NumericMatrix y(nRow,nCol);
  //important: CPP is column major order but R is row major; we address the R matrix cell by cell but handle the vector in CPP way;
  for(int i=0;i<nRow;i++)
    for(int j=0;j<nCol;j++)
      y(i,j)=x[i*nCol+j];
  return (y);
}


NumericMatrix array_to_Rmatrix(std::vector<int> x, int nCol)
{
  int nRow=x.size()/nCol;
  NumericMatrix y(nRow,nCol);
  //important: CPP is column major order but R is row major; we address the R matrix cell by cell but handle the vector in CPP way;
  for(int i=0;i<nRow;i++)
    for(int j=0;j<nCol;j++)
      y(i,j)=x[i*nCol+j];
  return (y);
}










#define AS_VECTOR_DOUBLE(src) std::vector<double>(&src[0],&src[0]+sizeof(src)/sizeof(double))
#define AS_VECTOR_DOUBLE_SIZE(src,size) std::vector<double>(&src[0],&src[0]+size)

#define AS_MATRIX_DOUBLE(src)  array_to_Rmatrix(std::vector<double>(&src[0][0],&src[0][0]+sizeof(src)/sizeof(double)),sizeof(src[0])/sizeof(double))
#define AS_MATRIX_DOUBLE_SIZE(src,size)  array_to_Rmatrix(std::vector<double>(&src[0][0],&src[0][0]+size*sizeof(src[0])/sizeof(double)),sizeof(src[0])/sizeof(double))

#define AS_MATRIX_INT(src)  array_to_Rmatrix(std::vector<int>(&src[0][0],&src[0][0]+sizeof(src)/sizeof(int)),sizeof(src[0])/sizeof(int))
#define AS_MATRIX_INT_SIZE(src,size)  array_to_Rmatrix(std::vector<int>(&src[0][0],&src[0][0]+size*sizeof(src[0])/sizeof(int)),sizeof(src[0])/sizeof(int))

#define AS_VECTOR_INT(src) std::vector<int>(&src[0],&src[0]+sizeof(src)/sizeof(int))
#define AS_VECTOR_INT_SIZE(src,size) std::vector<int>(&src[0],&src[0]+size)

#define READ_R_VECTOR(src,dest) {if(src.size()==sizeof(dest)/sizeof(dest[0])) {std::copy(src.begin(),src.end(),&dest[0]); return(0);} else return(ERR_INCORRECT_VECTOR_SIZE);}

#define READ_R_MATRIX(src,dest) {if(src.size()==sizeof(dest)/sizeof(dest[0][0])) {std::copy(src.begin(),src.end(),&dest[0][0]); return(0);} else return(ERR_INCORRECT_VECTOR_SIZE);}



//////////////////////////////////////////////////////////////////////SETTINGS//////////////////////////////////////////////////



struct settings
{
  int record_mode;    //0: nothing recorded, 1:agents recorded, 2: events recorded, 3:selected events recorded;

  int events_to_record[100];  //valid only if record_mode=3
  int n_events_to_record;

  int agent_creation_mode; //0: one agent at a time; 1: agents are created and saved in agent_stack. 2: saved agents in agent_stack are used (require create_agents before running the model)

  int update_continuous_outcomes_mode;    //0:update only on fixed and end events; 1: update before any event;

  int n_base_agents;

  int runif_buffer_size;
  int rnorm_buffer_size;
  int rexp_buffer_size;

  int agent_stack_size;
  int event_stack_size;
} settings;


//' Sets model settings.
//' @param name a name
//' @param value a value
//' @return 0 if successful.
//' @export
// [[Rcpp::export]]
int Cset_settings_var(std::string name,NumericVector value)
{
  if(name=="record_mode") {settings.record_mode=value[0]; return(0);}
  if(name=="events_to_record")
  {
    settings.n_events_to_record=0;
    for(int i=0;i<value.size();i++)
    {
      settings.events_to_record[i]=value[i];
      settings.n_events_to_record++;
    }
    return(0);
  }
  if(name=="agent_creation_mode") {settings.agent_creation_mode=value[0]; return(0);}
  if(name=="update_continuous_outcomes_mode") {settings.update_continuous_outcomes_mode=value[0]; return(0);}
  if(name=="n_base_agents") {settings.n_base_agents=value[0]; return(0);}
  if(name=="runif_buffer_size") {settings.runif_buffer_size=value[0]; return(0);}
  if(name=="rnorm_buffer_size") {settings.rnorm_buffer_size=value[0]; return(0);}
  if(name=="rexp_buffer_size") {settings.rexp_buffer_size=value[0]; return(0);}
  if(name=="agent_stack_size") {settings.agent_stack_size=value[0]; return(0);}
  if(name=="event_stack_size") {settings.event_stack_size=value[0]; return(0);}
  return(ERR_INCORRECT_SETTING_VARIABLE);
}

//' Returns current settings.
//' @return current settings.
//' @export
// [[Rcpp::export]]
List Cget_settings()
{
  return Rcpp::List::create(
    Rcpp::Named("record_mode")=settings.record_mode,
    Rcpp::Named("events_to_record")=AS_VECTOR_DOUBLE_SIZE(settings.events_to_record,settings.n_events_to_record),
    Rcpp::Named("agent_creation_mode")=settings.agent_creation_mode,
    Rcpp::Named("update_continuous_outcomes_mode")=settings.update_continuous_outcomes_mode,
    Rcpp::Named("n_base_agents")=settings.n_base_agents,
    Rcpp::Named("runif_buffer_size")=settings.runif_buffer_size,
    Rcpp::Named("rnorm_buffer_size")=settings.rnorm_buffer_size,
    Rcpp::Named("rexp_buffer_size")=settings.rexp_buffer_size,
    Rcpp::Named("agent_stack_size")=settings.agent_stack_size,
    Rcpp::Named("event_stack_size")=settings.event_stack_size
  );
}


struct runtime_stats
{
  int agent_size;
  int n_runif_fills;
  int n_rnorm_fills;
  int n_rexp_fills;
} runtime_stats;


void reset_runtime_stats()
{
  char *x=reinterpret_cast <char *>(&runtime_stats);
  for(unsigned i=0;i<sizeof(runtime_stats);i++)
    x[i]=0;
}

//' Returns run time stats.
//' @return agent size as well as memory and random variable fill stats.
//' @export
// [[Rcpp::export]]
List Cget_runtime_stats()
{
  return Rcpp::List::create(
    Rcpp::Named("agent_size")=runtime_stats.agent_size,
    Rcpp::Named("n_runif_fills")=runtime_stats.n_runif_fills,
    Rcpp::Named("n_rnorm_fills")=runtime_stats.n_rnorm_fills,
    Rcpp::Named("n_rexp_fills")=runtime_stats.n_rexp_fills
  );
}





////////////////////////////////////////////////////////////////////RANDOM/////////////////////////////////////////////////
//these stuff are internal so no expoert/import;
double *runif_buffer;
long runif_buffer_pointer;

double *rnorm_buffer;
long rnorm_buffer_pointer;

double *rexp_buffer;
long rexp_buffer_pointer;




double* R_runif(int n)
{
  NumericVector temp(runif(n));
  return(&temp(0));
}
double* R_runif(int n, double * address)
{
  NumericVector temp(runif(n));
  std::copy(temp.begin(),temp.end(),address);
  return(address);
}
int runif_fill()
{
  R_runif(settings.runif_buffer_size,runif_buffer);
  runif_buffer_pointer=0;
  ++runtime_stats.n_runif_fills;
  return(0);
}
double rand_unif()
{
  if(runif_buffer_pointer==settings.runif_buffer_size) {runif_fill();}
  double temp=runif_buffer[runif_buffer_pointer];
  runif_buffer_pointer++;
  return(temp);
}




double* R_rnorm(int n)
{
  NumericVector temp(rnorm(n));
  return(&temp(0));
}
double* R_rnorm(int n, double * address)
{
  NumericVector temp(rnorm(n));
  std::copy(temp.begin(),temp.end(),address);
  return(address);
}
int rnorm_fill()
{
  R_rnorm(settings.rnorm_buffer_size,rnorm_buffer);
  rnorm_buffer_pointer=0;
  ++runtime_stats.n_rnorm_fills;
  return(0);
}
double rand_norm()
{
  if(rnorm_buffer_pointer==settings.rnorm_buffer_size) {rnorm_fill();}
  double temp=rnorm_buffer[rnorm_buffer_pointer];
  rnorm_buffer_pointer++;
  return(temp);
}

//bivariate normal, method 1: standard normal with rho;
void rbvnorm(double rho, double x[2])
{
  x[0]=rand_norm();
  double mu=rho*x[0];
  double v=(1-rho*rho);

  x[1]=rand_norm()*sqrt(v)+mu;
}





double* R_rexp(int n)
{
  NumericVector temp(rexp(n,1));
  return(&temp(0));
}
double* R_rexp(int n, double * address)
{
  NumericVector temp(rexp(n,1));
  std::copy(temp.begin(),temp.end(),address);
  return(address);
}
int rexp_fill()
{
  R_rexp(settings.rexp_buffer_size,rexp_buffer);
  rexp_buffer_pointer=0;
  ++runtime_stats.n_rexp_fills;
  return(0);
}
double rand_exp()
{
  if(rexp_buffer_pointer==settings.rexp_buffer_size) {rexp_fill();}
  double temp=rexp_buffer[rexp_buffer_pointer];
  rexp_buffer_pointer++;
  return(temp);
}

// [[Rcpp::export]]
NumericVector Xrexp(int n, double rate)
{
  double *temp=R_rexp(n);
  return(temp[0]/rate);
}













////////////////////////////////////////////////////////////////////INPUT/////////////////////////////////////////////
struct input
{
  struct
  {
    int time_horizon;
    double y0;  //calendar year
    double age0;  //age to start
    double discount_cost;
    double discount_qaly;
  } global_parameters;

  struct
  {
    double p_female;

    double height_0_betas[5]; //intercept, sex, age, age2, sex*age;
    double height_0_sd;
    double weight_0_betas[7]; //intercept, sex, age, age2, sex*age, height, year;
    double weight_0_sd;  //currently, a sample is made at baseline and every one is moved in parallel trajectories
    double height_weight_rho;

    double p_prevalence_age[111]; //age distribution of prevalent (time_at_creation=0) agents
    double p_incidence_age[111]; //age distribution of incident (time_at_creation>0) agents
    double l_inc_betas[3]; //intercept, calendar year and its square
    double p_bgd_by_sex[111][2];
    double ln_h_bgd_betas[9]; //intercept, calendar year, its square, age, b_mi, n_mi, b_stroke, n_stroke, hf_status
  } agent;


  struct
  {
    double ln_h_COPD_betas_by_sex[7][2]; //INCIDENCE: intercept, age, age2, pack_years, current_smoking, year, asthma;
    double logit_p_COPD_betas_by_sex[7][2]; //PREVALENCE: intercept, age, age2, pack_years, current_smoking, year, asthma;
  } COPD;


  struct
  {
    double bg_cost_by_stage[5];

  } cost;

  struct
  {
    double bg_util_by_stage[5];

  } utility;


  struct
  {
  } project_specific;


} input;




//' Returns inputs
//' @return all inputs
//' @export
// [[Rcpp::export]]
List Cget_inputs()
{
  List out=Rcpp::List::create(
    Rcpp::Named("global_parameters")=Rcpp::List::create(
      Rcpp::Named("age0")=input.global_parameters.age0,
      Rcpp::Named("time_horizon")=input.global_parameters.time_horizon,
      Rcpp::Named("y0")=input.global_parameters.y0,
      Rcpp::Named("discount_cost")=input.global_parameters.discount_cost,
      Rcpp::Named("discount_qaly")=input.global_parameters.discount_qaly
    ),
    Rcpp::Named("agent")=Rcpp::List::create(
      Rcpp::Named("p_female")=input.agent.p_female,
      Rcpp::Named("height_0_betas")=AS_VECTOR_DOUBLE(input.agent.height_0_betas),
      Rcpp::Named("height_0_sd")=input.agent.height_0_sd,
      Rcpp::Named("weight_0_betas")=AS_VECTOR_DOUBLE(input.agent.weight_0_betas),
      Rcpp::Named("weight_0_sd")=input.agent.weight_0_sd,
      Rcpp::Named("height_weight_rho")=input.agent.height_weight_rho,

      Rcpp::Named("p_prevalence_age")=AS_VECTOR_DOUBLE(input.agent.p_prevalence_age),
      Rcpp::Named("p_incidence_age")=AS_VECTOR_DOUBLE(input.agent.p_incidence_age),
      Rcpp::Named("p_bgd_by_sex")=AS_MATRIX_DOUBLE(input.agent.p_bgd_by_sex),
      Rcpp::Named("l_inc_betas")=AS_VECTOR_DOUBLE(input.agent.l_inc_betas),
      Rcpp::Named("ln_h_bgd_betas")=AS_VECTOR_DOUBLE(input.agent.ln_h_bgd_betas)
    ),
    Rcpp::Named("COPD")=Rcpp::List::create(
      Rcpp::Named("ln_h_COPD_betas_by_sex")=AS_MATRIX_DOUBLE(input.COPD.ln_h_COPD_betas_by_sex),
      Rcpp::Named("logit_p_COPD_betas_by_sex")=AS_MATRIX_DOUBLE(input.COPD.logit_p_COPD_betas_by_sex)
    ),

    Rcpp::Named("cost")=Rcpp::List::create(
      Rcpp::Named("bg_cost_by_stage")=AS_VECTOR_DOUBLE(input.cost.bg_cost_by_stage)
    ),
    Rcpp::Named("utility")=Rcpp::List::create(
      Rcpp::Named("bg_util_by_stage")=AS_VECTOR_DOUBLE(input.utility.bg_util_by_stage)
    )
  ,

  Rcpp::Named("project_specific")=Rcpp::List::create(
    //Put your project-specific outputs here;
  )
  );

  return(out);
}


//' Sets input variables.
//' @param name a string
//' @param value a number
//' @return 0 if successful
//' @export
// [[Rcpp::export]]
int Cset_input_var(std::string name, NumericVector value)
{
  if(name=="global_parameters$age0") {input.global_parameters.age0=value[0]; return(0);}
  if(name=="global_parameters$time_horizon")  {input.global_parameters.time_horizon=value[0]; return(0);}
  if(name=="global_parameters$discount_cost") {input.global_parameters.discount_cost=value[0]; return(0);}
  if(name=="global_parameters$discount_qaly") {input.global_parameters.discount_qaly=value[0]; return(0);}

  if(name=="agent$p_female") {input.agent.p_female=value[0]; return(0);}
  if(name=="agent$p_prevalence_age") READ_R_VECTOR(value,input.agent.p_prevalence_age);
  if(name=="agent$height_0_betas") READ_R_VECTOR(value,input.agent.height_0_betas);
  if(name=="agent$height_0_sd") {input.agent.height_0_sd=value[0]; return(0);}
  if(name=="agent$weight_0_betas") READ_R_VECTOR(value,input.agent.weight_0_betas);
  if(name=="agent$weight_0_sd") {input.agent.weight_0_sd=value[0]; return(0);}
  if(name=="agent$height_weight_rho") {input.agent.height_weight_rho=value[0]; return(0);}

  if(name=="agent$p_incidence_age") READ_R_VECTOR(value,input.agent.p_incidence_age);
  if(name=="agent$p_bgd_by_sex") READ_R_MATRIX(value,input.agent.p_bgd_by_sex);
  if(name=="agent$l_inc_betas") READ_R_VECTOR(value,input.agent.l_inc_betas);
  if(name=="agent$ln_h_bgd_betas") READ_R_VECTOR(value,input.agent.ln_h_bgd_betas);

  if(name=="COPD$ln_h_COPD_betas_by_sex") READ_R_MATRIX(value,input.COPD.ln_h_COPD_betas_by_sex);
  if(name=="COPD$logit_p_COPD_betas_by_sex") READ_R_MATRIX(value,input.COPD.logit_p_COPD_betas_by_sex);


  if(name=="utility$bg_util_by_stage") READ_R_VECTOR(value,input.utility.bg_util_by_stage);


  //Define your project-specific inputs here;

  return(ERR_INCORRECT_INPUT_VAR);
}


//' Returns a sample output for a given year and gender.
//' @param year a number
//' @param sex a number, 0 for male and 1 for female
//' @return that specific output
//' @export
// [[Rcpp::export]]
double get_sample_output(int year, int sex)
{
  return input.agent.p_bgd_by_sex[year][(int)sex];
}







/////////////////////////////////////////////////////////////////AGENT/////////////////////////////////////



struct agent
{
  long id;
  double local_time;
  bool alive;
  bool sex;

  double age_at_creation;
  double age_baseline; //Age at the time of getting COPD. Used for FEV1 Decline. Amin

  double time_at_creation;
  double followup_time; //Time since COPD?

  double height;
  double weight;
  double weight_LPT;
  double weight_baseline; //Weight at the time of getting COPD. Used for FEV1 Decline. Amin

  int smoking_status;   //0:not smoking, positive: current smoking rate (py per year), note that ex smoker status os determined also by pack_years
  double pack_years;

  int local_time_at_COPD;


  int cumul_exac[4];    //0:mild, 1:moderate, 2:severe, 3: very severe;
  double cumul_exac_time[4];

  double cumul_cost;
  double cumul_qaly;
  double annual_cost;
  double annual_qaly;

  double payoffs_LPT;

  double tte;
  int event; //carries the last event;

  double p_COPD;  //Not used in the model; here to facilitate calibration on incidence;

};


agent *agent_stack;
long agent_stack_pointer;

List get_agent(agent *ag)
{
  List out=Rcpp::List::create(
    Rcpp::Named("id")=(*ag).id,
    Rcpp::Named("local_time")=(*ag).local_time,
    Rcpp::Named("alive")=(*ag).alive,
    Rcpp::Named("sex")=(int)(*ag).sex,

    Rcpp::Named("height")=(*ag).height,
    Rcpp::Named("weight")=(*ag).weight,

    Rcpp::Named("age_at_creation")=(*ag).age_at_creation,
    Rcpp::Named("time_at_creation")=(*ag).time_at_creation

  );
  out["weight_baseline"] = (*ag).weight_baseline; //added here because the function "create" above can take a limited number of arguments
  out["followup_time"] = (*ag).followup_time; //added here because the function "create" above can take a limited number of arguments
  out["age_baseline"] = (*ag).age_baseline; //added here because the function "create" above can take a limited number of arguments

  out["local_time_at_COPD"]=(*ag).local_time_at_COPD;

  out["cumul_cost"] = (*ag).cumul_cost;
  out["cumul_qaly"] = (*ag).cumul_qaly;
  out["annual_cost"] = (*ag).annual_cost;
  out["annual_qaly"] = (*ag).annual_qaly;

  out["tte"] = (*ag).tte;
  out["event"] = (*ag).event;

  out["p_COPD"] = (*ag).p_COPD;

  return out;
}





//This is a generic function as both agent_stack and event_stack are arrays of agents;
List get_agent(int id, agent agent_pointer[])
{
  return(get_agent(&agent_pointer[id]));
}



// [[Rcpp::export]]
List Cget_agent(long id)
{
  return(get_agent(id,agent_stack));
}

//' Returns agent Smith.
//' @return agent smith.
//' @export
// [[Rcpp::export]]
List Cget_smith()
{
  return(get_agent(&smith));
}


agent *create_agent(agent *ag,int id)
{
double _bvn[2]; //being used for joint estimation in multiple locations;

(*ag).id=id;
(*ag).alive=1;
(*ag).local_time=0;
(*ag).age_baseline = 0;

(*ag).weight_baseline = 0; //resetting the value for new agent
(*ag).followup_time = 0; //resetting the value for new agent
(*ag).local_time_at_COPD = 0; //resetting the value for new agent

(*ag).time_at_creation=calendar_time;
(*ag).sex=rand_unif()<input.agent.p_female;

double r=rand_unif();
double cum_p=0;

if(id<settings.n_base_agents) //the first n_base_agent cases are prevalent cases; the rest are incident ones;
  for(int i=input.global_parameters.age0;i<111;i++)
  {
    cum_p=cum_p+input.agent.p_prevalence_age[i];
    if(r<cum_p) {(*ag).age_at_creation=i+rand_unif(); break;}
  }
  else
    for(int i=input.global_parameters.age0;i<111;i++)
    {
      cum_p=cum_p+input.agent.p_incidence_age[i];
      if(r<cum_p) {(*ag).age_at_creation=i; break;}
    }

    rbvnorm(input.agent.height_weight_rho,_bvn);
  (*ag).height=_bvn[0]*input.agent.height_0_sd
    +input.agent.height_0_betas[0]
  +input.agent.height_0_betas[1]*(*ag).sex
  +input.agent.height_0_betas[2]*(*ag).age_at_creation
  +input.agent.height_0_betas[3]*(*ag).age_at_creation*(*ag).age_at_creation
  +input.agent.height_0_betas[4]*(*ag).age_at_creation*(*ag).sex;

  (*ag).weight=_bvn[1]*input.agent.weight_0_sd
    +input.agent.weight_0_betas[0]
  +input.agent.weight_0_betas[1]*(*ag).sex
  +input.agent.weight_0_betas[2]*(*ag).age_at_creation
  +input.agent.weight_0_betas[3]*(*ag).age_at_creation*(*ag).age_at_creation
  +input.agent.weight_0_betas[4]*(*ag).age_at_creation*(*ag).sex
  +input.agent.weight_0_betas[5]*(*ag).height
  +input.agent.weight_0_betas[6]*calendar_time;


  //COPD;
  double COPD_odds=exp(input.COPD.logit_p_COPD_betas_by_sex[0][(*ag).sex]
                         +input.COPD.logit_p_COPD_betas_by_sex[1][(*ag).sex]*(*ag).age_at_creation
                         +input.COPD.logit_p_COPD_betas_by_sex[2][(*ag).sex]*(*ag).age_at_creation*(*ag).age_at_creation
                         +input.COPD.logit_p_COPD_betas_by_sex[3][(*ag).sex]*(*ag).pack_years
                         +input.COPD.logit_p_COPD_betas_by_sex[4][(*ag).sex]*(*ag).smoking_status
                         +input.COPD.logit_p_COPD_betas_by_sex[5][(*ag).sex]*calendar_time)
                         //+input.COPD.logit_p_COPD_betas_by_sex[7]*(*ag).asthma
                         ;

  (*ag).p_COPD=COPD_odds/(1+COPD_odds);

  if(rand_unif()<COPD_odds/(1+COPD_odds))
  {
    (*ag).weight_baseline = (*ag).weight;
    (*ag).age_baseline = (*ag).local_time + (*ag).age_at_creation;
    (*ag).followup_time = 0 ;
    (*ag).local_time_at_COPD = (*ag).local_time;

  }


  //payoffs;
  (*ag).cumul_cost=0;
  (*ag).cumul_qaly=0;
  (*ag).annual_cost=0;
  (*ag).annual_qaly=0;

  (*ag).payoffs_LPT=0;

  return(ag);
}



// [[Rcpp::export]]
int Ccreate_agents()
{
  if(agent_stack==NULL) return(-1);
  for(int i=0;i<settings.agent_stack_size;i++)
  {
    create_agent(&agent_stack[i],i);
  }

  return(0);
}






/////////////////////////////////////////////////////////////////////////OUTPUT/////////////////////////////////////////////////

struct output
{
  int n_agents;
  double cumul_time;    //End variable by nature;
  int n_deaths;         //End variable by nature.
  int n_COPD;
  double total_pack_years;    //END  because agent records
  int total_exac[4];    //0:mild, 1:moderate, 2:severe; 3=very severe    END because agent records
  double total_exac_time[4];  //END because agent records

  double total_cost;    //END because agent records
  double total_qaly;  //END because agent records
} output;




void reset_output()
{
  output.n_agents=0;
  output.cumul_time=0;
  output.n_deaths=0;
  output.n_COPD=0;
  output.total_pack_years=0;
  output.total_exac[0]=0;output.total_exac[1]=0;output.total_exac[2]=0;output.total_exac[3]=0;
  output.total_exac_time[0]=0;output.total_exac_time[1]=0;output.total_exac_time[2]=0;output.total_exac_time[3]=0;
  output.total_cost=0;
  output.total_qaly=0;
}

//' Main outputs of the current run.
//' @return number of agents, cumulative time, number of deaths, number of COPD cases, QALYs.
//' @export
// [[Rcpp::export]]
List Cget_output()
{
  return Rcpp::List::create(
    Rcpp::Named("n_agents")=output.n_agents,
    Rcpp::Named("cumul_time")=output.cumul_time,
    Rcpp::Named("n_deaths")=output.n_deaths,
    Rcpp::Named("n_COPD")=output.n_COPD,
    Rcpp::Named("total_pack_years")=output.total_pack_years,
    Rcpp::Named("total_cost")=output.total_cost,
    Rcpp::Named("total_qaly")=output.total_qaly
  //Define your project-specific output here;
  );
}








#ifdef OUTPUT_EX

struct output_ex
{
  int n_alive_by_ctime_sex[1000][2];      //number of folks alive at each fixed time;
  int n_smoking_status_by_ctime[1000][3];
  int n_alive_by_ctime_age[1000][111];
  int n_current_smoker_by_ctime_sex[1000][2];
  double cumul_cost_ctime[1000];

  double cumul_qaly_ctime[1000];

  double cumul_time_by_smoking_status[3];
  double sum_time_by_ctime_sex[100][2];
  double sum_time_by_age_sex[111][2];

#if OUTPUT_EX > 1
  double cumul_non_COPD_time;
  double sum_p_COPD_by_ctime_sex[1000][2];
  double sum_pack_years_by_ctime_sex[1000][2];
  double sum_age_by_ctime_sex[1000][2];
  int n_death_by_age_sex[111][2];
  int n_alive_by_age_sex[111][2];
#endif

#if (OUTPUT_EX & OUTPUT_EX_COPD) > 0
  int n_COPD_by_ctime_sex[1000][2];
  int n_COPD_by_ctime_age[100][111];
  int n_inc_COPD_by_ctime_age[100][111];
  int n_COPD_by_ctime_severity[100][5]; //no COPD to GOLD 4;
  int n_COPD_by_age_sex[111][2];

#endif


#if (OUTPUT_EX & OUTPUT_EX_BIOMETRICS) > 0
  double sum_weight_by_ctime_sex[1000][2];
#endif

} output_ex;
#endif



void reset_output_ex()
{
#ifdef OUTPUT_EX
  char *x=reinterpret_cast <char *>(&output_ex);
  for(unsigned i=0;i<sizeof(output_ex);i++)
    x[i]=0;
#endif
}

//' Extra outputs from the model
//' @return Extra outputs from the model.
//' @export
// [[Rcpp::export]]
List Cget_output_ex()
{
  List out=Rcpp::List::create(
#ifdef OUTPUT_EX
    Rcpp::Named("n_alive_by_ctime_sex")=AS_MATRIX_INT_SIZE(output_ex.n_alive_by_ctime_sex,input.global_parameters.time_horizon),
    Rcpp::Named("n_alive_by_ctime_age")=AS_MATRIX_INT_SIZE(output_ex.n_alive_by_ctime_age,input.global_parameters.time_horizon),
    Rcpp::Named("n_smoking_status_by_ctime")=AS_MATRIX_INT_SIZE(output_ex.n_smoking_status_by_ctime,input.global_parameters.time_horizon),
    Rcpp::Named("n_current_smoker_by_ctime_sex")=AS_MATRIX_INT_SIZE(output_ex.n_current_smoker_by_ctime_sex,input.global_parameters.time_horizon),
    Rcpp::Named("cumul_cost_ctime")=AS_VECTOR_DOUBLE_SIZE(output_ex.cumul_cost_ctime,input.global_parameters.time_horizon),
    Rcpp::Named("cumul_qaly_ctime")=AS_VECTOR_DOUBLE_SIZE(output_ex.cumul_qaly_ctime,input.global_parameters.time_horizon),
    Rcpp::Named("cumul_time_by_smoking_status")=AS_VECTOR_DOUBLE(output_ex.cumul_time_by_smoking_status),
    Rcpp::Named("cumul_non_COPD_time")=output_ex.cumul_non_COPD_time,
    Rcpp::Named("sum_p_COPD_by_ctime_sex")=AS_MATRIX_DOUBLE_SIZE(output_ex.sum_p_COPD_by_ctime_sex,input.global_parameters.time_horizon),
    Rcpp::Named("sum_pack_years_by_ctime_sex")=AS_MATRIX_DOUBLE_SIZE(output_ex.sum_pack_years_by_ctime_sex,input.global_parameters.time_horizon),
    Rcpp::Named("sum_age_by_ctime_sex")=AS_MATRIX_DOUBLE_SIZE(output_ex.sum_age_by_ctime_sex,input.global_parameters.time_horizon),
    Rcpp::Named("n_death_by_age_sex")=AS_MATRIX_INT(output_ex.n_death_by_age_sex),
    Rcpp::Named("n_alive_by_age_sex")=AS_MATRIX_INT(output_ex.n_alive_by_age_sex),
    Rcpp::Named("sum_time_by_ctime_sex")=AS_MATRIX_DOUBLE_SIZE(output_ex.sum_time_by_ctime_sex,input.global_parameters.time_horizon),
    Rcpp::Named("sum_time_by_age_sex")=AS_MATRIX_DOUBLE(output_ex.sum_time_by_age_sex)
#endif
#if (OUTPUT_EX & OUTPUT_EX_BIOMETRICS) > 0
  ,Rcpp::Named("sum_weight_by_ctime_sex")=AS_MATRIX_DOUBLE_SIZE(output_ex.sum_weight_by_ctime_sex,input.global_parameters.time_horizon)
#endif
  );


#if (OUTPUT_EX & OUTPUT_EX_COPD)>0
  out["n_COPD_by_ctime_sex"]=AS_MATRIX_INT_SIZE(output_ex.n_COPD_by_ctime_sex,input.global_parameters.time_horizon),
    out["n_COPD_by_ctime_age"]=AS_MATRIX_INT_SIZE(output_ex.n_COPD_by_ctime_age,input.global_parameters.time_horizon),
    out["n_inc_COPD_by_ctime_age"]=AS_MATRIX_INT_SIZE(output_ex.n_inc_COPD_by_ctime_age,input.global_parameters.time_horizon),
    out["n_COPD_by_ctime_severity"]=AS_MATRIX_INT_SIZE(output_ex.n_COPD_by_ctime_severity,input.global_parameters.time_horizon),
    out["n_COPD_by_age_sex"]=AS_MATRIX_INT(output_ex.n_COPD_by_age_sex);

#endif

    return(out);
}


//This function must run ONLY on start and fixed events; any other place and will mess up!
void update_output_ex(agent *ag)
{
#ifdef OUTPUT_EX
  int time=floor((*ag).local_time+(*ag).time_at_creation);
  int local_time=floor((*ag).local_time);

  //if(time>=(*ag).time_at_creation)
  {
    int age=floor((*ag).age_at_creation+(*ag).local_time);
    output_ex.n_alive_by_ctime_age[time][age-1]+=1;   //age-1 -> adjusting for zero based system in C.
    output_ex.n_alive_by_ctime_sex[time][(*ag).sex]+=1;
    output_ex.n_alive_by_age_sex[age-1][(*ag).sex]+=1;
    if((*ag).smoking_status==1)
    {
      output_ex.n_smoking_status_by_ctime[time][1]+=1;
      output_ex.n_current_smoker_by_ctime_sex[time][(*ag).sex]+=1;
    }
    else
      if((*ag).pack_years>0)
        output_ex.n_smoking_status_by_ctime[time][2]+=1;
      else
        output_ex.n_smoking_status_by_ctime[time][0]+=1;

      output_ex.cumul_cost_ctime[time]+=(*ag).annual_cost;
      output_ex.cumul_qaly_ctime[time]+=(*ag).annual_qaly;

      double odds=exp(input.COPD.logit_p_COPD_betas_by_sex[0][(*ag).sex]
                        +input.COPD.logit_p_COPD_betas_by_sex[1][(*ag).sex]*((*ag).age_at_creation+(*ag).local_time)
                        +input.COPD.logit_p_COPD_betas_by_sex[2][(*ag).sex]*pow((*ag).age_at_creation+(*ag).local_time,2)
                        +input.COPD.logit_p_COPD_betas_by_sex[3][(*ag).sex]*(*ag).pack_years
                        +input.COPD.logit_p_COPD_betas_by_sex[4][(*ag).sex]*(*ag).smoking_status
                        +input.COPD.logit_p_COPD_betas_by_sex[5][(*ag).sex]*(calendar_time+(*ag).local_time)
      );
      output_ex.sum_p_COPD_by_ctime_sex[time][(*ag).sex]+=odds/(1+odds);
      output_ex.sum_pack_years_by_ctime_sex[time][(*ag).sex]+=(*ag).pack_years;
      output_ex.sum_age_by_ctime_sex[time][(*ag).sex]+=(*ag).age_at_creation+(*ag).local_time;

#if (OUTPUT_EX & OUTPUT_EX_BIOMETRICS)>0
      output_ex.sum_weight_by_ctime_sex[time][(*ag).sex]+=(*ag).weight;
#endif

#if (OUTPUT_EX & OUTPUT_EX_COPD)>0

      output_ex.n_COPD_by_age_sex[age-1][(*ag).sex]+=1;
#endif

  }
#endif
}







/////////////////////////////////////////////////////////////////////////LPTs////////////////////////////////////////////////////////////////////////////////


void payoffs_LPT(agent *ag)
{


  (*ag).payoffs_LPT=(*ag).local_time;
}


///////////////////////////////////////////////////////////////////EVENT/////////////////////////////////////////////////////////



enum events
{
  event_start=0,
  event_fixed=1,
  event_birthday=2,
  event_smoking_change=3,
  event_COPD=4,

  event_hf=12,
  event_bgd=13,
  event_end=14
};
/*** R
events<-c(
    event_start=0,
    event_fixed=1,
    event_birthday=2,
    event_smoking_change=3,
    event_COPD=4,

    event_bgd=13,
    event_end=14
)
  */








agent *event_start_process(agent *ag)
{
#ifdef OUTPUT_EX
  update_output_ex(ag);
#endif
  return(ag);
}


agent *event_end_process(agent *ag)
{

  ++output.n_agents;
  output.cumul_time+=(*ag).local_time;
  output.n_deaths+=!(*ag).alive;


  payoffs_LPT(ag);

  output.total_pack_years+=(*ag).pack_years;
  output.total_exac[0]+=(*ag).cumul_exac[0];
  output.total_exac[1]+=(*ag).cumul_exac[1];
  output.total_exac[2]+=(*ag).cumul_exac[2];
  output.total_exac[3]+=(*ag).cumul_exac[3];

  output.total_exac_time[0]+=(*ag).cumul_exac_time[0];
  output.total_exac_time[1]+=(*ag).cumul_exac_time[1];
  output.total_exac_time[2]+=(*ag).cumul_exac_time[2];
  output.total_exac_time[3]+=(*ag).cumul_exac_time[3];


  output.total_cost+=(*ag).cumul_cost;
  output.total_qaly+=(*ag).cumul_qaly;



#ifdef OUTPUT_EX
  //NO!!! We do not update many of output_ex stuff here. It might fall within the same calendar year of the last fixed event and results in double counting.
  //If it falls after that still we ignore as it is a partially observed year.
#endif
#if OUTPUT_EX>1

  int age=floor((*ag).local_time+(*ag).age_at_creation);
  //Rprintf("age at death=%f\n",age);
  if((*ag).alive==false)  output_ex.n_death_by_age_sex[age-1][(*ag).sex]+=1;

  double time=(*ag).time_at_creation+(*ag).local_time;
  while(time>(*ag).time_at_creation)
  {
    int time_cut=floor(time);
    double delta=min(time-time_cut,time-(*ag).time_at_creation);
    if(delta==0) {time_cut-=1; delta=min(time-time_cut,time-(*ag).time_at_creation);}
    output_ex.sum_time_by_ctime_sex[time_cut][(*ag).sex]+=delta;
    time-=delta;
  }


  //double _age=(*ag).age_at_creation+(*ag).local_time;
  //while(_age>(*ag).age_at_creation)
  //{
  //  int age_cut=floor(_age);
  //  double delta=min(_age-age_cut,_age-(*ag).age_at_creation);
  //  if(delta==0) {age_cut-=1; delta=min(_age-age_cut,_age-(*ag).age_at_creation);}
  //  output_ex.sum_time_by_age_sex[age_cut][(*ag).sex]+=delta;
  //  _age-=delta;
  //}

  double _age=(*ag).age_at_creation+(*ag).local_time;
  int _low=floor((*ag).age_at_creation);
  int _high=ceil(_age);
  for(int i=_low;i<=_high;i++)
  {
    double delta=min(i+1,_age)-max(i,(*ag).age_at_creation);
    if(delta>1e-10) {
      output_ex.sum_time_by_age_sex[i-1][(*ag).sex]+=delta;
    }
  }

#endif

  return(ag);
}



agent *event_stack;
int event_stack_pointer;


int push_event(agent *ag)
{
  if(event_stack_pointer==settings.event_stack_size) return(ERR_EVENT_STACK_FULL);
  if(settings.record_mode==record_mode_some_event)
  {
    int i;
    for(i=0;i<settings.n_events_to_record;i++)
    {
      if(settings.events_to_record[i]==(*ag).event)
      {
        event_stack[event_stack_pointer]=*ag;
        ++event_stack_pointer;
        return 0;
      }
    }
    return 0;
  }
  event_stack[event_stack_pointer]=*ag;
  ++event_stack_pointer;
  return(0);
}


//' Returns the events stack.
//' @param i number of event
//' @return events
//' @export
// [[Rcpp::export]]
List Cget_event(int i)
{
  return(get_agent(i,event_stack));
}

//' Returns total number of events.
//' @return number of events
//' @export
// [[Rcpp::export]]
int Cget_n_events() //number of events, not n events themselves;
  {
  return(event_stack_pointer);
  }

//' Returns all events of an agent.
//' @param id agent ID.
//' @return all events of agent \code{id}
//' @export
// [[Rcpp::export]]
DataFrame Cget_agent_events(int id) //Returns ALLva events of an agent;
  {
  DataFrame dfout;

  for(int i=0;i<event_stack_pointer;i++)
  {
    if(event_stack[i].id==id)
    {
      dfout.push_back(get_agent(i,event_stack));
    }
  }
  return(dfout);
  }


//' Returns all events of a certain type.
//' @param event_type a number
//' @return all events of the type \code{event_type}
//' @export
// [[Rcpp::export]]
DataFrame Cget_events_by_type(int event_type) //Returns all events of a given type for an agent;
  {
  DataFrame dfout;

  for(int i=0;i<event_stack_pointer;i++)
  {
    if(event_stack[i].event==event_type)
    {
      dfout.push_back(get_agent(i,event_stack));
    }
  }
  return(dfout);
  }


//' Returns all events.
//' @return all events
//' @export
// [[Rcpp::export]]
DataFrame Cget_all_events() //Returns all events from all agents;
  {
  DataFrame dfout;

  for(int i=0;i<event_stack_pointer;i++)
  {
    dfout.push_back(get_agent(i,event_stack));
  }
  return(dfout);
  }

//' Returns a matrix containing all events
//' @return a matrix containing all events
//' @export
// [[Rcpp::export]]
NumericMatrix Cget_all_events_matrix()
{
  NumericMatrix outm(event_stack_pointer,25);
  CharacterVector eventMatrixColNames(25);

// 'create' helper function is limited to 20 enteries

  eventMatrixColNames(0)  = "id";
  eventMatrixColNames(1)  = "local_time";
  eventMatrixColNames(2)  = "female";
  eventMatrixColNames(3)  = "time_at_creation";
  eventMatrixColNames(4)  = "age_at_creation";
  eventMatrixColNames(5)  = "pack_years";
  eventMatrixColNames(7)  = "event";
  eventMatrixColNames(13) = "localtime_at_COPD";
  eventMatrixColNames(14) = "age_at_COPD";
  eventMatrixColNames(15) = "weight_at_COPD";
  eventMatrixColNames(16) = "height";
  eventMatrixColNames(17) = "followup_after_COPD";


  colnames(outm) = eventMatrixColNames;
  for(int i=0;i<event_stack_pointer;i++)
  {
    agent *ag=&event_stack[i];
    outm(i,0)=(*ag).id;
    outm(i,1)=(*ag).local_time;
    outm(i,2)=(*ag).sex;
    outm(i,3)=(*ag).time_at_creation;
    outm(i,4)=(*ag).age_at_creation;

    outm(i,7)=(*ag).event;

    outm(i,13)=(*ag).local_time_at_COPD;
    outm(i,14)=(*ag).age_baseline;
    outm(i,15)=(*ag).weight_baseline;
    outm(i,16)=(*ag).height;
    outm(i,17)=(*ag).followup_time;

  }

  return(outm);
}



//////////////////////////////////////////////////////////////////EVENT_COPD////////////////////////////////////;
double event_COPD_tte(agent *ag)
{

  double rate=exp(input.COPD.ln_h_COPD_betas_by_sex[0][(*ag).sex]
                    +input.COPD.ln_h_COPD_betas_by_sex[1][(*ag).sex]*((*ag).age_at_creation+(*ag).local_time)
                    +input.COPD.ln_h_COPD_betas_by_sex[2][(*ag).sex]*pow((*ag).age_at_creation+(*ag).local_time,2)
                    +input.COPD.ln_h_COPD_betas_by_sex[3][(*ag).sex]*(*ag).pack_years
                    +input.COPD.ln_h_COPD_betas_by_sex[4][(*ag).sex]*(*ag).smoking_status
                    +input.COPD.ln_h_COPD_betas_by_sex[5][(*ag).sex]*(calendar_time+(*ag).local_time)
  );


  double tte;
  if(rate==0) tte=HUGE_VAL; else tte=rand_exp()/rate;
  //return(HUGE_VAL);
  return(tte);
}



void event_COPD_process(agent *ag)
{
  (*ag).weight_baseline = (*ag).weight;
  (*ag).age_baseline = (*ag).local_time+(*ag).age_at_creation;
  (*ag).followup_time = 0 ;
  (*ag).local_time_at_COPD = (*ag).local_time;


#if OUTPUT_EX>1
           output_ex.cumul_non_COPD_time+=(*ag).local_time;
#endif
#if (OUTPUT_EX & OUTPUT_EX_COPD) > 0
           output_ex.n_inc_COPD_by_ctime_age[(int)floor((*ag).time_at_creation+(*ag).local_time)][(int)(floor((*ag).age_at_creation+(*ag).local_time))]+=1;
#endif
}


////////////////////////////////////////////////////////////////////EVENT_bgd/////////////////////////////////////;
double event_bgd_tte(agent *ag)
{
  double age=(*ag).local_time+(*ag).age_at_creation;
  double time=(*ag).time_at_creation+(*ag).local_time;
  int age_cut=floor(age);

  double _or=exp(
    input.agent.ln_h_bgd_betas[0]
  +input.agent.ln_h_bgd_betas[1]*time
    +input.agent.ln_h_bgd_betas[2]*time*time
    +input.agent.ln_h_bgd_betas[3]*age);

    double ttd=HUGE_VAL;
    double p=input.agent.p_bgd_by_sex[age_cut][(int)(*ag).sex];
    if(p==0) return(ttd);

    double odds=p/(1-p)*_or;
    p=odds/(1+odds);

    if(p==1)
    {
      ttd=0;
    }
    else
    {
      double rate=-log(1-p);
      if(rate>0)  ttd=rand_exp()/rate; else ttd=HUGE_VAL;
      //if(rand_unif()<p) ttd=rand_unif();
    }
    return(ttd);
}

void event_bgd_process(agent *ag)
{
  (*ag).alive=false;
}


/////////////////////////////////////////////////////////////////////EVENT_FIXED/////////////////////////////////;
#define EVENT_FIXED_FREQ 1

double event_fixed_tte(agent *ag)
{
  return(floor((*ag).local_time*EVENT_FIXED_FREQ)/EVENT_FIXED_FREQ+1/EVENT_FIXED_FREQ-(*ag).local_time);
}



agent *event_fixed_process(agent *ag)
{
  (*ag).weight+=input.agent.weight_0_betas[6];
  (*ag).weight_LPT=(*ag).local_time;

  payoffs_LPT(ag);

#ifdef OUTPUT_EX
  update_output_ex(ag);
#endif

  //resetting annual cost and qaly
  (*ag).annual_cost=0;
  (*ag).annual_qaly=0;

  //COPD;
  double COPD_odds=exp(input.COPD.logit_p_COPD_betas_by_sex[0][(*ag).sex]
                         +input.COPD.logit_p_COPD_betas_by_sex[1][(*ag).sex]*(*ag).age_at_creation
                         +input.COPD.logit_p_COPD_betas_by_sex[2][(*ag).sex]*(*ag).age_at_creation*(*ag).age_at_creation
                         +input.COPD.logit_p_COPD_betas_by_sex[3][(*ag).sex]*(*ag).pack_years
                         +input.COPD.logit_p_COPD_betas_by_sex[4][(*ag).sex]*(*ag).smoking_status
                         +input.COPD.logit_p_COPD_betas_by_sex[5][(*ag).sex]*calendar_time)
                         //+input.COPD.logit_p_COPD_betas_by_sex[7]*(*ag).asthma
                         ;

  (*ag).p_COPD=COPD_odds/(1+COPD_odds);

  return(ag);
}








/////////////////////////////////////////////////////////event_birthday/////////////////////////////////////////////////

double event_birthday_tte(agent *ag)
{
  return(HUGE_VAL);
  double age=(*ag).age_at_creation+(*ag).local_time;
  double delta=1-(age-floor(age));
  if(delta==0) delta=1;
  //Rprintf("%f,%f\n",delta,(*ag).local_time+(*ag).age_at_creation);
  return(delta);
}


agent *event_birthday_process(agent *ag)
{
  //Rprintf("%f\n",(*ag).local_time+(*ag).age_at_creation);
  return(ag);
}






/////////////////////////////////////////////////////////////////////////MODEL///////////////////////////////////////////
// [[Rcpp::export]]
int Callocate_resources()
{
  if(runif_buffer==NULL)
    runif_buffer=(double *)malloc(settings.runif_buffer_size*sizeof(double));
  else
    realloc(runif_buffer,settings.runif_buffer_size*sizeof(double));
  if(runif_buffer==NULL) return(ERR_MEMORY_ALLOCATION_FAILED);
  runif_buffer_pointer=settings.runif_buffer_size; //invoikes fill next time;

  if(rnorm_buffer==NULL)
    rnorm_buffer=(double *)malloc(settings.rnorm_buffer_size*sizeof(double));
  else
    realloc(rnorm_buffer,settings.rnorm_buffer_size*sizeof(double));
  if(rnorm_buffer==NULL) return(ERR_MEMORY_ALLOCATION_FAILED);
  rnorm_buffer_pointer=settings.rnorm_buffer_size;

  if(rexp_buffer==NULL)
    rexp_buffer=(double *)malloc(settings.rexp_buffer_size*sizeof(double));
  else
    realloc(rexp_buffer,settings.rexp_buffer_size*sizeof(double));
  if(rexp_buffer==NULL) return(ERR_MEMORY_ALLOCATION_FAILED);
  rexp_buffer_pointer=settings.rexp_buffer_size;

  if(agent_stack==NULL)
    agent_stack=(agent *)malloc(settings.agent_stack_size*sizeof(agent));
  else
    realloc(agent_stack,settings.agent_stack_size*sizeof(agent));
  if(agent_stack==NULL) return(ERR_MEMORY_ALLOCATION_FAILED);
  agent_stack_pointer=0;

  if(event_stack==NULL)
    event_stack=(agent *)malloc(settings.event_stack_size*sizeof(agent));
  else
    realloc(event_stack,settings.event_stack_size*sizeof(agent));
  if(event_stack==NULL) return(ERR_MEMORY_ALLOCATION_FAILED);

  return(0);
}


// [[Rcpp::export]]
List Cget_pointers()
{
  return(Rcpp::List::create(
      //Rcpp::Named("runif_buffer_address")=reinterpret_cast<long &>(runif_buffer),
      //Rcpp::Named("rnorm_buffer_address")=reinterpret_cast<long &>(rnorm_buffer),
      //Rcpp::Named("rexp_buffer_address")=reinterpret_cast<long &>(rexp_buffer),
      //Rcpp::Named("agent_stack")=reinterpret_cast<long &>(agent_stack),
      //Rcpp::Named("event_stack")=reinterpret_cast<long &>(event_stack)
  )
  );
}



// [[Rcpp::export]]
int Cdeallocate_resources()
{
  try
  {
    if(runif_buffer!=NULL) {free(runif_buffer); runif_buffer=NULL;}
    if(rnorm_buffer!=NULL) {free(rnorm_buffer); rnorm_buffer=NULL;}
    if(rexp_buffer!=NULL) {free(rexp_buffer); rexp_buffer=NULL;}
    if(agent_stack!=NULL) {free(agent_stack); agent_stack=NULL;}
    if(event_stack!=NULL) {free(event_stack); event_stack=NULL;}
  }catch(const std::exception& e){};
  return(0);
}




// [[Rcpp::export]]
int Cdeallocate_resources2()
{
  try
  {
    delete[] runif_buffer;
    delete[] rnorm_buffer;
    delete[] rexp_buffer;
    delete[] agent_stack;
    delete[] event_stack;
  }catch(const std::exception& e){};
  return(0);
}





int Callocate_resources2()
{
  //runif_buffer=(double *)malloc(runif_buffer_size*sizeof(double));
  runif_buffer=new double[settings.runif_buffer_size];
  if(runif_buffer==NULL) return(ERR_MEMORY_ALLOCATION_FAILED);
  runif_buffer_pointer=settings.runif_buffer_size; //invoikes fill next time;

  rnorm_buffer=new double[settings.rnorm_buffer_size];
  if(rnorm_buffer==NULL) return(ERR_MEMORY_ALLOCATION_FAILED);
  rnorm_buffer_pointer=settings.rnorm_buffer_size;

  rexp_buffer=new double[settings.rexp_buffer_size];
  if(rexp_buffer==NULL) return(ERR_MEMORY_ALLOCATION_FAILED);
  rexp_buffer_pointer=settings.rexp_buffer_size;

  agent_stack=new agent[settings.agent_stack_size];
  if(agent_stack==NULL) return(ERR_MEMORY_ALLOCATION_FAILED);
  agent_stack_pointer=0;

  event_stack=new agent[settings.event_stack_size];
  if(event_stack==NULL) return(ERR_MEMORY_ALLOCATION_FAILED);

  return(0);
}

// [[Rcpp::export]]
int Cinit_session() //Does not deal with memory allocation only resets counters etc;
  {
  event_stack_pointer=0;

  reset_output();
  reset_output_ex();
  reset_runtime_stats(); runtime_stats.agent_size=sizeof(agent);

  calendar_time=0;
  last_id=0;

  return(0);
  }



// [[Rcpp::export]]
int Cmodel(int max_n_agents)
{
  if(max_n_agents<1) return(0);

  agent *ag;

  while(calendar_time<input.global_parameters.time_horizon && max_n_agents>0)
    //for(int i=0;i<n_agents;i++)
  {
    max_n_agents--;
    //calendar_time=0; NO! calendar_time is set to zero at init_session. Cmodel should be resumable;

    switch(settings.agent_creation_mode)
    {
    case agent_creation_mode_one:
      ag=create_agent(&smith,last_id);
      break;

    case agent_creation_mode_all:
      ag=create_agent(&agent_stack[last_id],last_id);
      break;

    case agent_creation_mode_pre:
      ag=&agent_stack[last_id];
      break;

    default:
      return(-1);
    }

    (*ag).tte=0;
    event_start_process(ag);
    (*ag).event=event_start;
    if(settings.record_mode==record_mode_event || settings.record_mode==record_mode_agent || settings.record_mode==record_mode_some_event)
    {
      int _res=push_event(ag);
      if(_res<0) return(_res);
    }


    while(calendar_time+(*ag).local_time<input.global_parameters.time_horizon && (*ag).alive && (*ag).age_at_creation+(*ag).local_time<MAX_AGE)
    {
      double tte=input.global_parameters.time_horizon-calendar_time-(*ag).local_time;;
      int winner=-1;
      double temp;

      temp=event_fixed_tte(ag);
      if(temp<tte)
      {
        tte=temp;
        winner=event_fixed;
      }

      temp=event_birthday_tte(ag);
      if(temp<tte)
      {
        tte=temp;
        winner=event_birthday;
      }


      temp=event_COPD_tte(ag);
      if(temp<tte)
      {
        tte=temp;
        winner=event_COPD;
      }

    
      temp=event_bgd_tte(ag);
      if(temp<tte)
      {
        tte=temp;
        winner=event_bgd;
      }


      if(calendar_time+(*ag).local_time+tte<input.global_parameters.time_horizon)
      {
        (*ag).tte=tte;
        (*ag).local_time=(*ag).local_time+tte;

        if(settings.update_continuous_outcomes_mode==1)
        {
          payoffs_LPT(ag);
        }

        switch(winner)
        {
        case event_fixed:
          event_fixed_process(ag);
          (*ag).event=event_fixed;
          break;
        case event_birthday:
          event_birthday_process(ag);
          (*ag).event=event_birthday;
          break;
        case event_COPD:
          event_COPD_process(ag);
          (*ag).event=event_COPD;
          break;

         
        case event_bgd:
          event_bgd_process(ag);
          (*ag).event=event_bgd;
          break;
        }
        if(settings.record_mode==record_mode_event || settings.record_mode==record_mode_some_event)
        {
          int _res=push_event(ag);
          if(_res<0) return(_res);
        }
      }
      else
      {//past TH, set the local time to TH as the next step will be agent end;
        (*ag).tte=input.global_parameters.time_horizon-calendar_time-(*ag).local_time;
        (*ag).local_time=(*ag).local_time+(*ag).tte;
      }
    }//while (within agent)

    event_end_process(ag);
    (*ag).event=event_end;
    if(settings.record_mode==record_mode_event || settings.record_mode==record_mode_agent || settings.record_mode==record_mode_some_event)
    {
      int _res=push_event(ag);
      if(_res<0) return(_res);
    }

    if (output.n_agents>settings.n_base_agents)  //now we are done with prevalent cases and are creating incident cases;
      {
      double incidence = exp(input.agent.l_inc_betas[0]+input.agent.l_inc_betas[1]*calendar_time+input.agent.l_inc_betas[2]*calendar_time*calendar_time);
      if(incidence<0.000000000000001) calendar_time=input.global_parameters.time_horizon; else {
        if (calendar_time!=0) calendar_time=calendar_time+1/(incidence*settings.n_base_agents); else calendar_time=calendar_time+1; //suprresing incidence cases in the first year
      }
      }
    last_id++;
  }//Outer while (between-agent)
  return(0);
  }











