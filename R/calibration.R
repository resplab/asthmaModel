calib_params <- list(COPD = list(prevalence = 0.193))

#' Calibrates explicit mortality by Amin
#' @param n_sim number of agents
#' @return difference in mortality rates and life table
#' @export
calibrate_explicit_mortality2 <- function(n_sim = 10^7) {
  cat("Difference between life table and observed mortality\n")
  cat("You need to put the returned values into model_input$manual$explicit_mortality_by_age_sex\n")
  cat(paste("n_sim=", n_sim, "\n"))

  settings <- default_settings
  settings$record_mode <- record_mode["record_mode_none"]
  settings$agent_stack_size <- 0
  settings$n_base_agents <- n_sim
  settings$event_stack_size <- 0
  init_session(settings = settings)

  input <- model_input$values
  input$agent$l_inc_betas[1,] <- (-1000)  #No incidence (Life table is only valid for baseline)
  input$global_parameters$time_horizon <- 20
  input$manual$explicit_mortality_by_age_sex <- input$manual$explicit_mortality_by_age_sex * 0

  cat("working...\n")
  res <- run(input = input)

  cat("Mortality rate was", Cget_output()$n_death/Cget_output()$cumul_time, "\n")

  difference <- (Cget_output_ex()$n_death_by_age_sex/Cget_output_ex()$sum_time_by_age_sex) - input$agent$p_bgd_by_sex

  difference <- as.data.frame(t(difference))
  print(difference)
  write.csv(difference[1,], "male_mortality.csv")
  write.csv(difference[2,], "female_mortality.csv")


  terminate_session()
  return(difference)
}


#' Calibrates explicit mortality
#' @param n_sim number of agents
#' @return difference in mortality rates and life table
#' @export
calibrate_explicit_mortality <- function(n_sim = 10^8) {
  cat("Difference between life table and observed mortality\n")
  cat("You need to put the returned values into model_input$manual$explicit_mortality_by_age_sex\n")
  cat(paste("n_sim=", n_sim, "\n"))

  settings <- default_settings
  settings$record_mode <- record_mode["record_mode_none"]
  settings$agent_stack_size <- 0
  settings$n_base_agents <- n_sim
  settings$event_stack_size <- 0
  init_session(settings = settings)

  input <- model_input
  input$agent$l_inc_betas[1,] <--100  #No incidence (Life table is only valid for baseline)
  input$global_parameters$time_horizon <- 1
  input$manual$explicit_mortality_by_age_sex <- input$manual$explicit_mortality_by_age_sex * 0

  cat("working...\n")
  res <- run(input = input)

  cat("Mortality rate was", Cget_output()$n_death/Cget_output()$cumul_time, "\n")

  difference <- model_input$agent$p_bgd_by_sex[41:111, ] - (Cget_output_ex()$n_death_by_age_sex[41:111, ]/Cget_output_ex()$sum_time_by_age_sex[41:111,
                                                                                                                                               ])
  plot(40:110, difference[, 1], type = "l", col = "blue", xlab = "age", ylab = "Difference")
  legend("topright", c("male", "female"), lty = c(1, 1), col = c("blue", "red"))
  lines(40:110, difference[, 2], type = "l", col = "red")
  title(cex.main = 0.5, "Difference between expected (life table) to simulated mortality, by sex and age")

  difference[which(is.nan(difference))] <- 0
  difference[which(abs(difference) == Inf)] <- 0
  difference <- rbind(matrix(rep(0, 80), ncol = 2), difference)

  terminate_session()
  return(difference)
}


#' Solves stochastically for COPD incidence rate equation.
#' @param nIterations number of iterations for the numberical solution
#' @param nPatients number of simulated agents.
#' @param time_horizon in years
#' @return regression co-efficients as files
#' @export

calibrate_COPD_inc<-function(nIterations=500,
                           nPatients=100000,
                           time_horizon=20)
{

  latest_COPD_inc_logit <- cbind(
    male =c(Intercept = 0, age = 0 ,age2 = 0, pack_years = 0, smoking_status = 0, year = 0, asthma = 0)
    ,female =c(Intercept= 0, age = 0, age2 =0, pack_years = 0, smoking_status = 0, year = 0, asthma = 0))


  cat("iteration", "intercept_men", "age_coeff_men", "packyears_coeff_men", "intercept_women", "age_coeff_women", "packyears_coeff_women", file="iteration_coeff.csv",sep=",",append=FALSE, fill=FALSE)
  cat("\n",file="iteration_coeff.csv",sep=",",append=TRUE)

  cat("iteration","resid_intercept_men", "resid_age_coeff_men", "resid_packyears_coeff_men", "resid_intercept_women", "resid_packyears_coeff_women" ,"resid_age_coeff_women" , file="iteration_resid.csv",sep=",",append=FALSE, fill=FALSE)
  cat("\n",file="iteration_resid.csv",sep=",",append=TRUE)

  for (i in 1:nIterations){

    settings<-default_settings
    settings$record_mode<-record_mode["record_mode_some_event"]
    settings$events_to_record = c(1)
    settings$agent_stack_size<-0
    settings$n_base_agents<- nPatients
    settings$event_stack_size <- 1e+06 * 1.7 * 20
    init_session(settings=settings)
    input<-model_input$values

    #  input$smoking$mortality_factor_current <- 1 #checking to see if these two values are throwing off the regression
    #  input$smoking$mortality_factor_former <- 1

    # if (i == 1) {
    #    latest_COPD_inc_logit <- input$COPD$ln_h_COPD_betas_by_sex #starting with the current regression
    #  }

    input$COPD$ln_h_COPD_betas_by_sex <- latest_COPD_inc_logit
    input$agent$ln_h_bgd_betas <- t(as.matrix(c(intercept = 0, y = 0, y2 = 0, age = 0, b_mi = 0, n_mi = 0, b_stroke = 0,
                                                n_stroke = 0, hf = 0)))  #Disabling longevity
    run(input=input)
    data<-as.data.frame(Cget_all_events_matrix())
    terminate_session()

    dataF<-data[which(data[,'event']==events["event_fixed"]),]
    dataF[,'age']<-dataF[,'local_time']+dataF[,'age_at_creation']
    dataF[,'copd']<-(dataF[,'gold']>0)*1
    dataF[,'gold2p']<-(dataF[,'gold']>1)*1
    dataF[,'gold3p']<-(dataF[,'gold']>2)*1
    dataF[,'year']<-dataF[,'local_time']+dataF[,'time_at_creation']

    if (exists('rxGlm')) {
      res_male<-rxGlm(data=dataF[which(dataF[,'sex']==0),],formula=copd~age+pack_years,family=binomial(link=logit))
      res_female<-rxGlm(data=dataF[which(dataF[,'sex']==1),],formula=copd~age+pack_years,family=binomial(link=logit))
    }
    else {
      res_male<-glm(data=dataF[which(dataF[,'sex']==0),],formula=copd~age+pack_years,family=binomial(link=logit))
      res_female<-glm(data=dataF[which(dataF[,'sex']==1),],formula=copd~age+pack_years,family=binomial(link=logit))
    }

    coefficients(res_male)
    coefficients(res_female)

    latest_COPD_prev_logit <- cbind(
      male =c(Intercept =summary(res_male)$coefficients[1,1],age = summary(res_male)$coefficients[2,1] ,age2 = 0, pack_years = summary(res_male)$coefficients[3,1], smoking_status = 0, year = 0,asthma = 0)
      ,female =c(Intercept =summary(res_female)$coefficients[1,1] ,age = summary(res_female)$coefficients[2,1], age2 =0, pack_years = summary(res_female)$coefficients[3,1], smoking_status = 0 ,year = 0,asthma = 0))

    residual <-  input$COPD$logit_p_COPD_betas_by_sex-latest_COPD_prev_logit
    message(c(i, "th loop:"))
    message ("residual is:")
    print (residual)

    cat(i, residual[1,1], residual[2,1], residual[4,1], residual[1,2], residual[2,2], residual[4,2], file="iteration_resid.csv",sep=",",append=TRUE, fill=FALSE)
    cat("\n",file="iteration_resid.csv",sep=",",append=TRUE)


    latest_COPD_inc_logit <- latest_COPD_inc_logit + residual

    message ("latest inc logit is:")
    print (latest_COPD_inc_logit)

    cat(i, latest_COPD_inc_logit[1,1], latest_COPD_inc_logit[2,1], latest_COPD_inc_logit[4,1], latest_COPD_inc_logit[1,2], latest_COPD_inc_logit[2,2], latest_COPD_inc_logit[4,2], file="iteration_coeff.csv",sep=",",append=TRUE, fill=FALSE)
    cat("\n",file="iteration_coeff.csv",sep=",",append=TRUE)


  }

  iteration_coeff <- readr::read_csv("./iteration_coeff.csv")
  iteration_resid <- readr::read_csv("./iteration_resid.csv")


  plot(ggplot2::qplot(iteration, age_coeff_men, data=iteration_coeff, size=I(1), main = "Age Coefficient Convergence for Men"))
  plot(ggplot2::qplot(iteration, age_coeff_women, data=iteration_coeff, size=I(1), main = "Age Coefficient Convergence for Women"))

  plot(ggplot2::qplot(iteration, packyears_coeff_men, data=iteration_coeff, size=I(1), main = "Smoking (packyears) Coefficient Convergence for Men"))
  plot(ggplot2::qplot(iteration, packyears_coeff_women, data=iteration_coeff, size=I(1), main = "Smoking (packyears) Coefficient Convergence for women"))

  plot(ggplot2::qplot(iteration, intercept_men, data=iteration_coeff, size=I(1), main = "Logit intercept Convergence for Men"))
  plot(ggplot2::qplot(iteration, intercept_women, data=iteration_coeff, size=I(1), main = "Logit intercept Convergence for women"))


  plot(ggplot2::qplot(iteration, resid_age_coeff_men, data=iteration_resid, size=I(1), main = "Residue for Age Coefficient - Men"))
  plot(ggplot2::qplot(iteration, resid_age_coeff_women, data=iteration_resid, size=I(1), main = "Residue for Age Coefficient - Women"))

  plot( ggplot2::qplot(iteration, resid_packyears_coeff_men, data=iteration_resid, size=I(1),  main = "Residue for Cigarette Smoking (packyears) Coefficient - Men"))
  plot( ggplot2::qplot(iteration, resid_packyears_coeff_women, data=iteration_resid, size=I(1),  main = "Residue for Cigarette Smoking (packyears) Coefficient - Women"))

  plot( ggplot2::qplot(iteration, resid_intercept_men, data=iteration_resid, size=I(1),  main = "Residue for logit intercept - Men"))
  plot( ggplot2::qplot(iteration, resid_intercept_women, data=iteration_resid, size=I(1),  main = "Residue for logit intercept - Women"))

  if (exists('rxGlm')) {
    res_male<-rxGlm(data=dataF[which(dataF[,'sex']==0),],formula=copd~age+pack_years+year,family=binomial(link=logit))
    res_female<-rxGlm(data=dataF[which(dataF[,'sex']==1),],formula=copd~age+pack_years+year,family=binomial(link=logit))
  }
  else {
    res_male<-glm(data=dataF[which(dataF[,'sex']==0),],formula=copd~age+pack_years+year,family=binomial(link=logit))
    res_female<-glm(data=dataF[which(dataF[,'sex']==1),],formula=copd~age+pack_years+year,family=binomial(link=logit))
  }
  message(coefficients(res_male))
  message(coefficients(res_female))

}



