# The following declarations are already defined in mode.WIP.cpp
# they are replicated here to make it compatible with the package. Amin

record_mode<-c(
  record_mode_none=0,
  record_mode_agent=1,
  record_mode_event=2,
  record_mode_some_event=3
)

events<-c(
  event_start=0,
  event_fixed=1,
  event_birthday=2,

  event_COPD=4,

  event_bgd=13,
  event_end=14
)


agent_creation_mode<-c(
  agent_creation_mode_one=0,
  agent_creation_mode_all=1,
  agent_creation_mode_pre=2
)


errors<-c(
  ERR_INCORRECT_SETTING_VARIABLE=-1,
  ERR_INCORRECT_VECTOR_SIZE=-2,
  ERR_INCORRECT_INPUT_VAR=-3,
  ERR_EVENT_STACK_FULL=-4,
  ERR_MEMORY_ALLOCATION_FAILED=-5
)

# End of declarations

#' Returns a list of default model input values
#' @param None
#' @export
init_input <- function() {
  input <- list()
  input_help <- list()
  input_ref <- list()


  input$global_parameters <- list(age0 = 40, time_horizon = 20, discount_cost = 0.03, discount_qaly = 0.03)
  input_help$global_parameters <- list(age0 = "Starting age in the model", time_horizon = "Model time horizon", discount_cost = "Discount value for cost outcomes",
                                       discount_qaly = "Discount value for QALY outcomes")


  input_help$agent$p_female <- "Proportion of females in the population"
  input$agent$p_female <- 0.5
  input_ref$agent$p_female <- "Model assumption"


  input_help$agent$height_0_betas <- "Regressoin coefficients for estimating height (in meters) at baseline"
  input$agent$height_0_betas <- t(as.matrix(c(intercept = 1.82657, sex = -0.13093, age = -0.00125, age2 = 2.31e-06, sex_age = -0.0001651)))
  input_ref$agent$height_0_betas <- ""


  input_help$agent$height_0_sd <- "SD representing heterogeneity in baseline height"
  input$agent$height_0_sd <- 0.06738
  input_ref$agent$height_0_sd <- ""


  input_help$agent$weight_0_betas <- "Regressoin coefficients for estimating weiight (in Kg) at baseline"
  input$agent$weight_0_betas <- t(as.matrix(c(intercept = 50, sex = -5, age = 0.1, age2 = 0, sex_age = 0, height = 1, year = 0.01)))
  input_ref$agent$weight_0_betas <- ""


  input_help$agent$weight_0_sd <- "SD representing heterogeneity in baseline weight"
  input$agent$weight_0_sd <- 5
  input_ref$agent$weight_0_sd <- ""


  input_help$agent$height_weight_rho <- "Correlaiton coefficient between weight and height at baseline"
  input$agent$height_weight_rho <- 0
  input_ref$agent$height_weight_rho <- ""

  input$agent$p_prevalence_age <- c(rep(0, 40), c(473.9, 462.7, 463, 469.3, 489.9, 486.3, 482.7, 479, 483.7, 509, 542.8, 557.7,
                                                  561.4, 549.7, 555.3, 544.6, 531.6, 523.9, 510.2, 494.1, 486.5, 465.6, 442.8, 424.6, 414.9, 404.3, 394.4, 391.5, 387.3, 330.9,
                                                  304.5, 292.2, 277.4, 255.1, 241.1, 223.2, 211.4, 198.8, 185.6, 178, 166.7, 155.2, 149.1, 140.6, 131.9, 119.7, 105.8, 95.4,
                                                  83.4, 73.2, 62.4, 52.3, 42.7, 34.7, 27, 19.9, 13.2, 8.8, 5.9, 3.8, 6.8), rep(0, 10))

  input_help$agent$p_prevalence_age <- "Age pyramid at baseline (taken from CanSim.052-0005.xlsm for year 2015)"
  input$agent$p_prevalence_age <- input$agent$p_prevalence_age/sum(input$agent$p_prevalence_age)
  input_ref$agent$p_prevalence_age <- "CanSim.052-0005.xlsm for year 2015. THANKS TO AMIN!"


  input_help$agent$p_incidence_age <- "Discrete distribution of age for the incidence population (population arriving after the first date) - generally estimated through calibration"
  input$agent$p_incidence_age <- c(rep(0, 40), c(1), 0.02* exp(-(0:59)/5), rep(0, 111 - 40 - 1 - 60))
  input$agent$p_incidence_age <- input$agent$p_incidence_age/sum(input$agent$p_incidence_age)
  input_ref$agent$p_incidence_age <- ""


  input_help$agent$p_bgd_by_sex <- "Life table"
  input$agent$p_bgd_by_sex <- cbind(male = c(0.00522, 3e-04, 0.00022, 0.00017, 0.00013, 0.00011, 1e-04, 9e-05, 8e-05, 8e-05, 9e-05,
                                             1e-04, 0.00012, 0.00015, 2e-04, 0.00028, 0.00039, 0.00051, 0.00059, 0.00066, 0.00071, 0.00075, 0.00076, 0.00076, 0.00074,
                                             0.00071, 7e-04, 0.00069, 7e-04, 0.00071, 0.00074, 0.00078, 0.00082, 0.00086, 0.00091, 0.00096, 0.00102, 0.00108, 0.00115,
                                             0.00123, 0.00132, 0.00142, 0.00153, 0.00165, 0.00179, 0.00194, 0.00211, 0.00229, 0.00251, 0.00275, 0.00301, 0.00331, 0.00364,
                                             0.00401, 0.00441, 0.00484, 0.00533, 0.00586, 0.00645, 0.00709, 0.0078, 0.00859, 0.00945, 0.0104, 0.01145, 0.0126, 0.01387,
                                             0.01528, 0.01682, 0.01852, 0.0204, 0.02247, 0.02475, 0.02726, 0.03004, 0.0331, 0.03647, 0.04019, 0.0443, 0.04883, 0.05383,
                                             0.05935, 0.06543, 0.07215, 0.07957, 0.08776, 0.0968, 0.10678, 0.1178, 0.12997, 0.14341, 0.15794, 0.17326, 0.18931, 0.20604,
                                             0.21839, 0.23536, 0.2529, 0.27092, 0.28933, 0.30802, 0.32687, 0.34576, 0.36457, 0.38319, 0.40149, 0.41937, 0.43673, 0.4535,
                                             0.4696, 1), female = c(0.00449, 0.00021, 0.00016, 0.00013, 1e-04, 9e-05, 8e-05, 7e-05, 7e-05, 7e-05, 8e-05, 8e-05, 9e-05,
                                                                    0.00011, 0.00014, 0.00018, 0.00022, 0.00026, 0.00028, 0.00029, 3e-04, 3e-04, 0.00031, 0.00031, 3e-04, 3e-04, 3e-04, 0.00031,
                                                                    0.00032, 0.00034, 0.00037, 4e-04, 0.00043, 0.00047, 0.00051, 0.00056, 6e-04, 0.00066, 0.00071, 0.00077, 0.00084, 0.00092,
                                                                    0.001, 0.00109, 0.00118, 0.00129, 0.0014, 0.00153, 0.00166, 0.00181, 0.00197, 0.00215, 0.00235, 0.00257, 0.0028, 0.00307,
                                                                    0.00336, 0.00368, 0.00403, 0.00442, 0.00485, 0.00533, 0.00586, 0.00645, 0.0071, 0.00782, 0.00862, 0.00951, 0.01051, 0.01161,
                                                                    0.01284, 0.0142, 0.01573, 0.01743, 0.01934, 0.02146, 0.02384, 0.02649, 0.02947, 0.0328, 0.03654, 0.04074, 0.04545, 0.05074,
                                                                    0.05669, 0.06338, 0.07091, 0.0794, 0.08897, 0.09977, 0.11196, 0.12542, 0.13991, 0.15541, 0.1719, 0.18849, 0.20653, 0.22549,
                                                                    0.24526, 0.26571, 0.28671, 0.3081, 0.3297, 0.35132, 0.3728, 0.39395, 0.41461, 0.43462, 0.45386, 0.47222, 1))
  input_ref$agent$p_bgd_by_sex <- "Life table"

  input_help$agent$l_inc_betas <- "Ln of incidence rate of the new population - Calibration target to keep populatoin size and age pyramid in line with calibration"
  input$agent$l_inc_betas <- t(as.matrix(c(intercept = -3.55, y = 0.01, y2 = 0))) # intercept is the result of model calibration,
  input_ref$agent$l_inc_betas <- ""


  input_help$agent$ln_h_bgd_betas <- "Increased Longevity Over time and effect of other variables"
  input$agent$ln_h_bgd_betas <- t(as.matrix(c(intercept = 0, y = -0.025, y2 = 0, age = 0, b_mi = 0, n_mi = 0, b_stroke = 0,
                                              n_stroke = 0, hf = 0)))  #AKA longevity
  input_ref$agent$ln_h_bgd_betas <- ""



  ## COPD
  input_help$COPD$logit_p_COPD_betas_by_sex <- "Logit of the probability of having COPD (FEV1/FVC<0.7) at time of creation (separately by sex)"
  input$COPD$logit_p_COPD_betas_by_sex <- cbind(male = c(intercept = -4.522189  , age = 0.033070   , age2 = 0, pack_years = 0.025049   ,
                                                         current_smoking = 0, year = 0, asthma = 0),
                                                female = c(intercept = -4.074861   , age = 0.027359   , age2 = 0, pack_years = 0.030399   ,
                                                           current_smoking = 0, year = 0, asthma = 0))
  input_ref$COPD$logit_p_COPD_betas_by_sex <- "CanCold - Shahzad's Derivation. Last Updated on 2017-09-19, ne wmodel with no currnet smoker term"


  input_help$COPD$ln_h_COPD_betas_by_sex <- "Log-hazard of developing COPD (FEV1/FVC<LLN) for those who did not have COPD at creation time (separately by sex)"
  input$COPD$ln_h_COPD_betas_by_sex <- cbind(male = c(Intercept = -7.86555434, age = 0.03261784, age2 = 0, pack_years =  0.03076498,
                                                      smoking_status = 0, year = 0, asthma = 0),
                                             female = c(Intercept = -7.75884176, age = 0.02793072, age2 = 0, pack_years = 0.04184035,
                                                        smoking_status =  0, year = 0, asthma = 0))
  input_ref$COPD$ln_h_COPD_betas_by_sex <- "Amin's Iterative solution. Last Updated on 2018-02-10 (0.18.0)"


  ##cost and utility
  input$cost$bg_cost_by_stage=t(as.matrix(c(N=0, I=615, II=1831, III=2619, IV=3021)))
  input_help$cost$bg_cost_by_stage="Annual direct costs for non-COPD, and COPD by GOLD grades"
  #  input$cost$ind_bg_cost_by_stage=t(as.matrix(c(N=0, I=40, II=80, III=134, IV=134))) #TODO Not implemented in C yet.
  #  input_help$cost$ind_bg_cost_by_stage="Annual inddirect costs for non-COPD, and COPD by GOLD grades"
  input$cost$exac_dcost=t(as.matrix(c(mild=29,moderate=726,severe=9212, verysevere=20170)))
  input_help$cost$exac_dcost="Incremental direct costs of exacerbations by severity levels"

  #input$cost$doctor_visit_by_type<-t(as.matrix(c(50,150)))

  input$utility$bg_util_by_stage=t(as.matrix(c(N=0.85, I=0.81,II=0.72,III=0.68,IV=0.58)))
  input_help$utility$bg_util_by_stage="Background utilities for non-COPD, and COPD by GOLD grades"
  #  input$utility$exac_dutil=t(as.matrix(c(mild=-0.07, moderate=-0.37/2, severe=-0.3)))
  input$utility$exac_dutil=cbind(
    gold1=c(mild=-0.0225, moderate=-0.0225, severe=-0.0728, verysevere=-0.0728),
    gold2=c(mild=-0.0155, moderate=-0.0155, severe=-0.0683, verysevere=-0.0683),
    gold3=c(mild=-0.0488, moderate=-0.0488, severe=-0.0655, verysevere=-0.0655),
    gold4=c(mild=-0.0488, moderate=-0.0488, severe=-0.0655, verysevere=-0.0655)
  );
  input_help$utility$exac_dutil="Incremental change in utility during exacerbations by severity level"

  input$manual$MORT_COEFF<-1
  input$manual$smoking$intercept_k<-1


  #Proportion of death by COPD that should be REMOVED from background mortality
  input$manual$explicit_mortality_by_age_sex<-cbind(
    #   0 for <35         35-39           40-44           45-49          50-54      55-59           60-64        65-69          70-74       75-79       80+
    male =c(0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0.000249949663705019,0.000279361481107238,0.000143184143539575,4.58552059432562e-05,0.000102594155577809,6.52263020912318e-05,1.07778470262612e-05,5.81042290186133e-05,2.18336719635301e-05,4.49710556825597e-05,-4.0267754970943e-05,-4.99669760287346e-05,-8.59620496745581e-05,-3.95551572650197e-05,-0.000165025695836827,-0.000238037931304093,-0.00032685432729955,-0.000438715069064686,-0.00050600469223705,0.000580239786723037,0.00165281035519525,0.00178882748260027,0.00197360741088778,0.00208265734639317,0.0023769834442027,0.00260422648107871,0.00292357104958589,0.00313545134525439,0.00334684816453369,0.00287752763870146,0.00240973265966247,0.00265902746355965,0.00303749042857185,0.00316156673273281,0.00360416997146693,0.0040639317255912,0.00447304738915796,0.00499378954488701,0.00557874849764433,0.00505770115735889,0.00407777319105142,0.00456095766267728,0.0055432401185574,0.00714354429349771,0.00859356082756144,0.00962513757175244,0.0112473627269577,0.013117771620995,0.0165918069682699,0.0198035368680714,0.022929228931603,0.0275506298462985,0.0305483419760513,0.0374875651996102,0.0394902254213564,0.0468895514727401,0.051803798156478,0.0587768050374814,0.0651606935902558,0.0723572320833362,0.0838078268330134,0.087813075182472,0.0804257382809533,0.0899887008867733,0.0746611452642845,0.0773748834918532,0.149696405844052,0.167229923246881,0.125657988543488, 0, 0

    )
    ,female =c(0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,-0.00077,0.000204044403520189,0.00012527721873265,9.38179764267892e-05,8.52310693595392e-05,2.0948736613633e-05,0.000112963841564107,0.00010433676504019,5.4970941674673e-05,8.40452455971427e-05,7.13507996050295e-05,-5.32163960539644e-05,0.000108600166102096,-2.58547086080699e-05,-3.39656984665966e-05,-0.000104561408155271,-7.61106203700976e-05,-0.000145376981544917,-8.95689640121699e-05,-0.000166434371840576,0.000243219925937975,0.0008650546976534,0.000930888336636944,0.00101216109953299,0.00123284222432014,0.00133853130591277,0.00144269885860267,0.00168219643935857,0.00180689077125785,0.00189747191320053,0.00189498835689917,0.00144977074941933,0.00152408964706417,0.00171179411065646,0.00188158359328075,0.00200917449209433,0.00255429028811746,0.00251462637075366,0.00297904911157468,0.00305247152311787,0.00305909639703686,0.00247270876123033,0.00263944132927282,0.00297318580456147,0.00399439553966772,0.00430369126521429,0.00519320421612733,0.00694185653392955,0.00759956858672857,0.00934224983773695,0.0112825987417631,0.0127892562246108,0.0174914683720296,0.0188209248942959,0.0219164492207358,0.0264718581662204,0.0301728155832754,0.0330406840844788,0.0374281244493398,0.0432518126953188,0.0531098375652765,0.0623997534486314,0.0591768901970186,0.0737223620452372,0.0753080451656278,0.104676330930181,0.108606384925549,0.156783719504522,0.114000430396816,0.191472188310203,0,0

    ))


  model_input <- list ("values" = input, "help" = input_help, "ref" = input_ref)
  return (model_input)
}

model_input <- init_input()
