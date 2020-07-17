report_mode <- 1
# If 1, we are generating a report!

petoc <- function() {
  if (report_mode == 0) {
    message("Press [Enter] to continue")
    r <- readline()
    if (r == "q") {
      terminate_session()
      stop("User asked for termination.\n")
    }
  }
}


#' Basic tests of model functionalty. Serious issues if the test does not pass.
#' @return tests results
#' @export
sanity_check <- function() {
  init_session()

  cat("test 1: zero all costs\n")
  input <- model_input$values
  for (el in get_list_elements(input$cost)) input$cost[[el]] <- input$cost[[el]] * 0
  res <- run(1, input = input)
  if (Cget_output()$total_cost != 0)
    message("Test failed!") else message("Test passed!")


  cat("test 2: zero all utilities\n")
  input <- model_input$values
  for (el in get_list_elements(input$utility)) input$utility[[el]] <- input$utility[[el]] * 0
  res <- run(input = input)
  if (Cget_output()$total_qaly != 0)
    message("Test failed!") else message("Test passed!")


  cat("test 3: one all utilities ad get one QALY without discount\n")
  input <- model_input$values
  input$global_parameters$discount_qaly <- 0
  for (el in get_list_elements(input$utility)) input$utility[[el]] <- input$utility[[el]] * 0 + 1
  input$utility$exac_dutil = input$utility$exac_dutil * 0
  res <- run(input = input)
  if (Cget_output()$total_qaly/Cget_output()$cumul_time != 1)
    message("Test failed!") else message("Test passed!")


  cat("test 4: zero mortality (both bg and exac)\n")
  input <- model_input$values
  input$exacerbation$logit_p_death_by_sex <- input$exacerbation$logit_p_death_by_sex * 0 - 10000000  # log scale'
  input$agent$p_bgd_by_sex <- input$agent$p_bgd_by_sex * 0
  input$manual$explicit_mortality_by_age_sex <- input$manual$explicit_mortality_by_age_sex * 0
  res <- run(input = input)
  if (Cget_output()$n_deaths != 0) {
    cat (Cget_output()$n_deaths)
    stop("Test failed!")
  } else message("Test passed!")
  terminate_session()
}


#' Returns results of validation tests for population module
#' @param incidence_k a number (default=1) by which the incidence rate of population will be multiplied.
#' @param remove_COPD 0 or 1, indicating whether COPD-caused mortality should be removed
#' @param savePlots 0 or 1, exports 300 DPI population growth and pyramid plots comparing simulated vs. predicted population
#' @return validation test results
#' @export
validate_population <- function(remove_COPD = 0, incidence_k = 1, savePlots = 0) {
  cat("Validate_population(...) is responsible for producing output that can be used to test if the population module is properly calibrated.\n")
  petoc()

  settings <- default_settings
  settings$record_mode <- record_mode["record_mode_none"]
  settings$agent_stack_size <- 0
  settings$n_base_agents <- 1e+06
  settings$event_stack_size <- 1
  init_session(settings = settings)
  input <- model_input$values  #We can work with local copy more conveniently and submit it to the Run function

  cat("\nBecause you have called me with remove_COPD=", remove_COPD, ", I am", c("NOT", "indeed")[remove_COPD + 1], "going to remove COPD-related mortality from my calculations")
  petoc()

  cat(getwd())
  # CanSim.052.0005<-read.csv(system.file ('extdata', 'CanSim.052.0005.csv', package = 'epicR'), header = T); #package ready
  # reading
  x <- aggregate(CanSim.052.0005[, "value"], by = list(CanSim.052.0005[, "year"]), FUN = sum)
  x[, 2] <- x[, 2]/x[1, 2]
  x <- x[1:input$global_parameters$time_horizon, ]
  plot(x, type = "l", ylim = c(0.5, max(x[, 2] * 1.5)), xlab = "Year", ylab = "Relative population size")
  title(cex.main = 0.5, "Relative populaton size")
  cat("The plot I just drew is the expected (well, StatCan's predictions) relative population growth from 2015\n")
  petoc()

  if (remove_COPD) {
    input$exacerbation$logit_p_death_by_sex <- -1000 + input$exacerbation$logit_p_death_by_sex
    input$manual$explicit_mortality_by_age_sex <- 0
  }

  input$agent$l_inc_betas[1] <- input$agent$l_inc_betas[1] + log(incidence_k)

  cat("working...\n")
  res <- run(input = input)
  if (res < 0) {
    stop("Something went awry; bye!")
    return()
  }

  n_y1_agents <- sum(Cget_output_ex()$n_alive_by_ctime_sex[1, ])
  legend("topright", c("Predicted", "Simulated"), lty = c(1, 1), col = c("black", "red"))

  cat("And the black one is the observed (simulated) growth\n")
   ######## pretty population growth curve

  CanSim <- tibble::as.tibble(CanSim.052.0005)
  CanSim <- tidyr::spread(CanSim, key = year, value = value)
  CanSim <- CanSim[, 3:51]
  CanSim <- colSums (CanSim)

  df <- data.frame(Year = c(2015:(2015 + model_input$values$global_parameters$time_horizon-1)), Predicted = CanSim[1:model_input$values$global_parameters$time_horizon] * 1000, Simulated = rowSums(Cget_output_ex()$n_alive_by_ctime_sex)/ settings$n_base_agents * 18179400) #rescaling population. There are about 18.6 million Canadians above 40
  message ("Here's simulated vs. predicted population table:")
  print(df)
  dfm <- reshape2::melt(df[,c('Year','Predicted','Simulated')], id.vars = 1)
  plot_population_growth  <- ggplot2::ggplot(dfm, aes(x = Year, y = value)) +  theme_tufte(base_size=14, ticks=F) +
    geom_bar(aes(fill = variable), stat = "identity", position = "dodge") +
    labs(title = "Population Growth Curve") + ylab ("Population") +
    labs(caption = "(based on population at age 40 and above)") +
    theme(legend.title=element_blank()) +
    scale_y_continuous(name="Population", labels = scales::comma)

  plot (plot_population_growth)
  if (savePlots) ggsave(paste0("PopulationGrowth",".tiff"), plot = last_plot(), device = "tiff", dpi = 300)


  pyramid <- matrix(NA, nrow = input$global_parameters$time_horizon, ncol = length(Cget_output_ex()$n_alive_by_ctime_age[1, ]) -
                      input$global_parameters$age0)

  for (year in 0:model_input$values$global_parameters$time_horizon - 1) pyramid[1 + year, ] <- Cget_output_ex()$n_alive_by_ctime_age[year +
                                                                                                                                       1, -(1:input$global_parameters$age0)]


  cat("Also, the ratio of the expected to observed population in years 10 and 20 are ", sum(Cget_output_ex()$n_alive_by_ctime_sex[10,
                                                                                                                                  ])/x[10, 2], " and ", sum(Cget_output_ex()$n_alive_by_ctime_sex[20, ])/x[20, 2])
  petoc()

  cat("Now evaluating the population pyramid\n")
  for (year in c(2015, 2025, 2034)) {
    cat("The observed population pyramid in", year, "is just drawn\n")
    x <- CanSim.052.0005[which(CanSim.052.0005[, "year"] == year & CanSim.052.0005[, "sex"] == "both"), "value"]
    #x <- c(x, rep(0, 111 - length(x) - 40))
    #barplot(x,  names.arg=40:110, xlab = "Age")
    #title(cex.main = 0.5, paste("Predicted Pyramid - ", year))
    dfPredicted <- data.frame (population = x * 1000, age = 40:100)


    # message("Predicted average age of those >40 y/o is", sum((input$global_parameters$age0:(input$global_parameters$age0 + length(x) -
    #                                                                                       1)) * x)/sum(x), "\n")
    # petoc()
    #
    # message("Simulated average age of those >40 y/o is", sum((input$global_parameters$age0:(input$global_parameters$age0 + length(x) -
    #                                                                                       1)) * x)/sum(x), "\n")
    # petoc()

    dfSimulated <- data.frame (population = pyramid[year - 2015 + 1, ], age = 40:110)
    dfSimulated$population <- dfSimulated$population * (-1) / settings$n_base_agents * 18179400 #rescaling population. There are 18179400 Canadians above 40

    p <- ggplot (NULL, aes(x = age, y = population)) + theme_tufte(base_size=14, ticks=F) +
         geom_bar (aes(fill = "Simulated"), data = dfSimulated, stat="identity", alpha = 0.5) +
         geom_bar (aes(fill = "Predicted"), data = dfPredicted, stat="identity", alpha = 0.5) +
         theme(axis.title=element_blank()) +
         ggtitle(paste0("Simulated vs. Predicted Population Pyramid in ", year)) +
         theme(legend.title=element_blank()) +
         scale_y_continuous(name="Population", labels = scales::comma) +
         scale_x_continuous(name="Age", labels = scales::comma)
    if (savePlots) ggsave(paste0("Population ", year,".tiff"), plot = last_plot(), device = "tiff", dpi = 300)

    plot(p)

  }


  message("This task is over... terminating")
  terminate_session()
}



