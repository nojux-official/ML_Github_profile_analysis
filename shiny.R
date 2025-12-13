library(shiny)
library(DT)
library(jsonlite)
library(httr)

# Define UI for application
ui <- fluidPage(
  # Application title
  titlePanel("CNN Image Classification - Github Profile Analysis"),
  
  sidebarLayout(
    sidebarPanel(
      h3("Model Predictions"),
      
      # JSON input for feature vectors
      textAreaInput(
        "query", 
        label = "Enter JSON with feature arrays (node IDs):",
        value = '{\n  "0": [1574, 3773, 3571, 2672, 2478],\n  "1": [1193, 376, 73, 290, 3129]\n}',
        rows = 10,
        cols = 40
      ),
      
      # Model selection
      selectInput("model", "Select Model:", choices = c("Latest")),
      
      # Submit button
      actionButton("predict_btn", "Get Predictions", class = "btn-primary"),
      
      hr(),
      
      # API status
      h4("API Status"),
      textOutput("api_status"),
      
      hr(),
      
      # Available models
      h4("Available Models"),
      textOutput("available_models")
    ),
    
    # Main panel with results
    mainPanel(
      # Results table
      h3("Prediction Results"),
      DT::dataTableOutput("predictions_table"),
      
      hr(),
      
      # Raw response
      h4("Raw API Response"),
      verbatimTextOutput("raw_response")
    )
  )
)

# Define server logic
server <- function(input, output, session) {
  
  # API configuration
  api_url <- "http://127.0.0.1:8000"
  
  # Check API status
  check_api_status <- function() {
    tryCatch({
      response <- GET(paste0(api_url, "/"))
      if (status_code(response) == 200) {
        return("✓ API is running")
      } else {
        return("✗ API error")
      }
    }, error = function(e) {
      return(paste("✗ API not accessible:", e$message))
    })
  }
  
  # Get available models
  get_available_models <- function() {
    tryCatch({
      response <- GET(paste0(api_url, "/models"))
      if (status_code(response) == 200) {
        data <- content(response, as = "parsed")
        return(data$models)
      }
    }, error = function(e) {
      return(NULL)
    })
  }
  
  # Update API status on load
  output$api_status <- renderText({
    check_api_status()
  })
  
  # Update available models
  output$available_models <- renderText({
    models <- get_available_models()
    if (!is.null(models) && length(models) > 0) {
      paste(models, collapse = "\n")
    } else {
      "No models found"
    }
  })
  
  # Update model choices
  observe({
    models <- get_available_models()
    if (!is.null(models) && length(models) > 0) {
      updateSelectInput(session, "model", choices = c("Latest", models))
    }
  })
  
  # Reactive predictions
  predictions_data <- eventReactive(input$predict_btn, {
    # Validate JSON input
    tryCatch({
      query <- input$query
      parsed_query <- fromJSON(query)
    }, error = function(e) {
      showNotification(paste("Invalid JSON:", e$message), type = "error")
      return(NULL)
    })
    
    if (is.null(parsed_query)) {
      return(NULL)
    }
    
    # Build request
    model_param <- if (input$model == "Latest") NULL else input$model
    
    # Call API
    tryCatch({
      showNotification("Sending request to API...", type = "message", duration = 2)
      
      request_body <- list(query = query)
      if (!is.null(model_param)) {
        request_body$model <- model_param
      }
      
      response <- POST(
        paste0(api_url, "/predict"),
        body = toJSON(request_body),
        content_type_json(),
        encode = "json"
      )
      
      if (status_code(response) == 200) {
        api_response <- content(response, as = "parsed")
        
        # Store raw response
        output$raw_response <<- renderPrint({
          toJSON(api_response, pretty = TRUE)
        })
        
        # Convert predictions to data frame
        if (!is.null(api_response$predictions) && length(api_response$predictions) > 0) {
          pred_list <- api_response$predictions
          
          # Create data frame from predictions
          df <- data.frame(
            image_id = names(pred_list),
            prediction = sapply(pred_list, function(x) x$prediction),
            probability = sapply(pred_list, function(x) x$probability),
            raw_output = sapply(pred_list, function(x) x$raw_output),
            stringsAsFactors = FALSE,
            row.names = NULL
          )
          
          showNotification("Predictions received!", type = "message", duration = 2)
          return(df)
        }
      } else {
        error_msg <- tryCatch(content(response)$error, error = function(e) "Unknown error")
        showNotification(paste("API Error:", error_msg), type = "error")
      }
      
      return(NULL)
    }, error = function(e) {
      showNotification(paste("Request failed:", e$message), type = "error")
      return(NULL)
    })
  })
  
  # Render predictions table
  output$predictions_table <- DT::renderDataTable({
    data <- predictions_data()
    if (!is.null(data)) {
      DT::datatable(
        data,
        options = list(pageLength = 10, scrollX = TRUE),
        rownames = FALSE
      )
    } else {
      DT::datatable(data.frame())
    }
  })
  
  # Initial raw response
  output$raw_response <- renderPrint({
    cat("Waiting for predictions...\n")
  })
}

# Run the application
shinyApp(ui = ui, server = server)