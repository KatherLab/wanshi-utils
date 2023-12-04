# Load necessary libraries
library(tidyverse)
library(reshape2)
library(RColorBrewer)
library(ggplot2)

# Load the csv file
df <- read.csv('/home/jeff/PycharmProjects/wanshi-utils/wanshi/visualizations/crona/final.csv', stringsAsFactors = FALSE)

# Replace commas in numbers with dots to make them floats
df <- as.data.frame(lapply(df, function(x) gsub(",", ".", x)))

# Cast all percentages to float
for (col in names(df)[3:length(names(df))]) {
  df[[col]] <- as.numeric(df[[col]])
}

# Reshape the data for visualization
df_melted <- melt(df, id.vars = c("groups", "Guideline Item"))

# Create the heatmap
ggplot(df_melted, aes(x = variable, y = `Guideline Item`, fill = value)) +
  geom_tile() +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 90, vjust = 0.5, hjust=1)) +
  facet_grid(groups ~ .) +
  scale_fill_gradient(low = "white", high = "grey") +
  labs(x = "Guideline Type", y = "Guideline Item", fill = "Percentage", title = "Heatmap of Guideline Items") +
  theme(strip.text.y = element_text(angle = 0))