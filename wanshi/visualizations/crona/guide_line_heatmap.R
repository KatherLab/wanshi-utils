# Load the necessary libraries
library(ComplexHeatmap)
# Load circlize package
library(circlize)
# Read the data
data <- read.csv("/home/jeff/PycharmProjects/wanshi-utils/wanshi/visualizations/crona/finial2.csv", sep = ",")
# Change the column names to make them easier to work with
colnames(data) <- c("Groups", "GuidelineItem", "High", "Intermediate", "Low", "Overall","General","Specific")

# Convert the percentage strings to numbers
data$High <- as.numeric(sub(",", ".", data$High))
data$Intermediate <- as.numeric(sub(",", ".", data$Intermediate))
data$Low <- as.numeric(sub(",", ".", data$Low))
data$Overall <- as.numeric(sub(",", ".", data$Overall))
data$General <- as.numeric(sub(",", ".", data$General))
data$Specific <- as.numeric(sub(",", ".", data$Specific))

# Create a row names vector
rownames(data) <- data$GuidelineItem

# Specify the order of levels in the "Groups" column
data$Groups <- factor(data$Groups, levels = unique(data$Groups))

# Transpose the heatmap data and annotation data
heatmap_data <- t(data[, -(1:2)])
annotation_data <- data.frame(Groups = rownames(data))

# Define colors for each unique group
group_colors <- c('#e76254', '#f9b45e', '#d4e1cc', '#5a9ab7', '#1e466e')
# Define the number of rows in each group
group_size <- 5
# Create a grouping factor
row_groups <- cut(seq_len(nrow(heatmap_data)),
                  breaks = seq(1, nrow(heatmap_data), by = group_size),
                  include.lowest = TRUE,
                  labels = FALSE)
gap_size <- 2  # Adjust this value as needed for larger gaps
font_family <- "Helvetica"  # Change to "Arial" if you prefer

# Modify the Heatmap function to include row splits
ht_map <- Heatmap(
  heatmap_data,
  name = "mat",
  top_annotation = HeatmapAnnotation(
    foo = anno_block(
      gp = gpar(fill = group_colors,col="white"),
      labels = unique(data$Groups),
      labels_gp = gpar(col = "black", fontsize = 14),
    )
  ),

  column_split = data$Groups,
  row_split = c(1,1,1,3,2,2),          # Add gaps between row groups
  row_gap = unit(2.5, "mm"),
  show_column_names = TRUE,
  show_row_names = TRUE,
  column_gap = unit(2.5, "mm"),

  cluster_rows = FALSE,             # Enable row clustering
  cluster_columns = FALSE,         # Assuming you still don't want to cluster columns
  row_names_side = "left",
  row_dend_width = unit(5, "mm"),
  col = colorRamp2(c(0, 1), c("white", "black")),  # Remove border around boxes
  row_names_gp = gpar(fontsize = 13),
  column_names_gp = gpar(fontsize = 13),
  column_names_rot = 45,
)

# Draw the heatmap with gaps between row groups
draw(ht_map)
