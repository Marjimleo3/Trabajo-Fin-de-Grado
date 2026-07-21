library(arrow)
library(corrplot)

df <- read_parquet("data/processed/modelizacion/db_final_codificada.parquet")
cor_matrix <- cor(as.data.frame(lapply(df, as.numeric)), use = "pairwise.complete.obs")

png("images/correlacion_lineal.png", width = 1800, height = 1600, res = 150)
corrplot(cor_matrix, method = "color", type = "lower", tl.cex = 0.7, tl.srt = 45,
         title = "Correlación lineal (Pearson)", mar = c(0, 0, 2, 0),
         addCoef.col = "black", number.cex = 0.45)
dev.off()
