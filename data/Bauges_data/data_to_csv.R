load("Bauges_data.Rdata")

write.csv(Bauges_data[,7:(ncol(Bauges_data)-2)],"Bauges_data.csv", 
          row.names = TRUE, col.names = TRUE)


