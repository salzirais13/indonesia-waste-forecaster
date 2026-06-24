library(readxl)
data<-read_excel("D:/kuliah/wice/Book1.xlsx")

#KNNImputer
library(VIM)

data1<-kNN(data, variable="TS", k=6)

#eksport
library(writexl)
write_xlsx(data1, "D:/kuliah/wice/knn2.xlsx")

#interpolasi
library(dplyr)
library(zoo)
data3<-data %>%
  mutate(TS = na.approx(TS))
write_xlsx(data3, "D:/kuliah/wice/interpolasi2.xlsx")

#RNN
install.packages("rnn")
library(rnn)
x <- data1$Tahun
y <- data1$TS
# Membuat model RNN
model <- trainr(y,x)
model <- train

install.packages("neuralnet")
library(neuralnet)
data_input <- x
data_output <- y
# Membentuk data sesuai dengan format RNN
data <- data.frame(input = data_input, output = data_output)
# Membuat model RNN
model <- neuralnet(output ~ input, data = data, hidden = 1)
model
next_input <- 30
predicted_output <- predict(model, newdata = next_input)
predicted_output <-predict(model,newdata = 30)
cat("Prediksi nilai pada timestep berikutnya:", predicted_output, "\n")
