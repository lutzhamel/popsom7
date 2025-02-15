test_that("map.predict function works", {
  data(iris)
  df<-subset(iris,select=-Species)
  labels<-subset(iris,select=Species)
  m <- map.build(df,labels,xdim=15,ydim=10,train=10000,seed=42)
  p <- map.predict(m,df)
  # print(p)

  # spot check predictions with high confidence
  # for correct prediction
  for (i in 1:150){
    if (p[i,2] > 0.9) {
      expect_equal(p[i,1],as.character(labels[i,1]))
      break
    }
  }
})
