test_that("map.build constructors work", {
  data(iris)
  df<-subset(iris,select=-Species)
  labels<-subset(iris,select=Species)
  m <- map.build(df,labels,xdim=20,ydim=15,train=100000,seed=42)

  expect_equal(m$convergence > 0.8,TRUE)
  expect_equal(length(m$unique.centroids)<9,TRUE)
})
