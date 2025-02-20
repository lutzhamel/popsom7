\name{map.marginal}
\alias{map.marginal}
\title{Plot Marginal Distribution}
\description{Generate a plot that shows the marginal probability distribution of the neurons and data.}
\usage{
map.marginal(map,marginal)
}
\arguments{
  \item{map}{An object of type 'map'.}
  \item{marginal}{The name of a training data dimension or index.}
}
\value{No return value, called for side effects.}

\author{Lutz Hamel, Robert Tatoian}
\examples{
data(iris)

## set data frame and labels
df <- subset(iris,select=-Species)
labels <- subset(iris,select=Species)

## build a map
m <- map.build(df,labels,xdim=15,ydim=10,train=10000)

## display marginal distribution of dimension 1
map.marginal(m,1)
}
