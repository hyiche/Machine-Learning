## R commands for eigenface
rm(list = ls())
library(imager)
M = 6
Y = as.list(numeric(M))
#meanface = array(0, c(320, 243,1,1))
meanface = array(0, c(60, 70, 1, 1))
par(mfrow=c(3,3),mar=c(0,0,0,0))
# data faces
i=1
M = 7
for(i in 1:M){
  if(i<10) face = load.image(paste('/Users/steve/Desktop/博二/機器學習/第三次報告/data/subject0',i,'.jpg',sep='',collapse=''))
  if(i>9) face = load.image(paste('Users/steve/Desktop/博二/機器學習/第三次報告/data/subject',i,'.jpg',sep='',collapse=''))
  face = resize(face, 60, 70)
  plot(face, axes=F, xlab='', ylab='')
}

# training faces
i=1
M = 21
par(mfrow=c(4,4),mar=c(0,0,0,0))
for(i in 8:M){
  if(i<10) face = load.image(paste('/Users/steve/Desktop/博二/機器學習/第三次報告/data/subject0',i,'.jpg',sep='',collapse=''))
  if(i>9) face = load.image(paste('/Users/steve/Desktop/博二/機器學習/第三次報告/data/subject',i,'.jpg',sep='',collapse=''))
  face = resize(face, 60, 70)
  face = grayscale(face)
  meanface = meanface + face/M
  Y[[i]] = face
  plot(face, axes=F, xlab='', ylab='')
}

ccccc = c("我","我",2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7)

### 轉成數據
y=matrix(NA,dim(face)[1]*dim(face)[2]* dim(face)[4],M)
for(i in 8:M) y[,i-7]=c(Y[[i]])
y = y[,c(1:14)]

plot(load.image(paste('/Users/steve/Desktop/博二/機器學習/第三次報告/data/subject',30,'.jpg',sep='',collapse='')))

### test face
test = load.image(paste('/Users/steve/Desktop/博二/機器學習/第三次報告/data/subject',31,'.jpg',sep='',collapse=''))
test = resize(test, 60, 70)
test = c(test)

###比對   Method 1 距離法
kk = NULL
for(i in 1:14){
  kk = c(kk, sum((y[,i] - test)^2))
}
kkk = which(kk == min(kk))
cat(ccccc[kkk], "\t")

###比對   Method 2 PCA
# deviation matrix
A = y-rowMeans(y)
L = t(A)%*%A
eigL = eigen(L)
eigenface = A%*%eigL$ve
St = matrix(0, dim(face)[1]*dim(face)[2]* dim(face)[4], dim(face)[1]*dim(face)[2]* dim(face)[4])
for(i in 1:14) St = St + t(t((y[,i]-rowMeans(y))))%*%t(y[,i]-rowMeans(y))
Wopt = colSums(t(eigenface) %*% St %*% eigenface)
kkk = which(Wopt == max(Wopt))
cat(ccccc[kkk], "\t")


###比對   Method 2 Fisher
# deviation matrix
mu = colMeans(y)
SB = matrix(0, dim(face)[1]*dim(face)[2]* dim(face)[4], dim(face)[1]*dim(face)[2]* dim(face)[4])
for(i in 1:14) SB = SB + t(t((mu[i]-rowMeans(y))))%*%t(mu[i]-rowMeans(y))
SW = matrix(0, dim(face)[1]*dim(face)[2]* dim(face)[4], dim(face)[1]*dim(face)[2]* dim(face)[4])
for(i in 1:14) SW = SW + t(t((y[,i]-mu[i])))%*%t(y[,i]-mu[i])

Wopt = colSums((t(eigenface) %*% SB %*% eigenface)/(t(eigenface) %*% SW %*% eigenface))
kkk = which(Wopt == max(Wopt))
cat(ccccc[kkk], "\t")



## R commands for eigenface
rm(list = ls())
library(imager)
M = 6
Y = as.list(numeric(M))
#meanface = array(0, c(320, 243,1,1))
meanface = array(0, c(30, 30, 1, 1))
par(mfrow=c(3,3),mar=c(0,0,0,0))
# data faces
i=1
M = 7
for(i in 1:M){
  if(i<10) face = load.image(paste('/Users/steve/Desktop/博二/機器學習/第三次報告/data/subject0',i,'.jpg',sep='',collapse=''))
  if(i>9) face = load.image(paste('Users/steve/Desktop/博二/機器學習/第三次報告/data/subject',i,'.jpg',sep='',collapse=''))
  face = resize(face, 30, 30)
  plot(face, axes=F, xlab='', ylab='')
}

# training faces
i=1
M = 21
par(mfrow=c(4,4),mar=c(0,0,0,0))
for(i in 8:M){
  if(i<10) face = load.image(paste('/Users/steve/Desktop/博二/機器學習/第三次報告/data/subject0',i,'.jpg',sep='',collapse=''))
  if(i>9) face = load.image(paste('/Users/steve/Desktop/博二/機器學習/第三次報告/data/subject',i,'.jpg',sep='',collapse=''))
  face = resize(face, 30, 30)
  face = grayscale(face)
  meanface = meanface + face/M
  Y[[i]] = face
  plot(face, axes=F, xlab='', ylab='')
}

### 轉成數據
y=matrix(NA,dim(face)[1]*dim(face)[2]* dim(face)[4],M)
for(i in 8:M) y[,i-7]=c(Y[[i]])
y = y[,c(1:14)]

plot(load.image(paste('/Users/steve/Desktop/博二/機器學習/第三次報告/data/subject',31,'.jpg',sep='',collapse='')))

### test face
test = load.image(paste('/Users/steve/Desktop/博二/機器學習/第三次報告/data/subject',31,'.jpg',sep='',collapse=''))
test = resize(test, 30, 30)
test = c(test)


###比對   Kernel PCA
K = y %*% t(y)
ones = 1/nrow(y)* matrix(1, nrow(y), nrow(y))
K_norm = K - ones %*% K - K %*% ones + ones %*% K %*% ones
res = eigen(K)
V = res$vectors
D = diag(res$values)
Y = K %*% V
St = matrix(0, dim(face)[1]*dim(face)[2]* dim(face)[4], dim(face)[1]*dim(face)[2]* dim(face)[4])
for(i in 1:14) St = St + t(t((y[,i]-rowMeans(y))))%*%t(y[,i]-rowMeans(y))
Wopt = colSums(t(Y) %*% St %*% Y)
kkk = which(Wopt == max(Wopt))
cat(ccccc[kkk], "\t")



